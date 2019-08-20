'''
This module handles the main logic of the game
'''
import os
import re
import time
import pandas as pd
from python.utils import *
import python.drumsynth as drumsynth
from scipy.io import wavfile
import pickle


###
def soundcheckDrum(drumkit_path, drumId):
    """
    Perform souncheck on a drum and store the audio
    :param drumkit_path: Drumkit folder
    :param drumId: int, Name of the drum
    :return: None
    """
    buffer = drumsynth.record_part(100)
    print(buffer.shape, buffer.min(), buffer.max())
    wavfile.write('{}/drum{}.wav'.format(drumkit_path, drumId),rate=SAMPLE_RATE, data=buffer)

def initKitBG(drumkit_path, L=10, drumwise=True, method='NMFD'):
    """
    Initialize drumkit from completed soundcheck audio. Finds prior templates for source separation,
    recalculates thresholds for peak picking, saves information to drum objects
    and stores the drumkit as a pickled file.
    :param drumkit_path: Folder containing soundcheck audio
    :param K: int, max number prior templates per drum
    :param L: int, number of signal frames to use in templates
    :param drumwise: Boolean, perform drumwise peak picking threshold recalculation
    or use same threshold for all drums
    :param method: 'NMFD' or 'NMF', the source separation approach to use
    :return: None
    """
    global drumkit
    drumkit=[]
    #read drums from folder
    kit = [f for f in os.listdir(drumkit_path) if not f.startswith('.')]
    kit.sort()
    for i in range(len(kit)):
        # HACK!!! remove when you have the time!!!
        if (kit[i][:4] != 'drum' or len(kit[i]) > 10): continue
        #set buffer
        sr,buffer = wavfile.read("{}/{}".format(drumkit_path, kit[i]), mmap=True)
        #preprocess
        filt_spec = stft(buffer)

        #find onsets
        peaks = getPeaksFromBuffer(filt_spec, N_PEAKS)
        if(peaks.shape[0]<N_PEAKS):
            raise Exception('drum nr. {} does not have the correct number of peaks, please re check'.format(i))
        # mean of cluster center
        freqtemps = findDefBins(peaks, filt_spec, L)

        # use OPTICS. Defines parameters for automatic clustering automatically!
        # freqtemps = findDefBinsOPTICS(peaks, filt_spec, L)

        # Store the start locations of different drums and concatenate soundcheck file
        if i == 0:
            filt_spec_all = filt_spec
            shifts = [0]
        else:
            shift = filt_spec_all.shape[0]
            filt_spec_all = np.vstack((filt_spec_all, filt_spec))
            shifts.append(shift)

        # put drums in a list of drums
        drumkit.append(
            Drum(name=[int(re.findall('\d+', kit[i])[0])], peaks=peaks, heads=freqtemps[0], tails=freqtemps[1],
                 threshold=0,
                 midinote=MIDINOTES[int(re.findall('\d+', kit[i])[0])]))

    # recalculate all threshods for peak picking
    recalculate_thresholds(filt_spec_all, shifts, drumkit, drumwise=drumwise, method=method)

    # Pickle the important data
    pickle.dump(drumkit, open("{}/pickledDrumkit.drm".format(drumkit_path), 'wb'))


def loadKit(drumkit_path):
    """
    Loads drumkit data to memory
    :param drumkit_path: path to drumkit
    :return: None
    """
    global drumkit
    drumkit = pickle.load(open("{}/pickledDrumkit.drm".format(drumkit_path), 'rb'))
    return drumkit


def playLive(drumkit_path, thresholdAdj=0.0,part_length=30, saveAll=False, createMidi=False):
    """
    #Records one drum part of the player transcribes it and stores to a csv or also to a midi file
    The audio is not saved.
    :param drumkit_path: Folder where parts are stored
    :param thresholdAdj: float, adjust all drums thresholds if needed
    :param saveAll: Boolean, weather to save every transcription in a separate file
    :param createMidi: Boolean, weather to save a midi file of the transcription
    :return: String, float, Filename of the recorded part and
    average tempo (relative to default tempo) of the part
    """

    # Record a take
    try:
        buffer = drumsynth.record_part(part_length)
    except Exception as e:
        print('liveplay:', e)
    # transcribe the take
    plst, deltaTempo = processLiveAudio(liveBuffer=buffer, drumkit=drumkit,
                                        iters=96, method='NMFD_iters', thresholdAdj=thresholdAdj)
    # Make an annotation of the separate drums hit times
    times = []
    bintimes = []
    for i in plst:
        hits = i.get_hits()
        if not len(hits):
            continue
        binhits = i.get_hits()
        hits = frame_to_time(hits)
        labels = np.full(len(hits), i.get_midinote(), np.int64)
        binlabels = np.full(len(binhits), i.get_name(), np.int64)
        inst = zip(hits, labels)
        bininst = zip(binhits, binlabels)
        times.extend(inst)
        bintimes.extend(bininst)
    if not len(times):
        print('no hits found from audio!')
        return False
    times.sort()
    bintimes.sort()
    bintimes = mergerowsandencode(bintimes)
    df = pd.DataFrame(times, columns=['time', 'inst'])
    df['duration'] = pd.Series(np.full((len(times)), 0, np.int64))
    df['vel'] = pd.Series(np.full((len(times)), 127, np.int64))
    bindf = pd.DataFrame(bintimes, columns=['inst'])

    if saveAll:
        fileName = '{}/takes/testbeat{}.csv'.format(drumkit_path, time.time())
    else:
        fileName = '{}/takes/lastTake.csv'.format(drumkit_path)

    bindf.to_csv(fileName, index=True, header=False, sep="\t")
    df = df[df.time != 0]
    print('done!')

    if createMidi:
        import madmom
        madmom.io.midi.write_midi(df.values, 'midi_testit_.mid')
        generated = splitrowsanddecode(bintimes)
        gen = pd.DataFrame(generated, columns=['time', 'inst'])
        gen.to_csv('generated_enc_dec0.csv', index=False, header=None, sep='\t')
        # change to time and midinotes
        gen['time'] = frame_to_time(gen['time'])
        gen['inst'] = to_midinote(gen['inst'])
        gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
        gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))
        madmom.io.midi.write_midi(gen.values, 'midi_testit_enc_dec0.mid')

    return fileName, float(deltaTempo)

def processLiveAudio(liveBuffer=None,spectrogram=None, drumkit=None, iters=0, method='NMFD', thresholdAdj=0.):
    """
    main logic for source separation, onset detection and tempo extraction and quantization
    :param liveBuffer: numpy array, the source audio
    :param drumkit: list of drums
    :param quant_factor: float, amount of quantization (change to boolean)
    :param iters: int, number of runs of nmfd for bagging separation
    :param method: The source separation method, 'NMF' or 'NMFD
    :param thresholdAdj: float, adjust the onset detection thresholds, one value for all drums.
    :return: list of drums containing onset locations in hits field and mean tempo of the take
    """

    if liveBuffer is not None:
        filt_spec = stft(liveBuffer)
    elif spectrogram is not None:
        filt_spec=spectrogram
    else:
        assert 'You must provide either a processed spectrogram or an audio file location'

    stacks = 1
    Wpre, total_heads=get_Wpre(drumkit)
    for i in range(int(stacks)):
        if method == 'NMFD' or method == 'ALL':
            H, Wpre, err1 = nmfd.NMFD(filt_spec.T, iters=iters, Wpre=Wpre, include_priors=True,
                                      n_heads=total_heads, hand_break=True)
        elif method == 'NMFD_iters':
            H, Wpre, err1 = nmfd.NMFD(filt_spec.T, iters=iters, Wpre=Wpre, include_priors=True,
                                      n_heads=total_heads, hand_break=False)
        if method == 'NMF' or method == 'ALL':
            H, err2 = nmfd.semi_adaptive_NMFB(filt_spec.T, Wpre=Wpre, iters=iters,
                                              n_heads=total_heads, hand_break=True)
        if method == 'ALL':
            errors = np.zeros((err1.size, 2))
            errors[:, 0] = err1
            errors[:, 1] = err2
        if i == 0:
            WTot, HTot = Wpre, H
        else:
            WTot += Wpre
            HTot += H

    H = (HTot) / stacks

    onsets = np.zeros(H[0].shape[0])
    total_heads = 0

    for i in range(len(drumkit)):
        heads = drumkit[i].get_heads()
        K1 = heads.shape[2]
        ind = total_heads
        for k in range(K1):
            index = ind + k
            HN = H[index]
            HN = HN / HN.max()
            if k == 0:
                H0 = HN
            else:
                H0 += HN
            total_heads += 1
        if i == 0:
            onsets = H0
        else:
            onsets = onsets + H0
        peaks = onset_detection.pick_onsets(H0, threshold=drumkit[i].get_threshold() + thresholdAdj)
        # remove extrahits used to level peak picking algorithm:
        peaks = peaks[np.where(peaks < filt_spec.shape[0] - 1)]
        drumkit[i].set_hits(peaks)
    return drumkit, Q_HOP/HOP_SIZE


#
#
# Debug code below


def extract_training_material(audio_folder, annotation_folder, train_audio_takes, train_annotation):
    print('Extracting templates.', end='', flush=True)
    global drumkit
    drumkit = []
    # 0,1,2 kick, snare, hh
    kick_heads = []
    kick_tails = []
    snare_heads = []
    snare_tails = []
    hihat_heads = []
    hihat_tails = []

    def get_window_edge(frame):
        return int(frame)

    for f in train_annotation:
        print('.', end='', flush=True)
        print(audio_folder + f.split('.')[0] + '.wav')
        #import librosa.core
        #file, sr=librosa.core.load(audio_folder + f.split('.')[0] + '.wav')
        #file=librosa.core.resample(file, sr, 44100)
        #wavfile.write(audio_folder + f.split('.')[0] + ' res.wav',44100, file)
        _,buffer = wavfile.read(audio_folder + f.split('.')[0] + '.wav')
        if len(buffer.shape) > 1:
            buffer = buffer[:, 0] + buffer[:, 1]
        filt_spec = stft(buffer)
        hits = pd.read_csv(annotation_folder + f, sep="\t", header=None)
        hits[0] = time_to_frame(hits[0], sr=44100, hop_length=HOP_SIZE)
        kicks_from_wav = 0
        snares_from_wav = 0
        hihats_from_wav = 0
        for i in range(1, hits.shape[0] - 1):
            if hits.iloc[i - 1][0] < hits.iloc[i][0] - 20 and hits.iloc[i][0] + 20 < hits.iloc[i + 1][0]:
                if hits.iloc[i][0] + 20 < filt_spec.shape[0]:
                    if int(hits.iloc[i][1]) == 0:
                        ind = get_window_edge(hits.iloc[i][0])
                        if kicks_from_wav <= 16:
                            kicks_from_wav += 1
                            kick_heads.append(filt_spec[ind:ind + 10])
                            kick_tails.append(filt_spec[ind + 10:ind + 20])
                    if hits.iloc[i][1] == 1:
                        ind = get_window_edge(hits.iloc[i][0])
                        if snares_from_wav <= 16:
                            snares_from_wav += 1
                            snare_heads.append(filt_spec[ind:ind + 10])
                            snare_tails.append(filt_spec[ind + 10:ind + 20])
                    if hits.iloc[i][1] == 2:
                        ind = get_window_edge(hits.iloc[i][0])
                        if hihats_from_wav <= 16:
                            hihats_from_wav += 1
                            hihat_heads.append(filt_spec[ind:ind + 10])
                            hihat_tails.append(filt_spec[ind + 10:ind + 20])

    def norm(a):
        a = np.array(a)
        a = np.reshape(a, (a.shape[0], -1))
        return a
        a = a - a.min()
        return a / a.max()
    print('.')
    temps_kick = create_templates(matrices=[norm(kick_heads), norm(kick_tails)])
    temps_snare = create_templates(matrices=[norm(snare_heads), norm(snare_tails)])
    temps_hats = create_templates(matrices=[norm(hihat_heads), norm(hihat_tails)])
    drumkit.append(Drum(name=[0],peaks=np.arange(0, len(kick_heads) * 20, 20), heads=temps_kick[0],
                      tails=temps_kick[1],
                      threshold=.4,#RMBA0.65
                      midinote=MIDINOTES[0]))
    drumkit.append(Drum(name=[1], peaks=np.arange(0, len(snare_heads) * 20, 20), heads=temps_snare[0],
                      tails=temps_snare[1],
                      threshold=0.7,#RMBA0.7
                      midinote=MIDINOTES[1]))
    drumkit.append(Drum(name=[2],peaks=np.arange(0, len(hihat_heads) * 20, 20), heads=temps_hats[0],
                      tails=temps_hats[1],
                      threshold=0.325,#RMBA0.25
                      midinote=MIDINOTES[2]))
    #shifts = [0]
    #filt_spec_all = np.array([val for pair in zip(kick_heads, kick_tails) for val in pair])
    #shifts.append(filt_spec_all.shape[0] * 10)
    #filt_spec_all = np.vstack((filt_spec_all, [val for pair in zip(snare_heads, snare_tails) for val in pair]))
    #shifts.append(filt_spec_all.shape[0] * 10)
    #filt_spec_all = np.vstack(
    #    (filt_spec_all, [val for pair in zip(hihat_heads, hihat_tails) for val in pair]))
    #filt_spec_all = np.reshape(filt_spec_all, (-1, 48))
    #print('Recalculating thresholds')
    #recalculate_thresholds(filt_spec_all, shifts, drumkit, drumwise=True, method='NMFD')
    # for i in range(3):
    #    index=i*2
    #    drums.append(
    #            Drum(name=[i], highEmph=0, peaks=[], heads=temps[index], tails=temps[index+1],
    #                 threshold=1/3.,
    #                 midinote=MIDINOTES[i], probability_threshold=1))
    #for i in drumkit:
    #    i.set_threshold(i.get_threshold()+.01)
    pickle.dump(drumkit, open("{}/pickledDrumkit.drm".format('.'), 'wb'))
    print('\ntotal: ', len(kick_heads), len(snare_tails), len(hihat_heads))


def make_drumkit():
    pass


def run_folder(audio_folder, annotation_folder):
    audio = [f for f in os.listdir(audio_folder) if not f.startswith('.')]
    annotation = [f for f in os.listdir(annotation_folder) if not f.startswith('.')]
    #annotation = [f for f in annotation if not f.endswith('train.wav')]
    take_names = set([i.split(".")[0] for i in annotation])
    audio_takes = np.array(sorted([i + '.wav' for i in take_names]))
    annotation = np.array(sorted(annotation))
    # np.random.seed(0)
    train_ind = np.random.choice(len(annotation), int(len(annotation) / 3))
    mask = np.zeros(annotation.shape, dtype=bool)
    mask[train_ind] = True
    train_audio_takes = audio_takes#[~mask]
    test_audio_takes = audio_takes#[mask]
    train_annotation = annotation#[~mask]
    test_annotation = annotation#[mask]
    #drumkit=pickle.load(open("{}/pickledDrumkit.drm".format('.'), 'rb'))
    #for i in drumkit:
    #    i.set_threshold(i.get_threshold()+.7)
    extract_training_material(audio_folder, annotation_folder, train_audio_takes, train_annotation)
    loadKit('.')
    sum = [0, 0, 0]
    for i in range(len(test_annotation)):
        res = test_run(annotated=True,
                       files=[audio_folder + test_audio_takes[i], annotation_folder + test_annotation[i]], method='NMFD')
        sum[0] += res[0]
        sum[1] += res[1]
        sum[2] += res[2]
    print('precision=', sum[0] / len(test_annotation))
    print('recall=', sum[1] / len(test_annotation))
    print('f-score=', sum[2] / len(test_annotation))
    return sum[0] / len(test_annotation), sum[1] / len(test_annotation), sum[2] / len(test_annotation)


def test_run(file_path=None, annotated=False, files=[None, None], method='NMFD', quantize=0., skip_secs=0.):
    prec, rec, fsc = [0., 0., 0.]
    if files[0] is not None:
        audio_file_path = files[0]
        if annotated and files[1] is not None:
            annot_file_path = files[1]
    else:
        audio_file_path = "{}drumBeatAnnod.wav".format(file_path)
        if annotated:
            annot_file_path = "{}midiBeatAnnod.csv".format(file_path)
    # print(audio_file_path)
    #print('.', end='', flush=True)
    try:
        #buffer = madmom.audio.Signal(audio_file_path, frame_size=FRAME_SIZE,
        #                             hop_size=HOP_SIZE)
        sr, buffer = wavfile.read(audio_file_path, mmap=True)
        # print(buffer.shape)
        if len(buffer.shape) > 1:
            buffer = buffer[:, 0] + buffer[:, 1]
    except Exception as e:
        print(e)
        print('jotain meni vikaan!')

    fs = np.zeros((256, 3))

    skip_secs = int(44100 * skip_secs)  # train:17.5
    for n in range(1):

        # initKitBG(filePath, 9, K=n)#, rm_win=n, bs_len=350)
        # t0 = time.time()
        plst, i = processLiveAudio(liveBuffer=buffer[skip_secs:],
                                   drumkit=drumkit, iters=75, method=method)
        # print('\nNMFDtime:%0.2f' % (time.time() - t0))
        # Print scores if annotated
        for k in range(1):
            if (annotated):
                # print f-score:
                # print('\n\n')
                hits = pd.read_csv(annot_file_path, sep="\t", header=None)
                precision, recall, fscore, true_tot = 0, 0, 0, 0
                for i in plst:
                    predHits = frame_to_time(i.get_hits())
                    # NMF need this coefficient to correct estimates
                    b = -0.002#-.02615#02625#02625#025#01615#SMT -0.002 for NMFD!
                    actHits = hits[hits[1] == i.get_name()[0]]
                    actHits = actHits.iloc[:, 0]
                    trueHits = k_in_n(actHits.values + b, predHits, window=0.025)
                    prec, rec, f_drum = f_score(trueHits, predHits.shape[0], actHits.shape[0])
                    # Multiply by n. of hits to get real f-score in the end.
                    precision += prec * actHits.shape[0]
                    recall += rec * actHits.shape[0]
                    fscore += (f_drum * actHits.shape[0])
                    true_tot += actHits.shape[0]
                    print(prec, rec, f_drum)
                    # add_to_samples_and_dictionary(i.drum, buffer, i.get_hits())
                prec = precision / true_tot
                fs[n, 0] = (precision / true_tot)
                rec = recall / true_tot
                fs[n, 1] = (recall / true_tot)
                fsc = fscore / true_tot
                fs[n, 2] = (fscore / true_tot)
                # return [prec, rec, fsc]
                print('Precision: {}'.format(prec))
                print('Recall: {}'.format(rec))
                print('F-score: {}'.format(fsc))
    return [prec, rec, fsc]
    
#todo Does the part lenght affect the error_limit?!?!?!?!?
#do we want fixed iterations, it's now fixed!!!
#do testing!
def debug():
    # debug
    # initKitBG('Kits/mcd2/',8,K)
    K=1
    #file = './Kits/mcd_pad/'
    method = 'NMFD'
    #file='../DXSamplet/'
    #initKitBG(file,K=K, drumwise=True, method=method)
    ## print('Kit init processing time:%0.2f' % (time.time() - t0))
    #loadKit(file)
    #print(test_run(file_path=file, annotated=True, method=method, quantize=0., skip_secs=0))
#
    #drumsynth.createWav('testbeat3.csv', 'sysAnnodQuantizedPart.wav', addCountInAndCountOut=False,
    #                    deltaTempo=1,
    #                    countTempo=1)
    #return
    prec_tot = 0
    rec_tot = 0
    fscore_tot = 0
    rounds = 1
    for i in range(rounds):
        #prec, rec, fscore = run_folder(audio_folder='../../libtrein/SMT_DRUMS/audio/',
        #                               annotation_folder='../../libtrein/SMT_DRUMS/annotations/')
        #prec, rec, fscore=run_folder(audio_folder='../../libtrein/ENST_Drums/audio_drums/', annotation_folder='../../libtrein/ENST_Drums/annotation/')
        #prec, rec, fscore=run_folder(audio_folder='../../libtrein/ENST_2/audio_drum_acc/', annotation_folder='../../libtrein/ENST_2/annotations/')

        prec, rec, fscore=run_folder(audio_folder='../../libtrein/SMT_DRUMS/audio/', annotation_folder='../../libtrein/SMT_DRUMS/annotations/')
        #prec, rec, fscore=run_folder(audio_folder='../../libtrein/rbma_13/audio/', annotation_folder='../../libtrein/rbma_13/annotations/drums/')

        prec_tot += prec
        rec_tot += rec
        fscore_tot += fscore
    print('Total numbers for {} rounds:\n'.format(rounds))
    print('precision=', prec_tot / rounds)
    print('recall=', rec_tot / rounds)
    print('f-score=', fscore_tot / rounds)
    # run_folder(audio_folder='../../libtrein/rbma_13/audio/', annotation_folder='../../libtrein/rbma_13/annotations/drums/')

    # testOnsDet(file, alg=0, ppAlg=0)
    # initKitBG('../DXSamplet/',9,K=K,rm_win=6)
    # loadKit('../trainSamplet/')
    # testOnsDet('../trainSamplet/', alg=0)
    # play('../trainSamplet/', K=K)
    # from math import factorial
    # def comb(n, k):
    #    return factorial(n) / factorial(k) / factorial(n - k)
    # hitsize=[]
    # for r in range(2,25):
    #    hitsize.append(2*(comb(r,2))+32)
    # showEnvelope(hitsize)


if __name__ == "__main__":
    debug()


#RMBA_tuned precision= 0.5285422876382673
#RMBA_tuned recall= 0.5327085855254259
#RMBA_tuned f-score= 0.49552240623032934