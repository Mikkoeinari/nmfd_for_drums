'''
This module handles the main logic of the game
'''
import os
import pickle
import time
import re
import pandas as pd
from scipy.io import wavfile
import pydub
from pydub import effects
import python.wide_crnn as vs
from python.utils import *
import matplotlib.pyplot as plt
from madmom.evaluation.onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation

def initKitBG(drumkit_path, K=1, L=10, drumwise=True, method='NMFD'):
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
    filt_spec_all=None
    print(kit)
    for i in range(len(kit)):
        # HACK!!! remove when you have the time!!!
        if (kit[i]=='pickledDrumkit.drm'): continue
        drum_name=kit[i].split('_')[1].split('.')[0]
        drum_id=name_to_index(drum_name)
        print('soundcheck:',drum_name, drum_id)
        #read file
        audio_segment = pydub.AudioSegment.from_mp3("{}/{}".format(drumkit_path, kit[i]))
        buffer = np.array(audio_segment.get_array_of_samples())
        #preprocess
        filt_spec = stft(buffer)
        #find onsets
        peaks = getPeaksFromBuffer(filt_spec, N_PEAKS)
        if(peaks.shape[0]<N_PEAKS):
            raise Exception('drum nr. {} does not have the correct number of peaks, please re check'.format(i))
        # mean of cluster center
        freqtemps = findDefBins(peaks, filt_spec, L)

        # Store the start locations of different drums and concatenate soundcheck file
        if filt_spec_all is None:
            filt_spec_all = filt_spec[:peaks[-1]+300,:]
            shifts = [0]
        else:
            shift = filt_spec_all.shape[0]
            filt_spec_all = np.vstack((filt_spec_all, filt_spec[:peaks[-1]+300,:]))
            shifts.append(shift)
        # put drums in a list of drums
        drumkit.append(
            Drum(name=[drum_id], peaks=peaks, heads=freqtemps[0], tails=freqtemps[1],
                 threshold=0,
                 midinote=None))
    print('recalculating thresholds')
    # recalculate all threshods for peak picking
    recalculate_thresholds(filt_spec_all, shifts, drumkit, drumwise=True, method=method)

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


def process_drum_data(liveBuffer=None, spectrogram=None, drumkit=None, iters=0, method='NMFD', thresholdAdj=0.):
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
        filt_spec = spectrogram
    else:
        assert 'You must provide either a processed spectrogram or an audio file location'

    stacks = 1
    Wpre, total_heads = get_Wpre(drumkit)
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
    return drumkit, Q_HOP / HOP_SIZE


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
        _, buffer = wavfile.read(audio_folder + f.split('.')[0] + '.wav')
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
    print(temps_kick[1].shape)
    drumkit.append(Drum(name=[0], peaks=np.arange(0, len(kick_heads) * 20, 20), heads=temps_kick[0],
                        tails=temps_kick[1],
                        threshold=.4,  # RMBA0.65
                        midinote=MIDINOTES[0]))
    drumkit.append(Drum(name=[1], peaks=np.arange(0, len(snare_heads) * 20, 20), heads=temps_snare[0],
                        tails=temps_snare[1],
                        threshold=0.7,  # RMBA0.7
                        midinote=MIDINOTES[1]))
    drumkit.append(Drum(name=[2], peaks=np.arange(0, len(hihat_heads) * 20, 20), heads=temps_hats[0],
                        tails=temps_hats[1],
                        threshold=0.325,  # RMBA0.25
                        midinote=MIDINOTES[2]))
    # shifts = [0]
    # filt_spec_all = np.array([val for pair in zip(kick_heads, kick_tails) for val in pair])
    # shifts.append(filt_spec_all.shape[0] * 10)
    # filt_spec_all = np.vstack((filt_spec_all, [val for pair in zip(snare_heads, snare_tails) for val in pair]))
    # shifts.append(filt_spec_all.shape[0] * 10)
    # filt_spec_all = np.vstack(
    #    (filt_spec_all, [val for pair in zip(hihat_heads, hihat_tails) for val in pair]))
    # filt_spec_all = np.reshape(filt_spec_all, (-1, 48))
    # print('Recalculating thresholds')
    # recalculate_thresholds(filt_spec_all, shifts, drumkit, drumwise=True, method='NMFD')
    # for i in range(3):
    #    index=i*2
    #    drums.append(
    #            Drum(name=[i], highEmph=0, peaks=[], heads=temps[index], tails=temps[index+1],
    #                 threshold=1/3.,
    #                 midinote=MIDINOTES[i], probability_threshold=1))
    # for i in drumkit:
    #    i.set_threshold(i.get_threshold()+.01)
    pickle.dump(drumkit, open("{}/pickledDrumkit.drm".format('.'), 'wb'))
    print('\ntotal: ', len(kick_heads), len(snare_tails), len(hihat_heads))


def run_folder(audio_folder, annotation_folder, soundcheck_folder=None, method='NMFD'):
    audio = [f for f in os.listdir(audio_folder) if not f.startswith('.')]
    annotation = [f for f in os.listdir(annotation_folder) if not f.startswith('.')]
    # annotation = [f for f in annotation if not f.endswith('train.wav')]
    take_names = set([i.split(".")[0] for i in annotation])
    audio_takes = np.array(sorted([i + '.wav' for i in take_names]))
    audio_takes=np.array(sorted(audio))
    annotation = np.array(sorted(annotation))
    # np.random.seed(0)
    train_ind = np.random.choice(len(annotation), int(len(annotation) / 3))
    mask = np.zeros(annotation.shape, dtype=bool)
    mask[train_ind] = True
    train_audio_takes = audio_takes  # [~mask]
    test_audio_takes = audio_takes  # [mask]
    train_annotation = annotation  # [~mask]
    test_annotation = annotation  # [mask]
    if soundcheck_folder is not None:
        initKitBG(soundcheck_folder,K=1, L=10,drumwise=True, method=method)
    elif soundcheck_folder is None:
        extract_training_material(audio_folder, annotation_folder, train_audio_takes, train_annotation)
    loadKit(soundcheck_folder)
    sum = [0, 0, 0]
    for i in range(len(test_annotation)):
        res = test_run(annotated=True,
                       files=[audio_folder + test_audio_takes[i], annotation_folder + test_annotation[i]],
                       method=method)
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
    # print('.', end='', flush=True)
    try:
        # buffer = madmom.audio.Signal(audio_file_path, frame_size=FRAME_SIZE,
        #                             hop_size=HOP_SIZE)
        #sr, buffer = wavfile.read(audio_file_path, mmap=True)
        audio_segment = pydub.AudioSegment.from_mp3(audio_file_path)
        # audio_segment=effects.normalize(audio_segment)
        buffer = np.array(audio_segment.get_array_of_samples())
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
        plst, i = process_drum_data(liveBuffer=buffer[skip_secs:],
                                    drumkit=drumkit, iters=75, method=method)

        # print('\nNMFDtime:%0.2f' % (time.time() - t0))
        # Print scores if annotated
        for k in range(1):
            if (annotated):
                # print f-score:
                # print('\n\n')
                hits = pd.read_csv(annot_file_path, sep="\t", header=None)
                hits.ix[:,1]=to_index(hits.ix[:,1])
                precision, recall, fscore, true_tot = 0, 0, 0, 0
                for i in plst:
                    print('drum:', index_to_name(i.get_name()[0]))
                    predHits = frame_to_time(i.get_hits())
                    # NMF need this coefficient to correct estimates
                    b =0# -0.002  # -.02615#02625#02625#025#01615#SMT -0.002 for NMFD!
                    actHits = hits[hits[1] == i.get_name()[0]]
                    actHits = actHits.iloc[:, 0]
                    trueHits = k_in_n(actHits.values + b, predHits, window=0.02)
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


def score_rnn_result(odf, annotation_file):

    try:
        hits = pd.read_csv(annotation_file, sep="\t", header=None)
    except:
        print('no drum hits present')
        return None
    tp, fp, fn, fsc_mean = 0, 0, 0, 0
    thresholds=list(np.zeros(18))
    #thresholds = [0.50222222, 0.47333333, 0.1,        0.03333333 ,0.12222222, 0.15111111,
    #              0.14888889, 0.35333333 ,0.09111111 ,0.16888889 ,0.09777778, 0.08666667,
    #              0.00666667 ,0.32888889 ,0.1 ,       0.03111111, 0.01111111 ,0.]
    #thresholds=[0.59534884, 0.56511628, 0.36976744]
    #thresholds = [0.24666667, 0.43333333, 0.17777778, 0.07333333, 0.24,       0.28222222,
    #              0.24888889, 0.18222222, 0.04444444, 0.24888889, 0.11111111, 0.08,
    #              0.03111111, 0.33555556, 0.08222222, 0.03555556, 0.02,       0.01]
    good_thresh = np.zeros((odf.shape[1]))
    drumwise_fscore = np.zeros((odf.shape[1]))
    mad_score=[]
    #vogl_to_mine = {0: 0, #bd
    #                1: 1, #sn
    #                2: -1,
    #                3: -1,
    #                4: 5, #ft
    #                5: -1,
    #                6: 4, #tt
    #                7: 2, #hh
    #                8: 8, #hhpedal
    #                9: 3, ##hhopen
    #                10: -1,
    #                11: 6, #rd
    #                12: -1,
    #                13: 7, #cr
    #                14: -1,
    #                15: -1,
    #                16: -1,
    #                17: -1,
    #                -1: -1}
    for i in range(odf.shape[1]):
        ons=([],[])
        drum_scores = [0, 0, 0]
        max_fsc=0.
        for l in range(10):
            threshold = 0. + l / 10.
        #threshold=thresholds[i]
            drum_odf = odf[:, i]/max(odf[:, i])
            peaks = onset_detection.pick_onsets(drum_odf, threshold=threshold, w=3.5)
            predHits = frame_to_time(peaks)
            actHits = hits[hits[1] == i]#vogl_to_mine.get(i)]
            actHits = actHits.iloc[:, 0]

            true_positive = k_in_n(actHits.values, predHits, window=0.02)
            prec, rec, f_drum = f_score(true_positive, predHits.shape[0], actHits.shape[0])

            if f_drum > max_fsc:
                max_fsc = f_drum
                good_thresh[i] = threshold
                drum_scores = [true_positive, predHits.shape[0] - true_positive, actHits.shape[0] - true_positive]
                ons=(predHits,actHits)

            tp += drum_scores[0]
            fp += drum_scores[1]
            fn += drum_scores[2]
            drumwise_fscore[i]=max_fsc
            mad_score.append(OnsetEvaluation(ons[0],ons[1],0.2))
            print(i,drum_scores, max_fsc, good_thresh[i])
    try:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        fsc = 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError as dbz:
        print(dbz)
        if tp==0 and fn==0:
            return [1.,1.,1.,drumwise_fscore,good_thresh]
        else:
            return [0,0,0,0, good_thresh]
    fssum=OnsetSumEvaluation(mad_score).fmeasure
    fsmean = OnsetMeanEvaluation(mad_score).fmeasure
    print('Precision: {}'.format(prec))
    print('Recall: {}'.format(rec))
    print('F-score sum: {}'.format(fssum))
    print('F-score mean: {}'.format(fsmean))
    print('Opt_thresh:', good_thresh)
    #print(OnsetSumEvaluation(mad_score), OnsetMeanEvaluation(mad_score))
    return [tp, fp, fn, fssum, fsmean, good_thresh]


from madmom.io.audio import load_audio_file  # to use mp3 in midi dataset


def get_audio_and_annotation(audio_file_path, annot_file_path):
    try:
        buffer, sr = load_audio_file(audio_file_path)
    except:
        return None, None  # Skip broken files
    buffer = buffer / np.max(np.abs(buffer))
    if len(buffer.shape) > 1:
        buffer = buffer[:, 0] + buffer[:, 1]
    filt_spec = stft(buffer, add_diff=True)

    try:
        hits = pd.read_csv(annot_file_path, sep="\t", header=None)
        hits.iloc[:, 0] = time_to_frame(hits.iloc[:, 0], SAMPLE_RATE, HOP_SIZE).astype(int)
    except Exception as e: #if no hits in annotation, return
        return filt_spec, None
    return filt_spec, hits.values


def rnn_train(audio_folder, annotation_folder):
    audio_files = [f for f in os.listdir(audio_folder) if not f.startswith('.')]
    annotation_files = [f for f in os.listdir(annotation_folder) if not f.startswith('.')]
    first = True
    for f in annotation_files:
        print('.', end='', flush=True)
        audio_file_path = audio_folder + f.split('.')[0] + '.wav'
        annot_file_path = annotation_folder + f
        audio, annotation = get_audio_and_annotation(audio_file_path, annot_file_path)
        X, y = vs.get_sequences(audio, annotation)
        if first:
            bigx = X
            bigy = y
        else:
            bigx = np.concatenate((bigx, X), axis=1)
            bigy = np.concatenate((bigy, y), axis=1)
        first = False
    print(bigx.shape, bigy.shape)
    model = vs.make_model()
    vs.train_model(model, (list(bigx), list(bigy)))
    vs.save_model(model)


def rnn_test(audio_file_path, annot_file_path):
    try:
        odf = pickle.load(open("{}/pickled.odf".format('.'), 'rb'))
        score_rnn_result(odf, annot_file_path)
        return
    except:
        print('no presaved odf')
    model = vs.make_model()
    model = vs.restore_best_weigths(model)
    audio, annotation = get_audio_and_annotation(audio_file_path, annot_file_path)
    X, y = vs.get_sequences(audio, annotation)
    print('get_predicting')
    t0 = time.time()
    odf = np.squeeze(np.array(vs.predict_odf(model, list(X))), axis=-1).T
    # pickle.dump(odf,open("{}/pickled.odf".format('.'), 'wb'))
    score_rnn_result(odf, annot_file_path)
    print('time:', time.time() - t0)
    #for i in range(odf.shape[1]):
    #    plt.figure(figsize=(6, 10))
    #plt.subplot(311)
    #    plt.plot(odf[:, i])
    #plt.subplot(312)
    #plt.plot(odf[:, 1])
    #plt.subplot(313)
    #plt.plot(odf[:, 2])
    #    plt.show()


def rnn_test_folder(audio_folder, annotation_folder, train=True, test_full_dataset=False,
                    file_ex='.wav', prefab_splits=None):
    if prefab_splits is not None:
        train_annotation = list(pd.read_csv(prefab_splits + '3-fold_cv_0.txt', sep="\t", header=None, usecols=[0]).values.flatten())
        train_annotation += list(pd.read_csv(prefab_splits + '3-fold_cv_acc_0.txt', sep="\t", header=None).values.flatten())
        train_annotation += list(pd.read_csv(prefab_splits + '3-fold_cv_1.txt', sep="\t", header=None).values.flatten())
        train_annotation += list(pd.read_csv(prefab_splits + '3-fold_cv_acc_1.txt', sep="\t", header=None).values.flatten())
        test_annotation = list(pd.read_csv(prefab_splits + '3-fold_cv_2.txt', sep="\t", header=None).values.flatten())
        test_annotation += list(pd.read_csv(prefab_splits + '3-fold_cv_acc_2.txt', sep="\t", header=None).values.flatten())
        test_audio_takes=test_annotation.copy()
        for f in range(len(train_annotation)):
            train_annotation[f] = train_annotation[f] + '.txt'
        for f in range(len(test_annotation)):
            test_audio_takes[f] = test_audio_takes[f] + file_ex
            test_annotation[f] = test_annotation[f] + '.txt'


    else:
        annotation = [f for f in os.listdir(annotation_folder) if not f.startswith('.')]
        audio = [f for f in os.listdir(audio_folder) if not f.startswith('.')]
        take_names = [os.path.splitext(i)[0] for i in annotation]
        take_names2 = [os.path.splitext(i)[0] for i in audio]
        good_takes = set(np.intersect1d(take_names, take_names2))
        audio_takes = np.array(sorted([i + file_ex for i in good_takes]))
        annotation = np.array(sorted([i + '.txt' for i in good_takes]))
        np.random.seed(7175)
        train_ind = np.random.choice(len(annotation), int(len(annotation) / 3))
        mask = np.zeros(annotation.shape, dtype=bool)
        mask[train_ind] = True
        test_audio_takes = audio_takes[mask]
        train_annotation = annotation[~mask]
        test_annotation = annotation[mask]


    gen = True
    if train:
        first = True
        if gen == True:
            print('making a model...')
            model = vs.make_model()
            print('initializing generators...')
            audios, annotations = [], []
            for f in train_annotation:
                audios.append(audio_folder + os.path.splitext(f)[0] + file_ex)
                annotations.append(annotation_folder + f)
            gens = vs.make_generators(audios, annotations)
            print('begin training...')
            vs.train_model(model, generators=gens)
            print('saving model...')
            vs.save_model(model)
        else:
            print('building training set...')
            for f in train_annotation:
                print('.', end='', flush=True)
                audio_file_path = audio_folder + os.path.splitext(f)[0] + file_ex
                annot_file_path = annotation_folder + f
                audio, annotation = get_audio_and_annotation(audio_file_path, annot_file_path)
                # Shuffle training material
                X, y = vs.get_sequences(audio, annotation, shuffle_sequences=True)
                if first:
                    bigx = X
                    bigy = y
                else:
                    bigx = np.concatenate((bigx, X), axis=1)
                    bigy = np.concatenate((bigy, y), axis=1)
                first = False
            use_validation_set = False
            if use_validation_set:
                first = True
                print('building validation set...')
                for f in test_annotation:
                    print('.', end='', flush=True)
                    audio_file_path = audio_folder + os.path.splitext(f)[0] + file_ex
                    annot_file_path = annotation_folder + f
                    audio, annotation = get_audio_and_annotation(audio_file_path, annot_file_path)
                    X, y = vs.get_sequences(audio, annotation)
                    if first:
                        val_bigx = X
                        val_bigy = y
                    else:
                        val_bigx = np.concatenate((val_bigx, X), axis=1)
                        val_bigy = np.concatenate((val_bigy, y), axis=1)
                    first = False
                model = vs.make_model()
                # model = vs.load_saved_model()
                # model = vs.restore_best_weigths(model)
                vs.train_model(model, (list(bigx), list(bigy)), (list(val_bigx), list(val_bigy)))
            model = vs.make_model()
            vs.train_model(model, (list(bigx), list(bigy)))
            vs.save_model(model)
            # model = vs.restore_best_weigths(model)
    else:
        # model = vs.load_saved_model()
        model = vs.make_model()
        model = vs.restore_best_weigths(model)
    sum = [1**-18, 1**-18, 1**-18, 1**-18, 1**-18]
    thre = np.zeros(18)
    if test_full_dataset:
        test_annotation = annotation
        test_audio_takes = audio_takes


    # if midi dataset create a smaller sample to reduce testing time 0.5% of the full material
    if file_ex=='.mp3':
        from sklearn.utils import resample
        test_audio_takes, test_annotation= resample(test_audio_takes,
                                                    test_annotation,
                                               n_samples=int(len(test_audio_takes)*.0166666666),
                                               replace=False,random_state=0)

    empties = 0  # files with no drum hits, we skip these in the tests
    for i in range(len(test_annotation)):
        audio, annotation = get_audio_and_annotation(audio_folder + test_audio_takes[i],
                                                     annotation_folder + test_annotation[i])
        # Do not shuffle test material!!!
        X, y = vs.get_sequences(audio, annotation, shuffle_sequences=False)
        if X is None:
            continue
        odf = np.squeeze(np.array(vs.predict_odf(model, list(X))), axis=-1).T
        res = score_rnn_result(odf, annotation_folder + test_annotation[i])
        if res is None:
            empties+=1
            continue
        sum[0] += res[0] #tp
        sum[1] += res[1] #fp
        sum[2] += res[2] #fn
        sum[3] += res[3] #fscore(sum)
        sum[4] += res[4] #fcsore(mean)
        thre=thre+res[5] #opt thresholds
    prec = sum[0] / (sum[0] + sum[1])
    rec = sum[0] / (sum[0] + sum[2])
    fsc = 2 * sum[0] / (2 * sum[0] + sum[1] + sum[2])
    print('#precision=', prec)
    print('#recall=', rec)
    print('#f-score=', fsc)
    print('#f-score_mean=', np.mean(sum[4]) / len(test_annotation))
    print('#optimal_thresholds=', thre / (len(test_annotation)-empties))
    return prec, rec, fsc, np.mean(sum[4]) / len(test_annotation), thre / len(test_annotation)
    #return 0,0,0,0,0 #dummies...
    #print('#precision=', sum[0] / len(test_annotation))
    #print('#recall=', sum[1] / len(test_annotation))
    #print('#f-score=', sum[2] / len(test_annotation))
    #print('#f-score_mean=', sum[3] / len(test_annotation))
    #print('#optimal_thresholds=', thre / len(test_annotation))
    #return sum[0] / len(test_annotation), sum[1] / len(test_annotation), sum[2] / len(test_annotation), sum[3] / len(
    #    test_annotation), thre / len(test_annotation)


def debug():
    # rnn_train(audio_folder='../../libtrein/ENST_Drums/audio_drums/', annotation_folder='../../libtrein/ENST_Drums/annotation/')
   ##prec_tot = 0
   #rec_tot = 0
   #fscore_tot = 0
   #rounds = 1
   #t0 = time.time()
   ##rnn_test('../../libtrein/trainSamplet/drumBeatAnnod.wav','../../libtrein/trainSamplet/midiBeatAnnod.csv')
   ##return
   #for i in range(rounds):
   #    # rnn_test('../../libtrein/rbma_13/audio/RBMA-13-Track-01.wav','../../libtrein/rbma_13/annotations/drums/RBMA-13-Track-01.txt')
   #    # prec, rec, fscore, fscore_sum, thresh  = rnn_test_folder(audio_folder='../../libtrein/ENST_Drums/audio_drums/',
   #    #                                    annotation_folder='../../libtrein/ENST_Drums/annotation/', train=False,
   #    #                                    test_full_dataset=True)
   #    # prec_tot += prec
   #    # rec_tot += rec
   #    # fscore_tot += fscore
   #    # prec_tot = 0
   #    # rec_tot = 0
   #    # fscore_tot = 0

   #    #prec, rec, fscore, fscore_mean, thresh = rnn_test_folder(audio_folder='../../libtrein/rbma_13/audio/',
   #    #                              annotation_folder='../../libtrein/rbma_13/annotations/drums/', train=False,
   #    #                                   test_full_dataset=True)
   #    #prec_tot += prec
   #    #rec_tot += rec
   #    #fscore_tot += fscore
   #    #break
   #    # prec, rec, fscore, fscore_mean, thresh  = rnn_test_folder(audio_folder='../../libtrein/SMT_DRUMS/audio/',
   #    #                                    annotation_folder='../../libtrein/SMT_DRUMS/annotations/', train=False,
   #    #                                    test_full_dataset=False, file_ex='.wav')
   #    prec, rec, fscore, fscore_mean, thresh = rnn_test_folder(audio_folder='../../libtrein/midi/mp3/',
   #                                                    annotation_folder='../../libtrein/midi/annotations/drums_l/',
   #                                                    train=True,
   #                                                    test_full_dataset=False, file_ex='.mp3',
   #                                                    prefab_splits='../../libtrein/midi/splits/')
   #    # prec, rec, fscore=rnn_test_folder(audio_folder='../../libtrein/rbma_13/audio/',
   #    #  annotation_folder='../../libtrein/rbma_13/annotations/drums/')
   #    # prec_tot += prec
   #    # rec_tot += rec
   #    # fscore_tot += fscore
   #print('Total numbers for {} rounds:\n'.format(rounds))
   #print('#precision=', prec_tot / rounds)
   #print('#recall=', rec_tot / rounds)
   #print('#f-score=', fscore_tot / rounds)
   #print('#Run time:', time.time() - t0)
   ##rnn_test_('../../libtrein/ENST_Drums/audio_drums/b132_MIDI-minus-one_blues-102_sticks.wav',
   ##                                                 '../../libtrein/ENST_Drums/annotation/b132_MIDI-minus-one_blues-102_sticks.txtcsv')
   #return
    # debug
    # initKitBG('Kits/mcd2/',8,K)
    # K = 1
    # file = './Kits/mcd_pad/'
    method = 'NMFD'
    # file='../DXSamplet/'
    # initKitBG(file,K=K, drumwise=True, method=method)
    ## print('Kit init processing time:%0.2f' % (time.time() - t0))
    # loadKit(file)
    # print(test_run(file_path=file, annotated=True, method=method, quantize=0., skip_secs=0))
    #
    # drumsynth.createWav('testbeat3.csv', 'sysAnnodQuantizedPart.wav', addCountInAndCountOut=False,
    #                    deltaTempo=1,
    #                    countTempo=1)
    # return
    prec_tot = 0
    rec_tot = 0
    fscore_tot = 0
    rounds = 1
    t0 = time.time()
    for i in range(rounds):
        # prec, rec, fscore=run_folder(audio_folder='../../libtrein/ENST_Drums/audio_drums/', annotation_folder='../../libtrein/ENST_Drums/annotation/')

        prec, rec, fscore = run_folder(audio_folder='./audio/',
                                       annotation_folder='./annotation/',soundcheck_folder='./soundcheck', method=method)

        # prec, rec, fscore=run_folder(audio_folder='../../libtrein/rbma_13/audio/', annotation_folder='../../libtrein/rbma_13/annotations/drums/')

        prec_tot += prec
        rec_tot += rec
        fscore_tot += fscore
    print('Total numbers for {} rounds:\n'.format(rounds))
    print('#precision=', prec_tot / rounds)
    print('#recall=', rec_tot / rounds)
    print('#f-score=', fscore_tot / rounds)
    print('#Run time:', time.time() - t0)
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

