import python.nmfd as nmfd
import python.onset_detection as onset_detection
import numpy as np
from python.constants import *
from scipy.fftpack import fft
from scipy.ndimage.filters import median_filter

# globals
# todo remove these!!!
max_n_frames = 10
total_priors = 0


class Drum(object):
    """
    A Drum is any user playable drumkit part representation

    Parameters
    ----------
    name : Int
        Name of the drum
    peaks : Numpy array
        Array of soundcheck hit locations, used for automatic recalculating of threshold
    heads : Numpy array
        The heads prior templates
    tails : Numpy array
        The tails prior templates
    midinote: int, optional
        midi note representing the drum
    threshold : float, optional
        peak picking threshold.
    hitlist : Numpy array
        Hit locations discovered in source separation

    Notes
    -----


    """

    def __init__(self, name, peaks, heads=None, tails=None,ntemplates=0,
                 midinote=MIDINOTE, threshold=THRESHOLD, hitlist=None):

        # set attributes
        self.name = name

        self.peaks = peaks
        self.ntemplates=ntemplates

        self.tails = tails

        self.heads = heads
        if midinote:
            self.midinote = midinote
        if threshold:
            self.threshold = float(threshold)
        else:
            self.threshold = 0.
        if hitlist:
            self.hitlist = hitlist
        self.set_ntemplates()

    def set_hits(self, hitlist):
        self.hitlist = hitlist

    def get_hits(self):
        return self.hitlist

    def concat_hits(self, hitlist):
        self.hitlist = np.concatenate((self.get_hits(), hitlist))

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_peaks(self, peaks):
        self.peaks = peaks

    def get_peaks(self):
        return self.peaks

    def set_heads(self, heads):
        self.heads = heads

    def get_heads(self):
        return self.heads

    def set_tails(self, tails):
        self.tails = tails


    def get_tails(self):
        return self.tails

    def set_midinote(self, midinote):
        self.midinote = int(midinote)

    def get_midinote(self):
        return self.midinote

    def set_threshold(self, threshold):
        self.threshold = float(threshold)

    def get_threshold(self):
        return self.threshold
    def set_ntemplates(self):
        if self.heads is not None and self.tails is not None:
            self.ntemplates=self.heads.shape[2]+self.tails.shape[2]
        else:
            self.ntemplates=0
    def get_ntemplates(self):
        return self.ntemplates

def findDefBins(frames=None, filteredSpec=None, ConvFrames=None, matrices=None):
    """
    Calculate the prior vectors for W to use in NMF by averaging the sample locations
    :param frames: Numpy array of hit locations (frame numbers)
    :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
    :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
     """
    global total_priors
    if matrices is None:
        gaps = np.zeros((frames.shape[0], ConvFrames))
        for i in range(frames.shape[0]):
            for j in range(gaps.shape[1]):
                gaps[i, j] = frames[i] + j
        a = np.reshape(filteredSpec[gaps.astype(int)], (N_PEAKS, -1))
    else:
        a = np.array(matrices[0])
    heads = np.reshape(np.mean(a, axis=0), (FILTERBANK_SHAPE, max_n_frames, 1), order='F')
    if matrices is None:
        tailgaps = np.zeros((frames.shape[0], ConvFrames))
        for i in range(frames.shape[0]):
            for j in range(gaps.shape[1]):
                tailgaps[i, j] = frames[i] + j + ConvFrames

        a2 = np.reshape(filteredSpec[tailgaps.astype(int)], (N_PEAKS, -1))

    else:
        a2 = np.array(matrices[1])
    tails = np.reshape(np.mean(a2, axis=0), (FILTERBANK_SHAPE, max_n_frames, 1), order='F')
    total_priors += 2
    return (heads, tails, 1, 1)

def getPeaksFromBuffer(filt_spec, numHits):
    """

    :param filt_spec: numpy array, the filtered spectrogram containing sound checked drum audio

    :param numHits: int, the number of hits to recover from filt_spec

    :return: numpy array, peak locations in filt_spec
    """
    threshold = 1
    searchSpeed = .1
    # peaks=cleanDoubleStrokes(madmom.features.onsets.peak_picking(superflux_3,threshold),resolution)
    H0=onset_detection.spectral_difference(filt_spec)
    # peaks = cleanDoubleStrokes(pick_onsets(H0 / H0.max(), delta=threshold), resolution)
    peaks = onset_detection.pick_onsets(H0, threshold=threshold)
    changed = False
    last = 0
    while (peaks.shape != (numHits,)):
        # Make sure we don't go over numHits
        # There is a chance of an infinite loop here!!! Make sure that don't happen
        if (peaks.shape[0] > numHits) or (peaks.shape[0] < last):
            if changed == False:
                searchSpeed = searchSpeed / 2
            changed = True
            threshold += searchSpeed
        else:
            changed = False
            threshold -= searchSpeed
        # peaks=cleanDoubleStrokes(madmom.features.onsets.peak_picking(superflux_3,threshold),resolution)
        # peaks = cleanDoubleStrokes(pick_onsets(H0, delta=threshold), resolution)
        last = peaks.shape[0]
        peaks = onset_detection.pick_onsets(H0, threshold=threshold)
    return peaks

def create_templates_optics(frames=None, filteredSpec=None, ConvFrames=None, matrices=None):
    """
    Calculate the prior vectors for W to use in NMF
    :param frames: Numpy array of hit locations (frame numbers)
    :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
    :param ConvFrames: int, number of frames the priors contain
    :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
    """

    from drum_off import OPTICS
    global total_priors
    if matrices is None:
        gaps = np.zeros((frames.shape[0], max_n_frames))
        # gaps = np.zeros((frames.shape[0], ConvFrames))
        for i in range(frames.shape[0]):
            for j in range(ConvFrames):
                gaps[i, j] = frames[i] + j
        a = np.reshape(filteredSpec[gaps.astype(int)], (N_PEAKS, -1))
        print(a.shape)
    else:
        a = np.array(matrices[0])
        # a=np.reshape(a,(a.shape[0], -1))
        # print(a.shape)

    opts = OPTICS.OPTICS(min_cluster_size=16, n_jobs=-1).fit(a)
    K1, unique_labels = np.unique(opts.labels_, return_inverse=True)
    indices = np.unique(unique_labels)
    heads = np.zeros((FILTERBANK_SHAPE, max_n_frames, K1.shape[0]))
    for i in indices:
        heads[:, :, i] = np.reshape(np.mean(a[unique_labels == i, :], axis=0), (FILTERBANK_SHAPE, max_n_frames),
                                    order='F')
    if matrices is None:
        tailgaps = np.zeros((frames.shape[0], max_n_frames))
        for i in range(frames.shape[0]):
            for j in range(tailgaps.shape[1]):
                tailgaps[i, j] = frames[i] + j + ConvFrames
        a2 = np.reshape(filteredSpec[tailgaps.astype(int)], (N_PEAKS, -1))
    else:
        a2 = np.array(matrices[1])
        # a2 = np.reshape(a2, (a2.shape[0], -1))
        # print(a2.shape)
    opts = OPTICS.OPTICS(min_cluster_size=16, n_jobs=-1).fit(a2)
    K2, unique_labels = np.unique(opts.labels_, return_inverse=True)
    indices = np.unique(unique_labels)
    tails = np.zeros((FILTERBANK_SHAPE, max_n_frames, K2.shape[0]))
    for i in indices:
        tails[:, :, i] = np.reshape(np.mean(a2[unique_labels == i, :], axis=0), (FILTERBANK_SHAPE, max_n_frames),
                                    order='F')
    total_priors += K1.shape[0] + K2.shape[0]
    print('OPTICS clusters', K1.shape[0] + K2.shape[0],'total:', total_priors)
    return (heads, tails, K1, K2)


def create_templates(frames=None, filteredSpec=None, ConvFrames=None, matrices=None):
    """
    Calculate the prior vectors for W to use in NMF by averaging the sample locations
    :param frames: Numpy array of hit locations (frame numbers)
    :param filteredSpec: Spectrogram, the spectrogram where the vectors are extracted from
    :return: tuple of Numpy arrays, prior vectors Wpre,heads for actual hits and tails for decay part of the sound
     """
    global total_priors
    if matrices is None:
        gaps = np.zeros((frames.shape[0], ConvFrames))
        for i in range(frames.shape[0]):
            for j in range(gaps.shape[1]):
                gaps[i, j] = frames[i] + j
        a = np.reshape(filteredSpec[gaps.astype(int)], (N_PEAKS, -1))
    else:
        a = np.array(matrices[0])
    heads = np.reshape(np.mean(a, axis=0), (FILTERBANK_SHAPE, max_n_frames, 1), order='F')
    if matrices is None:
        tailgaps = np.zeros((frames.shape[0], ConvFrames))
        for i in range(frames.shape[0]):
            for j in range(gaps.shape[1]):
                tailgaps[i, j] = frames[i] + j + ConvFrames

        a2 = np.reshape(filteredSpec[tailgaps.astype(int)], (N_PEAKS, -1))

    else:
        a2 = np.array(matrices[1])
    tails = np.reshape(np.mean(a2, axis=0), (FILTERBANK_SHAPE, max_n_frames, 1), order='F')
    total_priors += 2
    return (heads, tails, 1, 1)


def to_midinote(notes):
    """
    Transform drum names to their corresponding midinote
    :param notes: int or list, For instance kick, snare and closed hi-hat is [0,1,2]
    :return: list of corresponfing midinotes [36, 38, 42]
    """
    return list(midinote_to_index(i) for i in notes)

def to_index(midinotes):
    return list(midinote_to_index(i) for i in list(midinotes))

def midinote_to_index(note):
    try:
        return f_to_l_map[note]
    except KeyError as e:
        print ('unknown note', e)
        return None

def midinote_to_name(note):
    return names_l_map[f_to_l_map[note]]

def index_to_name(index):
    return names_l_map[index]

def name_to_index(name):
    return names_l_map.index(name)


def get_Wpre(drumkit, max_n_frames=max_n_frames):
    total_heads = 0
    global total_priors
    if total_priors==0:
        total_priors=sum([f.get_ntemplates() for f in drumkit])
        print(total_priors)
        #total_priors = len(drumkit) * 2
    Wpre = np.zeros((FILTERBANK_SHAPE, total_priors, max_n_frames))
    for i in range(len(drumkit)):
        heads = drumkit[i].get_heads()
        K1 = heads.shape[2]
        ind = total_heads
        for j in range(K1):
            Wpre[:, ind + j, :] = heads[:, :, j]
            total_heads += 1
    total_tails = 0
    for i in range(len(drumkit)):
        tails = drumkit[i].get_tails()
        K2 = tails.shape[2]
        ind = total_heads + total_tails
        for j in range(K2):
            Wpre[:, ind + j, :] = tails[:, :, j]
            total_tails += 1
    return Wpre, total_heads
#def get_Wpre(drumkit, max_n_frames=max_n_frames):
#    total_heads = 0
#    global total_priors
#    total_priors = len(drumkit) * 2
#    Wpre = np.zeros((FILTERBANK_SHAPE, total_priors, max_n_frames))
#    for i in range(len(drumkit)):
#        heads = drumkit[i].get_heads()
#        K1 = heads.shape[2]
#        ind = total_heads
#        for j in range(K1):
#            Wpre[:, ind + j, :] = heads[:, :, j]
#            total_heads += 1
#    total_tails = 0
#    for i in range(len(drumkit)):
#        tails = drumkit[i].get_tails()
#        K2 = tails.shape[2]
#        ind = total_heads + total_tails
#        for j in range(K2):
#            Wpre[:, ind + j, :] = tails[:, :, j]
#            total_tails += 1
#    return Wpre, total_heads

def recalculate_thresholds(filt_spec, shifts, drumkit, drumwise=False, method='NMFD'):
    """

    :param filt_spec: numpy array, The spectrum containing sound check
    :param shifts: list of integers, the locations of different drums in filt_spec
    :param drumkit: list of drums
    :param drumwise: boolean, if True the thresholds will be calculated per drum,
     if False a single trseshold is used for the whole drumkit
    :param rm_win: int, window size.
    :return: None
    """
    Wpre, total_heads = get_Wpre(drumkit)
    if method == 'NMFD':
        H, Wpre, err1 = nmfd.NMFD(filt_spec.T, iters=128, Wpre=Wpre, include_priors=True, n_heads=total_heads)
    else:
        H, err1 = nmfd.semi_adaptive_NMFB(filt_spec.T, Wpre=Wpre, iters=128, n_heads=total_heads)
    total_heads = 0
    print(H.shape)
    Hs = []
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
        H0 = H0 / H0.max()
        Hs.append(H0)

        if drumwise:
            deltas = np.linspace(0., 1, 100)
            f_zero = 0
            threshold = 0
            maxd = 0
            for d in deltas:
                if i < len(shifts) - 1:
                    peaks = onset_detection.pick_onsets(H0[shifts[i]:shifts[i + 1]], threshold=d)

                else:
                    peaks = onset_detection.pick_onsets(H0[shifts[i]:], threshold=d)

                drumkit[i].set_hits(peaks[np.where(peaks < filt_spec.shape[0] - 1)])
                predHits = drumkit[i].get_hits()
                actHits = drumkit[i].get_peaks()
                trueHits = k_in_n(actHits, predHits, window=1)
                prec, rec, f_drum = f_score(trueHits, predHits.shape[0], actHits.shape[0])
                if f_drum == f_zero:
                    maxd = d
                if f_drum > f_zero:
                    f_zero = f_drum
                    threshold = d
            # if optimal threshold is within a range [threshold, maxd] find a sweet spot empirically,
            #  increasing alpha lowers threshold and decreases precision
            if maxd > 0:
                alpha = 0.8
                beta = 1 - alpha
                threshold = (alpha * threshold + beta * maxd)

            # arbitrary minimum threshold check
            threshold = max((threshold, 0.05))
            # arbitrary maximum threshold check
            threshold = min((threshold, 0.333))
            print('threshold: %f, f-score: %f' % (threshold, f_zero))
            drumkit[i].set_threshold(threshold)

    if not drumwise:
        deltas = np.linspace(0., 1, 100)
        f_zero = 0
        threshold = 0
        maxd = 0
        for d in deltas:
            precision, recall, fscore, true_tot = 0, 0, 0, 0
            for i in range(len(drumkit)):
                peaks = onset_detection.pick_onsets(Hs[i], threshold=d)
                drumkit[i].set_hits(peaks[np.where(peaks < filt_spec.shape[0] - 1)])
                predHits = drumkit[i].get_hits()
                actHits = drumkit[i].get_peaks() + shifts[i] - 1
                trueHits = k_in_n(actHits, predHits, window=3)
                prec, rec, f_drum = f_score(trueHits, predHits.shape[0], actHits.shape[0])
                precision += prec * actHits.shape[0]
                recall += rec * actHits.shape[0]
                fscore += (f_drum * actHits.shape[0])
                true_tot += actHits.shape[0]

            if (fscore / true_tot) == f_zero:
                maxd = d
            if (fscore / true_tot) > f_zero:
                f_zero = (fscore / true_tot)
                threshold = d

        # if optimal threshold range, increase a bit
        if maxd > 0:
            alpha = 0.666
            beta = 1 - alpha
            threshold = (alpha * threshold + beta * maxd)

        # arbitrary minimum threshold check
        # threshold = max((threshold, 0.15))

        print('delta:', threshold, f_zero)
        for i in range(len(drumkit)):
            drumkit[i].set_threshold(threshold)


def rect_bark_filter_bank(filter_height=0.1, min_freq=20, max_freq=20000, n_bins=49,
                          fs=FRAME_SIZE, sr=SAMPLE_RATE):
    """
    Rectangular filtebank
    :param filter_height: float, height of filters
    :param min_freq: int, lowest filter frequency
    :param max_freq: int, highest filter frequency
    :param n_bins: int, amount of filters.
    :return: filterbank
    """
    # stft bins
    stft_bins = np.arange(fs >> 1) / (fs * 1. / sr)
    # hack for more bark freq, 57 is the max, otherwise increment the denominator.
    bark_freq = np.array((600 * np.sinh((np.arange(0, n_bins)) / 12)))
    # filter frequencies
    # Lose the low freq rumble
    bark_freq = bark_freq[bark_freq > min_freq]
    bark_freq = bark_freq[bark_freq < max_freq]
    filt_bank = np.zeros((len(stft_bins), len(bark_freq)))
    stft_bins = stft_bins[stft_bins >= bark_freq[0]]
    index = 0
    for i in range(0, len(bark_freq) - 1):
        while stft_bins[index] > bark_freq[i] and stft_bins[index] < bark_freq[i + 1] and index <= len(stft_bins):
            filt_bank[index][i] = filter_height
            index += 1
    filt_bank[index:, -1] += filter_height
    return np.array(filt_bank)


def stft(audio_signal, streaming=False, hs=HOP_SIZE,
         fs=FRAME_SIZE, sr=SAMPLE_RATE, add_diff=False):
    ##nr. frequency bins = Half of FRAME_SIZE
    n_frames = int(fs / 2)
    # HOP_LENGTH spaced index
    frames_index = np.arange(0, len(audio_signal), hs)
    # +2 frames to correct NMF systematic errors... +1 for NMFD
    err_corr =-1
    if streaming:
        err_corr = 0
    data = np.zeros((len(frames_index) + err_corr, n_frames), dtype=np.complex64)
    # Window
    win = np.hanning(fs)#np.kaiser(fs, np.pi ** 2)
    # STFT
    for frame in range(len(frames_index)):
        # Get one frame length audio clip
        one_frame = audio_signal[frames_index[frame]:frames_index[frame] + fs]
        # Pad last frame if needed
        if one_frame.shape[0] < fs:
            one_frame = np.pad(one_frame, (0, fs - one_frame.shape[0]), 'constant', constant_values=(0))
        # apply window
        fft_frame = one_frame * win
        # FFT
        data[frame + err_corr] = fft(fft_frame, fs, axis=0)[:n_frames]

    # mag spectrogram
    data = np.abs(data)
    # filter data
    filterbank = rect_bark_filter_bank(filter_height=0.1,
                                       min_freq=20, max_freq=17000, n_bins=FILTERBANK_SHAPE+1, fs=fs, sr=sr)
    data = data @ filterbank
    # for streaming we have to remove sys.error compesation.
    win = np.hanning(4)
    for i in range(data.shape[1]):
        data[:, i] = np.convolve(data[:, i], win, 'same')
    if add_diff:
        diff=data
        diff[1:,:]=np.diff(data, n=1, axis=0)
        data=np.concatenate((data, diff), axis=-1)
    return data


def frame_to_time(frames, sr=SAMPLE_RATE, hop_length=HOP_SIZE):
    """
    Transforms frame numbers to time values

    :param frames: list of integers to transform
    :param sr: int, Sample rate of the FFT
    :param hop_length: int, Hop length of FFT

    :return: Numpy array of time values
    """

    samples = (np.asanyarray(frames) * (hop_length)).astype(int)
    return np.asanyarray(samples) / float(sr)


def time_to_frame(times, sr=SAMPLE_RATE, hop_length=HOP_SIZE):
    """
    Transforms time values to frame numbers

    :param times: list of timevalues to transform
    :param sr: int, Sample rate of the FFT
    :param hop_length: int, Hop length of FFT

    :return: Numpy array of frame numbers
    """
    samples = (np.asanyarray(times) * float(sr))
    return np.rint(np.asanyarray(samples) / (hop_length))


def f_score(hits, hitNMiss, actual):
    """
    Function to calculate precisionm, recall and f-score
    :param hits: array of true positive hits
    :param hitNMiss: array of all detected hits
    :param actual: array of pre annotated hits
    :return: list of floats (precision, recall, fscore)
    :exception: e if division by zero occurs when no hits are detected, or there are no true hits returns zero values
    """
    try:

        precision = (float(hits) / hitNMiss)

        recall = (float(hits) / actual)

        fscore = (2 * ((precision * recall) / (precision + recall)))

        return (precision, recall, fscore)

    except Exception as e:

        # print('fscore: ',e)
        return (0.0, 0.0, 0.0)


def k_in_n(k, n, window=1):
    """
    Helper function to calculate true positive hits for precision, recall and f-score calculation

    :param k: numpy array, list of automatic annotation hits
    :param n: numpy array, list of pre annotated hits
    :param window: float, the windoe in which the hit is regarded as true positive
    :return: float, true positive hits
    """
    hits = 0

    for i in n:
        for j in k:
            if (j - window <= i <= j + window):
                hits += 1
                break
            if (j + window > i):
                break
    return float(hits)


def movingAverage(x, window=500):
    return median_filter(x, size=(window))


def cleanDoubleStrokes(hitList, resolution=10):
    retList = []
    lastSeenHit = 0
    for i in range(len(hitList)):
        if hitList[i] >= lastSeenHit + resolution:
            retList.append(hitList[i])
            lastSeenHit = hitList[i]
    return (np.array(retList))


def acceptHit(value, hits):
    """
    Helper method to clear mistakes of the annotation such as ride crash and an open hi hat.
    :param value: int the hit were encoding
    :param hits: binary string, The hits already found in that location
    :return: boolean, if it is ok to annotate this drum here.
    """
    # Test to discard mote than 3 drums on the same beat
    sum = 0
    for i in range(value):
        if np.bitwise_and(i, hits):
            sum += 1
        if sum > 3:
            return False
    # discard overlapping cymbals for ease of annotation and rare use cases of overlapping cymbals.
    offendingHits = np.array([
        # value == 2 and np.bitwise_and(3, hits),
        value == 3 and np.bitwise_and(8, hits),
        value == 3 and np.bitwise_and(2, hits),
        value == 3 and np.bitwise_and(7, hits),
        value == 6 and np.bitwise_and(7, hits),
        value == 6 and np.bitwise_and(3, hits),
        value == 6 and np.bitwise_and(2, hits),
        value == 7 and np.bitwise_and(6, hits),
        value == 8 and np.bitwise_and(2, hits),
        value == 8 and np.bitwise_and(3, hits)
    ])
    if offendingHits.any():
        pass
        # return False
    return True


def truncZeros(frames):
    """
    Enceode consecutive pauses in notation to negative integers
    :param frames: numpy array, the notation of all frames
    :return: numpy array, notation with pause frames truncated to neg integers
    """
    zeros = 0
    for i in range((frames.size)):
        if frames[i] == 0:
            zeros += 1
            # longest pause
            if zeros == MAX_PAUSE:
                frames[i - zeros + 1] = -zeros
                zeros = 0
                continue
        elif zeros != 0 and frames[i] != 0:
            # Encode pause to a negative integer
            frames[i - zeros] = -zeros
            zeros = 0
    # 0:1, 1:2, 2:3, 3:4 ->
    frames = frames[frames != 0]
    return frames


def mergerowsandencode(a):
    """
        Merges hits occuring at the same frame.
        i.e. A kick drum hit  and a closed hihat hit at
        frame 100 are encoded from
            100 0
            100 2
        to
            100 5
        where 5 is decoded into char array 000000101 in the RNN.

        Also the hits are assumed to be in tempo 120bpm and all the
        frames not in 120bpm 16th notes are quantized.

        :param a: numpy array of hits

        :return: numpy array of merged hits

        Notes
        -----
        The tempo is assumed to be the global value

        """
    # Sixteenth notes length in this sample rate and frame size
    sixtDivider = SAMPLE_RATE / Q_HOP / SXTH_DIV
    if QUANTIZE:
        for i in range(len(a)):
            # Move frames to nearest sixteenth note
            a[i] = (np.rint(a[i][0] / sixtDivider),) + a[i][1:]
    # Define max frames from the first detected hit to end
    # print(len(a[-1]))
    if (len(a) == 0):
        return []
    maxFrame = int(a[-1][0] - a[0][0] + 1)
    # print(maxFrame)
    # spaceholder for return array
    frames = np.zeros(maxFrame, dtype=int)
    for i in range(len(a)):
        # define true index by substracting the leading empty frames
        index = int(a[i][0] - a[0][0])
        if index >= frames.shape[0]:
            break
        # The actual hit information
        value = int(a[i][1])
        # Encode the hit into a character array, place 1 on the index of the drum #
        if acceptHit(value, frames[index]):
            # try:
            new_hit = np.bitwise_or(frames[index], 2 ** value)
            # if new_hit in possible_hits:
            frames[index] = new_hit
        # except:
        # print(frames[index], value)

    # return array of merged hits starting from the first occurring hit event
    if ENCODE_PAUSE:
        frames = truncZeros(frames)
    # print('frames',len(frames))
    return frames


def splitrowsanddecode(a, deltaTempo=1.0):
    """
        Split hits occuring at the same frame to separate lines containing one row per hit.
        i.e. A kick drum hit  and a closed hihat hit at
        frame 100 are decoded from
            100 5
        to
            100 0
            100 2

        :param a: numpy array of hits

        :return: numpy array of separated hits

        """
    decodedFrames = []
    # multiplier to make tempo the global tempo after generation.
    # print(deltaTempo)
    frameMul = 1 / deltaTempo
    if False:
        frameMul = SAMPLE_RATE / Q_HOP / SXTH_DIV
    i = 0
    pause = 0
    while i in range(len(a)):
        # if we find a pause we add that to the pause offset
        if ENCODE_PAUSE:
            if a[i] < 0:
                pause += (-1 * a[i]) - 1
                i += 1
                continue
        # split integer values to a binary array
        for j, k in enumerate(dec_to_binary(a[i])):
            if int(k) == 1:
                # store framenumber(index) and drum name to list, (MAX_DRUMS-1) to flip drums to right places,
                decodedFrames.append([int((i + pause) * frameMul), abs(j - (MAX_DRUMS - 1))])
        i += 1
    # return the split hits
    return decodedFrames

def dec_to_binary(f, str_len=MAX_DRUMS, ret_type='str'):
    """
    Returns a binary representation on a given integer
    :param f: an integer
    :return: A binary array representation of f
    """
    if f<0:
        leading_zeros=list(format(-f, "0{}b".format(str_len - MAX_DRUMS)))
        leading_zeros.extend(list(format(0, "0{}b".format(MAX_DRUMS))))

        bin_form=np.array(leading_zeros).astype(ret_type)
    else:
        bin_form=np.array(list(format(f, "0{}b".format(str_len)))).astype(ret_type)
    return list(bin_form)

def enc_to_int(a, is_long=False):
    int_form=0
    sign=1
    origa=a
    if is_long:
        if enc_to_int(a[MAX_DRUMS:])>0:
            sign=-1
            a=a[MAX_DRUMS:]
    for i in range(len(a)):
        if a[i]==1:
            int_form = np.bitwise_or(int_form, 2 ** i)

    if int_form==0 and sign==-1:print(origa);int_form=1
    return sign*int_form

