'''
A collection of constants used throughout the game
'''
FRAME_SIZE = 2 ** 11
HOP_SIZE = 441
# After quantization to 120bpm we could use hop size 1378.125 the resolution of 64th note at 120bpm
# or 918.75 the 64th triplet with every third frame the 32th.
# Float presicion is allowed since the hop size is only used for time to frame calculations.
K=1
Q_HOP = 689.0625*2 #have options[459.375,612.5,689.0625,1378.125]*[.5,1,2,4,8,16]
SAMPLE_RATE = 44100
MIDINOTE = 36  # kickdrum in standard midi notation
THRESHOLD = 0.0
PROBABILITY_THRESHOLD = 0.0
DEFAULT_TEMPO = 120  # 175.44bpm 1/16 with hop 516
DRUMKIT_PATH = '../trainSamplet/'

DELTA = 0.15
MIDINOTES = [36, 38, 42, 46, 44, 50, 48, 47, 43, 41, 51, 49, 57,
             55]  # BD, SN, CHH, OHH,SHH, TT,TT2,TT3, FT,FT2, RD, CR,CR2,splash, Here we need generality
MAX_DRUMS = 14  # Maximum kit size
MAX_PAUSE = 16
BIN_STRING_LEN=MAX_DRUMS+MAX_PAUSE
N_PEAKS = 32 # IF CHANGED ALL PREVIOUS SOUNDCHECKS INVALIDATE!!!
FILTERBANK_SHAPE=48
MS_IN_MIN = 60000
SXTH_DIV = 16
QUANTIZE = False
ENCODE_PAUSE = True

#Not used
#
# # Bark scale filterbank Shortest processing time, good results
# FILTERBANK = madmom.audio.filters.BarkFilterbank(
#     madmom.audio.stft.fft_frequencies(num_fft_bins=int(FRAME_SIZE / 2), sample_rate=SAMPLE_RATE),
#     num_bands='double', fmin=20.0, fmax=15500.0, norm_filters=False, unique_filters=True)
#
# # Mel scale filterbank
# #FILTERBANK= madmom.audio.filters.MelFilterbank(
# #    madmom.audio.stft.fft_frequencies(num_fft_bins=int(1024 / 2), sample_rate=SAMPLE_RATE),
# #    num_bands=20, fmin=20.0, fmax=17000.0, norm_filters=False, unique_filters=True)
# #
# # Logarithmic filterbank
# # FILTERBANK =madmom.audio.filters.LogarithmicFilterbank(
# #    madmom.audio.stft.fft_frequencies(num_fft_bins=int(FRAME_SIZE / 2), sample_rate=SAMPLE_RATE),
# #    num_bands=18, fmin=20.0, fmax=17000.0, fref=110.0, norm_filters=True, unique_filters=True, bands_per_octave=True)
