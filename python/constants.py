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
MAX_DRUMS = 19 # Maximum kit size
MAX_PAUSE = 16
BIN_STRING_LEN=MAX_DRUMS+MAX_PAUSE
N_PEAKS = 16 # IF CHANGED ALL PREVIOUS SOUNDCHECKS INVALIDATE!!!
FILTERBANK_SHAPE=48
MS_IN_MIN = 60000
SXTH_DIV = 16
QUANTIZE = False
ENCODE_PAUSE = True

# general midi to 18-class label system From vogl http://ifs.tuwien.ac.at/~vogl/dafx2018/
names_l_map = ['BD', 'SD', 'SS', 'CLP', 'LT', 'MT', 'HT', 'CHH', 'PHH', 'OHH', 'TB', 'RD', 'RB', 'CRC1','CRC2', 'SPC', 'CHC',
               'CB', 'CL']
num_l_drum_notes = len(names_l_map)
f_to_l_map = {35: 0,  # BD
              36: 0,  # BD
              38: 1,  # SD
              40: 1,  # SD
              37: 2,  # side stick
              39: 3,  # clap
              41: 4,  # TT  (lft)
              43: 4,  # (hft)
              45: 5,  # (lt)
              47: 5,  # (lmt)
              48: 6,  # (hmt)
              50: 6,  # (ht)
              42: 7,  # HH
              44: 8,  # pedal hh
              46: 9,  # open hh
              54: 10,  # tamborine
              51: 11,  # RD
              59: 11,  # ride 2
              53: 12,  # ride bell
              49: 13,  # crash
              57: 14,  # crash2
              55: 15,  # splash
              52: 16,  # chinese
              56: 17,  # cowbell
              75: 18,  # click
              78: 9,  # OHH
              79: 7,  # HH dtx
              }

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
