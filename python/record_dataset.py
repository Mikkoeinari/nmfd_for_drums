import sys
import threading
import time
from os.path import expanduser

import numpy as np
import pandas as pd
import pydub
import rtmidi
import sounddevice
from pydub import effects
from scipy.stats import gamma

from python.onset_detection import pick_onsets
from python.utils import stft, frame_to_time, time_to_frame, k_in_n

RATE = 44100
sounddevice.default.samplerate = 44100
sounddevice.default.channels = 1
sounddevice.default.dtype = 'int16'

home = expanduser("~")
# direct to your ffmpeg executable to use mp3 codec
pydub.AudioSegment.converter = home + "/Anaconda/bin/ffmpeg"

global _ImRecording
_ImRecording = False

# general midi to 18-class label system From vogl http://ifs.tuwien.ac.at/~vogl/dafx2018/
names_l_map = ['BD', 'SD', 'SS', 'CLP', 'LT', 'MT', 'HT', 'CHH', 'PHH', 'OHH', 'TB', 'RD', 'RB', 'CRC', 'SPC', 'CHC',
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
              57: 13,  # crash2
              55: 14,  # splash
              52: 15,  # chinese
              56: 16,  # cowbell
              75: 17,  # click
              78: 9,  # OHH
              79: 7,  # HH dtx
              }

# from https://stackoverflow.com/questions/13207678/whats-the-simplest-way-of-detecting-keyboard-input-in-python-from-the-terminal
global isWindows
isWindows = False

try:
    from win32api import STD_INPUT_HANDLE
    from win32console import GetStdHandle, KEY_EVENT, ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT

    isWindows = True
except ImportError as e:
    import sys
    import select
    import termios


class KeyPoller():
    def __enter__(self):
        global isWindows
        if isWindows:
            self.readHandle = GetStdHandle(STD_INPUT_HANDLE)
            self.readHandle.SetConsoleMode(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT)

            self.curEventLength = 0
            self.curKeysLength = 0

            self.capturedChars = []
        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

        return self

    def __exit__(self, type, value, traceback):
        if isWindows:
            pass
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def poll(self):
        if isWindows:
            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)

            eventsPeek = self.readHandle.PeekConsoleInput(10000)

            if len(eventsPeek) == 0:
                return None

            if not len(eventsPeek) == self.curEventLength:
                for curEvent in eventsPeek[self.curEventLength:]:
                    if curEvent.EventType == KEY_EVENT:
                        if ord(curEvent.Char) == 0 or not curEvent.KeyDown:
                            pass
                        else:
                            curChar = str(curEvent.Char)
                            self.capturedChars.append(curChar)
                self.curEventLength = len(eventsPeek)

            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)
            else:
                return None
        else:
            dr, dw, de = select.select([sys.stdin], [], [], 0)
            if not dr == []:
                return sys.stdin.read(1)
            return None


def soundcheck_listener():
    global _ImRecording
    data = []
    t = None
    with KeyPoller() as keyPoller:
        while True:
            key = keyPoller.poll()
            if not key is None:
                if key == "r":
                    if _ImRecording == False:
                        t = threading.Thread(target=record_part_with_midi, args=(30, data))
                        t.start()
                if key == "s":
                    if _ImRecording == True:
                        _ImRecording = False
                        t.join()
                        return data[0]


def drumpart_listener():
    global _ImRecording
    data = []
    t = None
    with KeyPoller() as keyPoller:
        while True:
            key = keyPoller.poll()
            if not key is None:
                if key == "r":
                    if _ImRecording == False:
                        t = threading.Thread(target=record_part_with_midi, args=(60, data))
                        t.start()
                        # t.join()
                        # return data[0][0], data[0][1], False
                elif key == "s":
                    if _ImRecording == True:
                        _ImRecording = False
                        t.join()
                        return data[0][0], data[0][1], True
                elif key == "q":
                    _ImRecording = False
                    t.join()
                    return data[0][0], data[0][1], True
                    exit('program shutdown complete')


def record_part_with_midi(part_len_seconds=60, data_container=[]):
    """
    records a drum take with midi annotation
    :return:
    """
    global _ImRecording
    _ImRecording = True
    midi_result = []
    midiin = rtmidi.MidiIn()
    available_ports = midiin.get_ports()

    if available_ports:
        midiin.open_port(0)
        print(midiin.get_message())
    else:
        midiin.open_virtual_port("My virtual input")
    print("* recording")
    try:
        audio = sounddevice.rec(part_len_seconds * RATE)
        last_time = time.perf_counter()
        midi_time = time.perf_counter() - last_time
        while _ImRecording and midi_time < part_len_seconds:
            midi_msg = midiin.get_message()
            if midi_msg is not None:
                if midi_msg[0][0] == 153 and midi_msg[0][2] > 24:
                    midi_time = time.perf_counter() - last_time
                    goodmsg = [midi_time, midi_msg[0][1], midi_msg[0][2]]
                    print(goodmsg, midi_time)
                    midi_result.append(goodmsg)
        sounddevice.stop()
        _ImRecording = False
    except Exception as e:
        print("* recording stopped by user")
        _ImRecording = False
    # close streams parse everything and return
    finally:
        print("* done recording")
        # Midi parse:
        midi_time = np.empty(0)
        midi_key = np.empty(0)
        midi_vel = np.empty(0)
        for i in range(0, len(midi_result)):
            midi_time = np.append(midi_time, float(midi_result[i][0]))
            midi_key = np.append(midi_key, str(midi_result[i][1]))
            midi_vel = np.append(midi_vel, str(midi_result[i][2]))
        midi_file = pd.DataFrame(index=midi_time, columns=['key', 'vel'])
        midi_file['key'] = midi_key
        midi_file['vel'] = midi_vel
        data_container.append((audio, midi_file))
        return data_container[0]


def soundcheck_kit(kit_name=None, kit_size=None):
    for i in range(kit_size):
        print('record samples for drum: {}, press "r" to start recording, "s" to stop and save samples'.format(i))
        audio, midi_notes = soundcheck_listener()
        # get most frequently hit note
        note = int(midi_notes['key'].value_counts().idxmax())
        label = f_to_l_map[note]
        instrument_name = names_l_map[label]
        sc_audio_filename = './soundcheck/' + kitname + '_' + instrument_name + '.mp3'
        audio_segment = pydub.AudioSegment(
            audio.tobytes(),
            frame_rate=RATE,
            sample_width=2,
            channels=1
        )
        audio_segment.export(sc_audio_filename, format='mp3')
        print('done')


def remove_annotation_latency(audio, annotation, grain=66):
    """
    Midi input has latency, it is compensated by movin annotation a little bit.
    The optimal shift is determined by comparing onsets in the audio and annotation.
    Onsets and annotation locations are transformed to a signal by representing percussive
    audio response as gamma pdf around onset location.

    :param audio: audio data
    :param annotation: annotation
    :return: rectified annotation
    """
    # pick peaks from spectral difference ODF

    spectrogram = stft(audio)
    diff = spectrogram
    diff[1:, :] = np.diff(spectrogram, n=1, axis=0)
    spec_diff=np.sum(diff, axis=1)
    spec_diff=np.clip(spec_diff/max(spec_diff), 0,1)
    annotation_times = annotation.index.values
    annotation_frames=time_to_frame(annotation_times)
    annot_signal = np.zeros(spec_diff.size)
    for i in annotation_frames:
        if i < annot_signal.size:
            annot_signal[int(i) - 1] += .2
            annot_signal[int(i)] += .4
            annot_signal[int(i)+1] += .2
        else:
            #print(int(i))
    best_modifier = 0

    # slide over spec diff to find optimal match, split to small segments as midi latency fluctuates over time.
    k=15
    n=grain
    while k <spec_diff.size-n-30:
        best_corr = np.inf
        for i in range(30):
            #from [-15:15]
            modifier = i - 15
            #sum of absolute value of the similarity of the arrays
            correlation=np.sum(np.abs(np.add(spec_diff[k:k+n],-annot_signal[k-modifier:k+n-modifier])))
            if correlation < best_corr:
                best_corr = correlation
                best_modifier = modifier
        selection=(annotation_frames>k) & (annotation_frames<k+n)
        annotation_frames[selection]=annotation_frames[selection]+best_modifier
        k+=n
    new_annod = pd.DataFrame(index=frame_to_time(annotation_frames), data=annotation.values)
    return new_annod


def parse_argv(argv):
    kitname, kitsize, is_soundcheck = None, None, None
    if len(argv) < 1:
        exit(
            'usage: >python record_dataset.py <drum_kit_name [string]> [<drum_kit_size> [int]] [<perform_soundcheck> [1|0]] ')
    else:
        try:
            kitname = argv[1]
            try:
                kitsize = int(argv[2])
            except ValueError as e:
                print(e + ' please use integer to specify kit size')
            try:
                is_soundcheck = int(argv[3])
            except ValueError as e:
                print(e + ' please use integer 1 perform soundceck')
        except IndexError as e:
            pass
        finally:
            print(kitname, kitsize, is_soundcheck)
            return kitname, kitsize, is_soundcheck


if __name__ == '__main__':
    # todo write parse function
    kitname, kitsize, is_soundcheck = parse_argv(sys.argv)
    # todo create folders automatically for audio, annotation and soundcheck data
    if is_soundcheck == 1 and kitsize is not None:
        soundcheck_kit(kit_name=kitname, kit_size=kitsize)
        print('soundcheck done!')

    ####test rectification with specific file
    audio_file_name = './audio/' + 'kakka1569863379.48289.mp3'
    annotation_file_name = './annotation/' + 'kakka1569863379.48289.csv'
    annod = pd.read_csv(annotation_file_name, index_col=0, header=None, sep='\t')
    audio_segment = pydub.AudioSegment.from_mp3(audio_file_name)
    audio = np.array(audio_segment.get_array_of_samples())
    rect_name = './annotation/' + 'kakka_rect.csv'
    rect_annod = remove_annotation_latency(audio, annod, grain=99)
    rect_annod.to_csv(rect_name, index=True, header=False, sep="\t",
                              float_format='%.3f')
    exit('moido')
    exit_status = False
    while not exit_status:
        print('record drumtake, press "r" to start recording, "s" to stop and save samples, "q" to exit program')

        timestamp = time.time()
        audio_file_name = './audio/' + kitname + str(timestamp) + '.mp3'
        annotation_file_name = './annotation/' + kitname + str(timestamp) + '.csv'
        audio, midi_transcription, exit_status = drumpart_listener()
        audio_segment = pydub.AudioSegment(
            audio.tobytes(),
            frame_rate=RATE,
            sample_width=2,
            channels=1
        )
        audio_segment = effects.normalize(audio_segment)
        audio_segment.export(audio_file_name, format='mp3')
        midi_transcription.to_csv(annotation_file_name, index=True, header=False, sep="\t")
        rect_name = './annotation/' + kitname + str(timestamp) + '_rect_.csv'
        for i in range(3):
            midi_transcription=remove_annotation_latency(audio, midi_transcription)
        midi_transcription.to_csv(rect_name, index=True, header=False, sep="\t",
                                                                    float_format='%.3f')
