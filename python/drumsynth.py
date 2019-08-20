'''
This module contains audio output related functionality
notes: migrate to scipy.io.wavfile to avoid format problems
'''
import pandas as pd
import numpy as np
import drum_off.utils as utils
import pyaudio
import wave
from scipy.io import wavfile

#global variable to stop all playback
#todo remove these!!!
_ImRunning = False
_ImRecording = False
#we store an arbitrary sample and use it's properties to init player
bd=None

def open_and_merge(filename):
    """
    read bytes from sample files
    :param filename: String, sample wav file
    :return: String, a string of bytes
    """
    global bd
    x=wave.open(filename, 'rb')
    bd=x
    x = x.readframes(x.getnframes())
    return x

#Read samples and store to sounds list
bdBytes = open_and_merge('./Sounds/bigkit/mono/bd.wav')
sdBytes = open_and_merge('./Sounds/bigkit/mono/sn.wav')
chhBytes = open_and_merge('./Sounds/bigkit/mono/chh.wav')
ohhBytes = open_and_merge('./Sounds/bigkit/mono/ohh.wav')
shhBytes = open_and_merge('./Sounds/bigkit/mono/shh.wav')
ttBytes = open_and_merge('./Sounds/bigkit/mono/tth.wav')
tt2Bytes = open_and_merge('./Sounds/bigkit/mono/tt.wav')
tt3Bytes = open_and_merge('./Sounds/bigkit/mono/tt2.wav')
ftBytes = open_and_merge('./Sounds/bigkit/mono/ft.wav')
ft2Bytes = open_and_merge('./Sounds/bigkit/mono/ft2.wav')
rdBytes = open_and_merge('./Sounds/bigkit/mono/rd.wav')
crBytes = open_and_merge('./Sounds/bigkit/mono/cr1.wav')
cr2Bytes = open_and_merge('./Sounds/bigkit/mono/cr2.wav')
cr3Bytes = open_and_merge('./Sounds/bigkit/mono/cr3.wav')

#Small kit sounds
#sounds = [bdBytes, sdBytes, chhBytes, ohhBytes, ttBytes, ftBytes, rdBytes, crBytes, shhBytes]
#Allow more drums as per midinotes:[36, 38, 42, 46, 44, 50, 48, 47, 43, 41, 51, 49, 57, 55]
sounds = [bdBytes, sdBytes, chhBytes, ohhBytes, shhBytes, ttBytes, tt2Bytes, tt3Bytes, ftBytes, ft2Bytes, rdBytes, crBytes, cr2Bytes, cr3Bytes]

def frame_to_time(frames, sr=bd.getframerate()):
    """
    Transforms frame numbers to time values
    :param frames: list of integers to transform
    :param sr: int, Sample rate of the FFT
    :param hop_length: int, Hop length of FFT
    :param hops_per_frame: ??
    :return: Numpy array of time values
    """
    # samples = (np.asanyarray(frames) * (hop_length / hops_per_frame)).astype(int)
    return frames / float(sr)

def createWav(filePath, outName=None, addCountInAndCountOut=True, deltaTempo=1.0, countTempo=1.0):
    """
    Creates a wav file from a notation .csv file
    :param filePath: String, the csv file that contains drum notation.
    :param outName: SString, filename of the output wav
    :param addCountInAndCountOut: Boolean, if True the wav will contain 8 pedal hi-hat hits as count in and count out.
    :param deltaTempo: Float, the original tempo of the notation before quantization to 120BPM
    :return: String, filename of the created Wav file
    """
    #print(addCountInAndCountOut)
    if outName is None:
        outName='./default.wav'
    d = pd.read_csv(filePath, header=None, sep="\t").values
    #d = list(utils.truncZeros(np.array(d[:, 1])))
    d=list(d[:,1])
    #todo: make sure there is notes in the file
    if addCountInAndCountOut:
        c=pd.read_csv('countIn.csv', header=None, sep="\t").values
        #c = list(utils.truncZeros(np.array(c[:, 1])))
        c=list(c[:,1])
        for i in range(len(c)):
            if c[i]<0:
                c[i]=c[i]/countTempo
        d=c+d+c

    d = utils.splitrowsanddecode(d, deltaTempo)
    gen = pd.DataFrame(d, columns=['time', 'inst'])
    gen['time'] = utils.frame_to_time(gen['time'],sr=int(utils.SAMPLE_RATE/2), hop_length=int(utils.Q_HOP))
    d = gen.values
    # file pointer
    cursor = 0
    if d.size<=1:
        return None
    # max file size
    fileLength = int((d[-1][0] * 44100 / 8) + len(sounds[int(d[-1][1])]))
    outfile = np.zeros((fileLength,), dtype=int)
    outfile = bytearray(outfile.tobytes())
    for i in range(len(d)):
        drum = int(d[i][1])

        if i != 0:
            gap = d[i][0] - d[i - 1][0]

            gapLen = int(np.rint(gap * 44100))
            cursor = cursor + gapLen

        endCursor = cursor + (len(sounds[drum]))
        data1 = np.fromstring(bytes(outfile[cursor:endCursor]), np.int16)
        data2 = np.fromstring(sounds[drum], np.int16)
        outfile[cursor:endCursor] = ((data1 * .5 + data2 * .5)).astype(np.int16).tostring()
    # fix result size to 10s.

    outfile = outfile[:cursor]
    wauva = wave.open(outName, 'w')

    # pitääkö tsekata sampleja lukiessa???
    wauva.setparams((bd.getnchannels(), bd.getsampwidth(), bd.getframerate(),0, 'NONE', 'not compressed'))
    wauva.writeframes(outfile)
    wauva.close()
    print('file ready...')
    return outName

def playWav(filePath):
    """
    Play back wav file
    :param filePath: String, the source file
    :return: None
    """

    global _ImRunning
    _ImRunning = True

    #Convert float32 wav to int16 bit depth.
    file_sample_rate, wf=wavfile.read(filePath)
    if wf.dtype==np.float32:
        wf=wf.flatten()
        wf=wf*float(32768)
        np.clip(wf, -32768, 32767, out=wf)
        wf=wf.astype(np.int16)
        print(wf.shape, wf.max(), wf.min())
        wavfile.write(filePath,file_sample_rate, wf)

    wf = wave.open(filePath, 'rb')

    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)

        return (data, pyaudio.paContinue)

    stream = p.open(format=p.get_format_from_width(bd.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                    #stream_callback=callback
                    )
    data = wf.readframes(1024)

    # play stream (3)
    while len(data) > 0 and _ImRunning:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()

    stream.close()

    wf.close()

    p.terminate()


def playFile(filePath, *args):
    """
    Create wav file and play it back.
    :param filePath: String, the source file
    :return: an instance of playFile
    """

    wavFile=createWav(filePath, *args)
    return playFile(wavFile)


def record_part(part_len_seconds=30):
    """
    records a drum take
    :return:
    """
    global _ImRecording
    _ImRecording = True
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * part_len_seconds)):
        if not _ImRecording:
            break
        data = stream.read(CHUNK)
        frames.extend(np.fromstring(data, np.int16))
    #add a second to prevent tails from going over the frames length
    frames.extend(np.zeros(44100, dtype=np.int16))
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    return  np.array(frames)

def record_stream():
    """
    records a drum take
    :return:
    """
    global _ImRecording
    _ImRecording = True
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    frames = []
    while _ImRecording:
        if not _ImRecording:
            break
        data = stream.read(CHUNK)
        #yield data
        yield np.fromstring(data, np.int16)

