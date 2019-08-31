import math

import pandas as pd
import tensorflow as tf
from keras import backend as kb
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import *
from madmom.io.audio import load_audio_file
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from python.MGU import MGU
from python.utils import stft, time_to_frame

seqLen = 100
stft_bands = 96
kit_size = 3
nr = 'gen32midi_diff(s100)_2nd'


def get_sequences(spectrogram=None, framed_annotation=None, shuffle_sequences=False, n_samples=0):
    """
    splits data into sequences
    :param spectrogram: Spectrogram containing the audio
    :param framed_annotation: annotation with frameresolution indexing
    :param shuffle_sequences: boolean, weather to shuffle the sequences
    :return: X, y: data sequences and matching targets
    """
    sequences = []
    targets = []
    # empty target
    target = np.zeros(kit_size, dtype=np.int)
    # pad edges
    if spectrogram is not None:
        spectrogram_padded = np.concatenate((np.concatenate((np.zeros((seqLen, stft_bands)), spectrogram)), np.zeros((seqLen, stft_bands))))
    else:
        return None, None #if audio file can not be loaded, skip it.
    # split spectrogram into sequences
    for start in range(spectrogram_padded.shape[0] - seqLen):
        sequences.append(spectrogram_padded[start:start + seqLen, :])
        targets.append(list(target))

    # insert drumhits into targets
    y = np.array(targets)
    if framed_annotation is not None:
        for index, drum in framed_annotation:
            try:
                y[index][drum] = 1
            except IndexError as ie:
                pass #skip annoying error messages
                #print('error: ', ie, index, drum, y.shape)
    X = np.array(sequences)
    if shuffle_sequences:
        X, y = resample(np.array(X), np.array(y), n_samples=n_samples, replace=False, random_state=0)
    return np.transpose(X, (2, 0, 1)), y.T


def focal_loss(gamma=.5, alpha=.75):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = kb.epsilon()

        pt_1 = kb.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = kb.clip(pt_0, epsilon, 1. - epsilon)

        return -kb.sum(alpha * kb.pow(1. - pt_1, gamma) * kb.log(pt_1)) \
               - kb.sum((1 - alpha) * kb.pow(pt_0, gamma) * kb.log(1. - pt_0))

    return focal_loss_fixed


def biased_bin_cross_loss(alpha=.8):
    def bbc_loss(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = kb.epsilon()

        pt_1 = kb.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = kb.clip(pt_0, epsilon, 1. - epsilon)

        return -kb.mean(alpha * kb.log(pt_1)) \
               - kb.mean((1 - alpha) * kb.log(1. - pt_0))

    return bbc_loss


def make_model(model_type='many2many'):
    if model_type == 'many2many':
        layer_size = 32
        dense_layer_size = 32

        def get_drum_layer(seqLen):
            in1 = Input(shape=(seqLen,))
            rs1 = Reshape((1, seqLen, 1))(in1)
            # conv layers

            rs1 = TimeDistributed(Conv1D(32, 3, activation='relu'))(rs1)
            #rs1 = TimeDistributed(Conv1D(32, 3, activation='relu'))(rs1)

            #rs1 = TimeDistributed(SpatialDropout1D(.33))(rs1)
            # rs1 = TimeDistributed(BatchNormalization())(rs1)

            rs1 = TimeDistributed(Flatten())(rs1)
            # recurrent layer
            mgu1 = (MGU(layer_size, activation='tanh',
                        return_sequences=False, dropout=0.33, recurrent_dropout=0.33, implementation=1))(rs1)
            return [in1, mgu1]

        def get_out_layer(in_layer):
            in_layer = Dense(dense_layer_size, activation='relu')(in_layer)
            #in_layer =Dropout(.5)(in_layer)
            dense2 = Dense(1, kernel_initializer='he_normal', activation='sigmoid')(in_layer)

            return [in_layer, dense2]

        in_layers = []
        for i in range(stft_bands):
            in_layers.append(get_drum_layer(seqLen))
        in_layers = np.array(in_layers)
        merged = Concatenate()(list(in_layers[:, 1]))
        # merged = Dense(dense_layer_size, activation='relu')(merged)

        out_layers = []
        for i in range(kit_size):
            out_layers.append(get_out_layer(merged))
        out_layers = np.array(out_layers)
        model = Model(list(in_layers[:, 0]), list(out_layers[:, 1]))

        print(model.summary())
        optr = adam(lr=0.0005)
        model.compile(loss=biased_bin_cross_loss(), metrics=['accuracy'], optimizer=optr)
        return model


class DataGenerator(object):
    def __init__(self, batch_size=32, shuffle_sequences=False, audio_files=None,
                 annotation_files=None, stratify=False):
        """

        :param batch_size:
        :param shuffle_sequences:
        :param audio_files:
        :param annotation_files:
        """

        self.batch_size = batch_size
        self.shuffle_sequences = shuffle_sequences
        self.audio_files = audio_files
        self.annotation_files = annotation_files
        self.stratify=stratify

    def get_generator(self):
        """
        build a keras compatible generator here
        :return: generator object
        """
        while True:
            sequences = []
            targets = []
            while len(targets) < self.batch_size:
                # shuffle files??
                audio_file, annotation_file = resample(self.audio_files,
                                                       self.annotation_files,
                                                       n_samples=1,
                                                       replace=True)
                try:
                    buffer, sr = load_audio_file(audio_file[0])
                except:
                    continue #Skip broken files
                buffer = buffer / np.max(np.abs(buffer))
                if len(buffer.shape) > 1:
                    buffer = buffer[:, 0] + buffer[:, 1]

                filt_spec = stft(buffer, add_diff=True)
                empty_data=False
                try:
                    hits = pd.read_csv(annotation_file[0], sep="\t", header=None)
                    hits.iloc[:, 0] = time_to_frame(hits.iloc[:, 0], sr, 441).astype(int)
                except Exception as e:
                    empty_data=True
                    #empty csv data error, disregard that and continue with all zero annotation
                # empty target
                target = np.zeros(kit_size, dtype=np.int)
                # pad edges
                spectrogram_padded = np.concatenate(
                    (np.concatenate((np.zeros((seqLen, stft_bands)), filt_spec)), np.zeros((seqLen, stft_bands))))
                # split spectrogram into sequences
                for start in range(spectrogram_padded.shape[0] - seqLen):
                    sequences.append(spectrogram_padded[start:start + seqLen, :])
                    targets.append(list(target))

            # insert drumhits into targets
            y = np.array(targets)
            if not empty_data:
                for index, drum in hits.values:
                    try:
                        y[index][drum] = 1
                    except IndexError as ie:
                        pass  # skip annoying error messages
                        # print('error: ', ie, index, drum, y.shape)
            X = np.array(sequences)
            # stratify training batch, first select y=false rows
            strat_x=X
            strat_y=y
            if self.stratify:
                try:
                    indexes = np.where(y[:, :]==0)[0]
                    if indexes.shape[0] < 1: #If there is not enough data to begin with
                        continue
                    selection = np.random.choice(indexes, self.batch_size*3)
                except IndexError as ie:
                    print(selection, ie)
                strat_x = list(X[selection])
                strat_y = list(y[selection])
                # then rows where y=True
                for i in range(kit_size):
                    indexes = np.where(y[:, i] == i)[0]
                    if indexes.shape[0] >0: #if there is not all drums present, skip empties and continue
                        try:
                            indexes = np.where(y[:, i] == i)[0]
                            selection = np.random.choice(indexes,self.batch_size)
                        except IndexError as ie:
                            print(ie)
                        strat_x.extend(list(X[selection]))
                        strat_y.extend(list(y[selection]))

            # shuffle a batch size of stratified samples
            X, y = resample(np.array(strat_x), np.array(strat_y), n_samples=self.batch_size, replace=False)
            yield list(np.transpose(X, (2, 0, 1))), list(y.T)


def make_generators(audio_takes, annotation):
    train_audios, val_audios, train_annotations, val_annotations = train_test_split(audio_takes,
                                                                                    annotation, test_size=0.15,
                                                                                    random_state=0)
    train_generator = DataGenerator(batch_size=32,
                                    shuffle_sequences=True,
                                    audio_files=train_audios,
                                    annotation_files=train_annotations, stratify=False).get_generator()
    val_generator = DataGenerator(batch_size=32,
                                  shuffle_sequences=True,
                                  audio_files=val_audios,
                                  annotation_files=val_annotations, stratify=False).get_generator()
    return train_generator, val_generator


def train_model(model, data=(None, None), generators=None):
    # Learning rate step scheduler
    def step_decay(epoch):
        initial_lrate = 0.0005
        drop = 0.5
        epochs_drop = 7.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    lr_schedule = LearningRateScheduler(step_decay)
    modelsaver = ModelCheckpoint(filepath="{}weights__{}.hdf5".format('test', nr), verbose=1,
                                 save_best_only=True)
    earlystopper = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=5, mode='auto')
    model.load_weights("{}weights__{}.hdf5".format('test', 'gen32midi_diff(s100)'))
    if generators[0] is not None:
        epoch_steps = 256
        val_steps = 128
        model.fit_generator(generator=generators[0], epochs=1000,
                            steps_per_epoch=epoch_steps, validation_steps=val_steps,
                            callbacks=[modelsaver, earlystopper],
                            validation_data=generators[1],
                            verbose=1)
    else:
        X, y = data
        model.fit(X, y, batch_size=32,
                  epochs=100,
                  callbacks=[modelsaver, earlystopper],  # , lr_schedule
                  validation_split=0.15,
                  verbose=1)

    model.load_weights("{}weights__{}.hdf5".format('test', nr))


def save_model(model):
    model.save('{}model_{}.hdf5'.format('test', nr))


def load_saved_model():
    model = load_model('{}model_{}.hdf5'.format('test', nr),
                       custom_objects={'MGU': MGU,
                                       'focal_loss_fixed': focal_loss(),
                                       'bbc_loss': biased_bin_cross_loss()})

    return model


def restore_best_weigths(model):
    model.load_weights("{}weights__{}.hdf5".format('test', nr))
    return model


def predict_odf(model, data):
    odf = model.predict(data)
    return odf
