import math

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import *
from keras import backend as kb
import tensorflow as tf
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from python.MGU import MGU
from python.utils import stft, time_to_frame
from madmom.io.audio import load_audio_file
import pandas as pd
seqLen =100
stft_bands = 48
kit_size = 3
nr='gen'

def get_sequences(spectrogram, framed_annotation, shuffle_sequences=False, n_samples=0):
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
    spectrogram_padded = np.concatenate((np.concatenate((np.zeros((seqLen, 48)), spectrogram)), np.zeros((seqLen, 48))))
    # split spectrogram into sequences
    for start in range(spectrogram_padded.shape[0] - seqLen):
        sequences.append(spectrogram_padded[start:start + seqLen, :])
        targets.append(list(target))

    # insert drumhits into targets
    y = np.array(targets)
    for index, drum in framed_annotation:
        try:
            y[index][drum] = 1
        except IndexError as ie:
            print('error: ', ie, index, drum, y.shape)
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


def make_model(model_type='many2many'):
    if model_type == 'many2many':
        layer_size = 32
        dense_layer_size = 32

        def get_drum_layer(seqLen):
            in1 = Input(shape=(seqLen,))
            rs1 = Reshape((1, seqLen, 1))(in1)
            #conv layers

            rs1 = TimeDistributed(Conv1D(32, 3, activation='relu'))(rs1)

            rs1 = TimeDistributed(SpatialDropout1D(.5))(rs1)
            #rs1 = TimeDistributed(BatchNormalization())(rs1)

            rs1 = TimeDistributed(Flatten())(rs1)
            #recurrent layer
            mgu1 = (MGU(layer_size, activation='tanh',
                        return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1))(rs1)
            return [in1, mgu1]

        def get_out_layer(in_layer):
            in_layer = Dense(dense_layer_size, activation='relu')(in_layer)
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
        optr = adam(lr=0.001)
        model.compile(loss=focal_loss(), metrics=['accuracy'], optimizer=optr)
        return model

class DataGenerator(object):
    def __init__(self,batch_size=32, shuffle_sequences=False, audio_files=None, annotation_files=None):
        """
        initialize class, pin producer
        :param image_producer: object, the producer of images
        """

        self.batch_size=batch_size
        self.shuffle_sequences=shuffle_sequences
        self.audio_files=audio_files
        self.annotation_files=annotation_files

    def get_generator(self):
        """
        build a keras compatible generator here
        :return: generator object
        """
        while True:
            sequences = []
            targets = []
            while len(targets)<self.batch_size:
                #shuffle files??
                audio_file, annotation_file = resample(self.audio_files,
                                                           self.annotation_files,
                                                           n_samples=1,
                                                           replace=True)
                buffer, sr = load_audio_file(audio_file[0])
                buffer = buffer / np.max(np.abs(buffer))
                if len(buffer.shape) > 1:
                    buffer = buffer[:, 0] + buffer[:, 1]

                filt_spec = stft(buffer)
                hits = pd.read_csv(annotation_file[0], sep="\t", header=None)
                hits.iloc[:, 0] = time_to_frame(hits.iloc[:, 0], sr, 441).astype(int)
                # empty target
                target = np.zeros(kit_size, dtype=np.int)
                # pad edges
                spectrogram_padded = np.concatenate(
                    (np.concatenate((np.zeros((seqLen, 48)), filt_spec)), np.zeros((seqLen, 48))))
                # split spectrogram into sequences
                for start in range(spectrogram_padded.shape[0] - seqLen):
                    sequences.append(spectrogram_padded[start:start + seqLen, :])
                    targets.append(list(target))

            # insert drumhits into targets
            y = np.array(targets)
            for index, drum in hits.values:
                try:
                    y[index][drum] = 1
                except IndexError as ie:
                    print('error: ', ie, index, drum, y.shape)
            X = np.array(sequences)
            #shuffle a batch size of samples
            X, y = resample(np.array(X), np.array(y), n_samples=self.batch_size, replace=False)
            yield list(np.transpose(X, (2, 0, 1))), list(y.T)

def make_generators(audio_takes, annotation):
    train_audios,val_audios,train_annotations,val_annotations=train_test_split(audio_takes,
                                                                               annotation,test_size=0.15,
                                                                               random_state=0)
    train_generator=DataGenerator(batch_size=32,
                                  shuffle_sequences=True,
                                  audio_files=train_audios,
                                  annotation_files=train_annotations).get_generator()
    val_generator=DataGenerator(batch_size=32,
                                  shuffle_sequences=True,
                                audio_files=val_audios,
                                annotation_files=val_annotations).get_generator()
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

    if generators[0] is not None:
        epoch_steps=500
        val_steps=100
        model.fit_generator(generator=generators[0], epochs=1000,
                            steps_per_epoch=epoch_steps, validation_steps=val_steps,
                            callbacks=[modelsaver, earlystopper],
                            validation_data=generators[1],
                            verbose=1)
    else:
        X, y = data
        model.fit(X, y, batch_size=32,
                  epochs=100,
                  callbacks=[modelsaver, earlystopper],#, lr_schedule
                  validation_split=0.15,
                  verbose=1)

    model.load_weights("{}weights__{}.hdf5".format('test', nr))


def save_model(model):
    model.save('{}model_{}.hdf5'.format('test', nr))


def load_saved_model():
    model = load_model('{}model_{}.hdf5'.format('test', nr),
                       custom_objects={'MGU': MGU, 'focal_loss_fixed': focal_loss()})

    return model


def restore_best_weigths(model):
    model.load_weights("{}weights__{}.hdf5".format('test', nr))
    return model


def predict_odf(model, data):
    odf = model.predict(data)
    return odf

# g0,a.85
# precision= 0.6756640848570448
# recall= 0.7521708661063248
# f-score= 0.6916115122385045
# Run time: 278.27399015426636

# g.5, a.75
# precision= 0.684371810015429
# recall= 0.740939304735122
# f-score= 0.6908751484347703
# Run time: 244.68893599510193


# g1., a0.625
# precision= 0.6859974611181358
# recall= 0.7092171746201668
# f-score= 0.6732353736808586
# Run time: 287.6512060165405

# g1.333, a0.5
# precision= 0.6927943149600362
# recall= 0.6645405934753582
# f-score= 0.6557779894264516
# Run time: 289.59490299224854

#3,3conv1d->flatten->32mgu->48d->1d
#precision= 0.9231552327663176
#recall= 0.9442570229741587
#f-score= 0.9301528737381303
#Run time: 4519.877339839935
#seq50 do0.50 loss 0.95 väliveto +-20ms 32d->1d
#precision= 0.9458202918109844
#recall= 0.9229382572526305
#f-score= 0.9281062154666473
#loss0.86420
#precision= 0.9398639613818435
#recall= 0.9226179453934024
#f-score= 0.9241900407141712
#loss0.81178
#precision= 0.9352944412573825
#recall= 0.9174232102249822
#f-score= 0.9218213845201506

#3x3conv->flat->12mgu->10d->1d
#precision= 0.9289545072772806
#recall= 0.9276646132842606
#f-score= 0.920966951792604
#precision= 0.937893897173712
#recall= 0.9299532060350575
#f-score= 0.9294476282494156
#0.85 loss
#precision= 0.9420606895696831
#recall= 0.9208570976455139
#f-score= 0.9248947839651388
#3x3conv->flat->8mgu->8d->1d 3x0.5do
#precision= 0.9127577299435182
#recall= 0.9196900732911548
#f-score= 0.9106610931744288

#32,3conv1d->flatten->32mgu->48d->1d do0.333 seq20
#precision= 0.9556980450633245
#recall= 0.9432519012567497
#f-score= 0.9454844795808248
#further trained
#precision= 0.9338271696191057
#recall= 0.9557806310467164
#f-score= 0.9398052933463275

#2x32,3conv1d->1x1conv1d->bn->flatten->32mgu->48d->1d do0.5 seq100
#0.90472 lossilla epoch 8, thresh 0.8
#precision= 0.9283913031564432
#recall= 0.9424096617222243
#f-score= 0.9302320576200963
#0.82782 lossilla epoch 15, thresh 0.75
#precision= 0.9368701897948395
#recall= 0.9268549186392996
#f-score= 0.9269329869440824
#tolerance 20ms
#precision= 0.9117307596338388
#recall= 0.9028615464143522
#f-score= 0.9025782585771673


#32,3conv1d->flatten->32mgu->48d->1d do0.50 seq20 a
#precision= 0.9412581299998377
#recall= 0.9314316360591649
#f-score= 0.9334029529949046
#precision= 0.9181600833636373
#recall= 0.9283493756443848
#f-score= 0.9204614871750004
#seq50 32d->1d--0.84907----------------------------------
#precision= 0.9345031233101175
#recall= 0.9350381806167339
#f-score= 0.9316346010372212
#0.101 loss do0.75
#precision= 0.9307373878962906
#recall= 0.9250013498737119
#f-score= 0.9228149007705141

#8,3conv1d->bn->flatten->32mgu->8d->1d do0.50 seq20 c 0.799...
#precision= 0.9281937494817759
#recall= 0.9232684453760764
#f-score= 0.9219220709359152

#8,3conv1d->bn->flatten->64mgu->48d->1d do0.50 seq20 d 0.86
#precision= 0.9305651399912972
#recall= 0.9232634717645652
#f-score= 0.9226769861634739

#16,3conv1d(stride2)->16,3conv1d->1x1conv1d->bn->flatten->32mgu->48d->1d do0.50 seq20 e
#precision= 0.8114087246101317 loss 1.16281
#recall= 0.8336524707477689
#f-score= 0.8139380932775163

#Deep stab-> f
#precision= 0.9165059255045263 loss 0.94217
#recall= 0.8983023555610843
#f-score= 0.9008169144377003
#precision= 0.9311755752227113 loss 0.81697
#recall= 0.9093249473282249
#f-score= 0.9160085187409079

#shallow 32x3/2->3,3/1->64x1/1 a
#Enst:
#precision= 0.5515706402364641 loss 1.11063
#recall= 0.7056448598216134
#f-score= 0.6055281740104498

#12,12,12 loss0.9 a
#precision= 0.9181600833636373
#recall= 0.9283493756443848
#f-score= 0.9204614871750004

#16 16 16 loss 0.82 do 0.625 batch32
#precision= 0.9359601133792943
#recall= 0.9320198656459021
#f-score= 0.9292351794961418
#Enst suoraan perään
#precision= 0.8719207371156754
#recall= 0.5809040103233627
#f-score= 0.6637609464776343

#161616 no_lrcshed seq 50
#precision= 0.9368900008172737
#recall= 0.932227835404062
#f-score= 0.9288986607241286
#Run time: 5679.843251943588
#seq100
#precision= 0.9389481688967344
#recall= 0.9167156371127877
#f-score= 0.9216351926836277
#Run time: 7058.523633003235

#gen