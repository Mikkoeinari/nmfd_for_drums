'''
Modified from Keras GRU layer and GRU cell
Chollet, François et. al. "Keras" 2015
https://keras.io
'''

import keras.backend as K
from keras import activations, regularizers, initializers, constraints
from keras.layers import GRU, GRUCell
import numpy as np

def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)


class MGUCell(GRUCell):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 reset_after=False,
                 **kwargs):
        super(MGUCell, self).__init__(units, **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 2),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 2),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (2 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, 2 * self.units)
            self.bias = self.add_weight(shape=bias_shape,
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if not self.reset_after:
                self.input_bias, self.recurrent_bias = self.bias, None
            else:
                # NOTE: need to flatten, since slicing in CNTK gives 2D array
                self.input_bias = K.flatten(self.bias[0])
                self.recurrent_bias = K.flatten(self.bias[1])
        else:
            self.bias = None

        # update gate, for mgu: forget gate
        self.kernel_f = self.kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, :self.units]
        # reset gate
        # self.kernel_r = self.kernel[:, self.units: self.units * 2]
        # self.recurrent_kernel_r = self.recurrent_kernel[:,
        #                          self.units:
        #                          self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units:self.units * 2]
        if self.implementation==1:
            if self.use_bias:
                # bias for inputs
                self.input_bias_f = self.input_bias[:self.units]
                # self.input_bias_r = self.input_bias[self.units: self.units * 2]
                self.input_bias_h = self.input_bias[self.units: self.units * 2]
                # bias for hidden state - just for compatibility with CuDNN
                if self.reset_after:
                    self.recurrent_bias_f = self.recurrent_bias[:self.units]
                    # self.recurrent_bias_r = self.recurrent_bias[self.units: self.units * 2]
                    self.recurrent_bias_h = self.recurrent_bias[self.units: self.units * 2]
            else:
                self.input_bias_f = None
                # self.input_bias_r = None
                self.input_bias_h = None
                if self.reset_after:
                    self.recurrent_bias_f = None
                    # self.recurrent_bias_r = None
                    self.recurrent_bias_h = None
        elif self.implementation==2:
            if self.use_bias:
                self.input_bias_h = self.input_bias[self.units: self.units * 2]
                # bias for hidden state - just for compatibility with CuDNN
                if self.reset_after:
                    self.recurrent_bias_h = self.recurrent_bias[self.units: self.units * 2]
            else:
                self.input_bias_h = None
                if self.reset_after:
                    self.recurrent_bias_h = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=3)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_f = inputs * dp_mask[0]
                inputs_h = inputs * dp_mask[1]
            else:
                inputs_f = inputs
                inputs_h = inputs

            x_f = K.dot(inputs_f, self.kernel_f)
            x_h = K.dot(inputs_h, self.kernel_h)
            if self.use_bias:
                x_f = K.bias_add(x_f, self.input_bias_f)
                x_h = K.bias_add(x_h, self.input_bias_h)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_f = h_tm1 * rec_dp_mask[0]
                h_tm1_h = h_tm1 * rec_dp_mask[1]
            else:
                h_tm1_f = h_tm1
                h_tm1_h = h_tm1

            recurrent_f = K.dot(h_tm1_f, self.recurrent_kernel_f)
            if self.reset_after and self.use_bias:
                recurrent_f = K.bias_add(recurrent_f, self.recurrent_bias_f)

            f = self.recurrent_activation(x_f + recurrent_f)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, self.recurrent_bias_h)

            else:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
            hh = self.activation(x_h + recurrent_h)

        # MGU2 with reduced forget gate
        elif self.implementation == 2:
            if 0. < self.dropout < 1.:
                inputs_h = inputs * dp_mask[1]
            else:
                inputs_h = inputs

            x_h = K.dot(inputs_h, self.kernel_h)
            if self.use_bias:
                x_h = K.bias_add(x_h, self.input_bias_h)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_f = h_tm1 * rec_dp_mask[0]
                h_tm1_h = h_tm1 * rec_dp_mask[1]
            else:
                h_tm1_f = h_tm1
                h_tm1_h = h_tm1

            recurrent_f = K.dot(h_tm1_f, self.recurrent_kernel_f)

            f = self.recurrent_activation(recurrent_f)

            # forget gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, self.recurrent_bias_h)
            else:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
            hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = f * h_tm1 + (1 - f) * hh

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation,
                  'reset_after': self.reset_after}
        base_config = super(GRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MGU(GRU):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 **kwargs):
        super(MGU, self).__init__(units,
                                  activation=activation,
                                  recurrent_activation=recurrent_activation,
                                  use_bias=use_bias,
                                  kernel_initializer=kernel_initializer,
                                  recurrent_initializer=recurrent_initializer,
                                  bias_initializer=bias_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  recurrent_regularizer=recurrent_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  activity_regularizer=activity_regularizer,
                                  kernel_constraint=kernel_constraint,
                                  recurrent_constraint=recurrent_constraint,
                                  bias_constraint=bias_constraint,
                                  dropout=dropout,
                                  recurrent_dropout=recurrent_dropout,
                                  implementation=implementation,
                                  return_sequences=return_sequences,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  reset_after=reset_after,
                                  **kwargs)
        self.cell = MGUCell(units,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            recurrent_initializer=recurrent_initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            recurrent_regularizer=recurrent_regularizer,
                            bias_regularizer=bias_regularizer,
                            kernel_constraint=kernel_constraint,
                            recurrent_constraint=recurrent_constraint,
                            bias_constraint=bias_constraint,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            implementation=implementation,
                            reset_after=reset_after)

    def prune_weights(self, prune_percentile):
        """
        Experimental weight pruning method, purely testing purposes, results erratic.
        :param prune_percentile: int, The percentile below the weights will be reinitialized
        :return: None, reinitialization is done in place.
        """
        weights=self.cell.get_weights()
        #for i in range(len(weights)):
        i=0
        prune=weights[i]<np.percentile(weights[i], prune_percentile)
        if self._initial_weights is not None:
            weights[i][prune]=self._initial_weights[i][prune]
        else:
            self._initial_weights=weights
            #summed=np.sum(weights[i], axis=0)
            #lowest_w=np.argmin(summed)
            #weights[i][:,lowest_w]=0
        self.cell.set_weights(weights)

