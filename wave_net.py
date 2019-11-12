import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from keras.optimizers import Adam

class WaveNet:
    def __init__(self, seq_length, signal_dims):
        self.seq_length = seq_length
        self.signal_dims = signal_dims,

    def get_model(self):
        # convolutional operation parameters
        n_filters = 32  # 32
        filter_width = 2
        dilation_rates = [2 ** i for i in range(8)] * 2

        # define an input history series and pass it through a stack of dilated causal convolution blocks.
        history_seq = Input(shape=(None, *self.signal_dims))
        x = history_seq

        skips = []
        for dilation_rate in dilation_rates:
            # preprocessing - equivalent to time-distributed dense
            x = Conv1D(16, 1, padding='same', activation='relu')(x)

            # filter convolution
            x_f = Conv1D(filters=n_filters,
                         kernel_size=filter_width,
                         padding='causal',
                         dilation_rate=dilation_rate)(x)

            # gating convolution
            x_g = Conv1D(filters=n_filters,
                         kernel_size=filter_width,
                         padding='causal',
                         dilation_rate=dilation_rate)(x)

            # multiply filter and gating branches
            z = Multiply()([Activation('tanh')(x_f),
                            Activation('sigmoid')(x_g)])

            # postprocessing - equivalent to time-distributed dense
            z = Conv1D(16, 1, padding='same', activation='relu')(z)

            # residual connection
            x = Add()([x, z])

            # collect skip connections
            skips.append(z)

        # add all skip connection outputs
        out = Activation('relu')(Add()(skips))

        # final time-distributed dense layers
        out = Conv1D(128, 1, padding='same')(out)
        out = Activation('relu')(out)
        out = Dropout(.2)(out)
        out = Conv1D(1, 1, padding='same')(out)

        # extract the last seq_length time steps as the training target
        def slice(x, seq_length):
            return x[:, -seq_length:, :]

        pred_seq_train = Lambda(slice, arguments={'seq_length': self.seq_length})(out)

        model = Model(history_seq, pred_seq_train)
        model.compile(Adam(), loss='mean_absolute_error')
        self.model = model
        return model

    def _predict_sequence(self, input_sequence):

        history_sequence = input_sequence.copy()
        pred_sequence = np.zeros((1, self.seq_length, 1))  # initialize output (pred_steps time steps)

        for i in range(self.seq_length):
            # record next time step prediction (last time step of model output)
            last_step_pred = self.model.predict(history_sequence)[0, -1, 0]
            pred_sequence[0, i, 0] = last_step_pred

            # add the next time step prediction to the history sequence
            history_sequence = np.concatenate([history_sequence,
                                               last_step_pred.reshape(-1, 1, 1)], axis=1)

        return pred_sequence

    def predict_and_plot(self, encoder_input_data, decoder_target_data, sample_ind, enc_tail_len=50):

        encode_series = encoder_input_data[sample_ind:sample_ind + 1, :, :]
        pred_series = self._predict_sequence(encode_series)

        encode_series = encode_series.reshape(-1, 1)
        pred_series = pred_series.reshape(-1, 1)
        target_series = decoder_target_data[sample_ind, :, :1].reshape(-1, 1)

        encode_series_tail = np.concatenate([encode_series[-enc_tail_len:], target_series[:1]])
        x_encode = encode_series_tail.shape[0]

        plt.figure(figsize=(10, 6))

        plt.plot(range(1, x_encode + 1), encode_series_tail)
        plt.plot(range(x_encode, x_encode + self.seq_length), target_series, color='orange')
        plt.plot(range(x_encode, x_encode + self.seq_length), pred_series, color='teal', linestyle='--')

        plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
        plt.legend(['Encoding Series', 'Target Series', 'Predictions'])

if __name__ == '__main__':
    wvnet = WaveNet(60, 1).get_model()
