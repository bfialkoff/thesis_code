import tensorflow as tf
import keras.backend as K
import numpy as np
import keras
from keras import Model
from keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Bidirectional

from robust_adaptative_loss import RobustAdaptativeLoss
from attention import AttentionLayer
keras.backend.clear_session()


layers = [35, 35] # Number of hidden neuros in each layer of the encoder and decoder

learning_rate = 5e-2
decay = 0 # Learning rate decay
optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay) # Other possible optimiser "sgd" (Stochastic Gradient Descent)

# The dimensionality of the input at each time step. In this case a 1D signal.

num_output_features = 1 # The dimensionality of the output at each time step. In this case a 1D signal.
# There is no reason for the input sequence to be of same dimension as the ouput sequence.
# For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.

def smooth_L1_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)

loss = 'mse' # smooth_L1_loss # Other loss functions are possible, see Keras documentation.

# Regularisation isn't really needed for this application
lambda_regulariser = 0.000001 # Will not be used if regulariser is None
regulariser = keras.regularizers.l2(lambda_regulariser)

batch_size = 16
steps_per_epoch = 8
epochs = 500

def define_nmt(input_shape, hidden_size):
    """ Defining a NMT model """

    # Define an input sequence and process it.
    encoder_inputs = keras.layers.Input(shape=input_shape)
    decoder_inputs = keras.layers.Input(shape=(None, 1))

    # Encoder GRU
    encoder_gru = Bidirectional(GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru'), name='bidirectional_encoder')
    encoder_out, encoder_fwd_state, encoder_back_state = encoder_gru(encoder_inputs)

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = GRU(hidden_size*2, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(
        decoder_inputs, initial_state=Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state])
    )

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer
    dense = Dense(input_shape[0], activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    return full_model

def create_model(input_shape, weights=None):
    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=input_shape)

    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    encoder = keras.layers.RNN(encoder_cells, return_state=True)

    encoder_outputs_and_states = encoder(encoder_inputs)

    # Discard encoder outputs and only keep the states.
    # The outputs are of no interest to us, the encoder's
    # job is to create a state describing the input sequence.
    encoder_states = encoder_outputs_and_states[1:]

    # The decoder input will be set to zero (see random_sine function of the utils module).
    # Do not worry about the input size being 1, I will explain that in the next cell.
    decoder_inputs = keras.layers.Input(shape=(None, 1))

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    # Set the initial state of the decoder to be the ouput state of the encoder.
    # This is the fundamental part of the encoder-decoder.
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

    # Only select the output of the decoder (not the states)
    decoder_outputs = decoder_outputs_and_states[0]

    # Attention layer
    #attn_layer = AttentionLayer(name='attention_layer')
    #attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Apply a dense layer with linear activation to set output to correct dimension
    # and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
    decoder_dense = keras.layers.Dense(num_output_features,
                                       activation='linear',
                                       kernel_regularizer=regulariser,
                                       bias_regularizer=regulariser)

    decoder_outputs = decoder_dense(decoder_outputs)

    # Create a model using the functional API provided by Keras.
    # The functional API is great, it gives an amazing amount of freedom in architecture of your NN.
    # A read worth your time: https://keras.io/getting-started/functional-api-guide/
    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=optimiser, loss=loss)
    if weights:
        model.load_weights(weights)
    init_op = tf.global_variables_initializer()
    K.get_session().run(init_op)
    return model


if __name__ == '__main__':
    signal_root_dir = 'C:\\Users\\betza\\OneDrive\\Desktop\\School\\Thesis\\Code\\files\\signals'
    from core.bll.signal_generator import SignalGenerator

    my_generator = SignalGenerator(signal_root_dir, batch_size=4)
    #for i, t in enumerate(my_generator.flow()):
    #    print(i)
    """
    train_data_generator = random_sine(batch_size=batch_size,
                                       steps_per_epoch=steps_per_epoch,
                                       input_sequence_length=input_sequence_length,
                                       target_sequence_length=target_sequence_length,
                                       min_frequency=0.1, max_frequency=10,
                                       min_amplitude=0.1, max_amplitude=1,
                                       min_offset=-0.5, max_offset=0.5,
                                       num_signals=num_signals, seed=1969)

    for i, t in enumerate(train_data_generator):
        print(i)

    input_shape = 1,  # I will need this to be 8, data_gen should return (batch_size, seq_len, num_channels)
    model = create_model(input_shape)
    model.fit_generator(train_data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)
    """