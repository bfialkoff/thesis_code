from pathlib import Path
from random import sample
import json

from keras import backend as K
from keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten, Input, TimeDistributed, MaxPooling1D
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from core.bll.signal_loader import SignalLoader

def save_data_split(train_indices, val_indices, test_indices, file_path):
    data_split = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices}
    with open(file_path, 'w') as f:
        json.dump(data_split, f)

def create_callbacks(experiment_dir):
    callback_list = []
    weights_file = experiment_dir.joinpath('weights', 'emg_to_force_coeffs_{epoch:02d}.hdf5').resolve()
    tensorboard_dir = experiment_dir.joinpath('tesnorboard').resolve()
    if not weights_file.parent.exists():
        weights_file.parent.mkdir(parents=True)
    if not tensorboard_dir.exists():
        tensorboard_dir.mkdir(parents=True)
    weights_file = str(weights_file)
    callback_list.append(ModelCheckpoint(weights_file))
    callback_list.append(TensorBoard(tensorboard_dir))
    return callback_list

def load_split_data(data_file):
    with open(data_file, 'r') as f:
        data_json = json.load(f)
    train_indices, val_indices, test_indices = data_json['train_indices'], data_json['val_indices'], data_json['test_indices']
    return train_indices, val_indices, test_indices


def split_data(num_signals, data_file, train_frac=0.7, test_frac=0.5):
    if data_file.exists():
        train_indices, val_indices, test_indices = load_split_data(data_file)
    else:
        signals = range(num_signals)
        train_indices = sample(signals, int(train_frac * num_signals))
        others = list(set(signals).difference(train_indices))
        val_indices = sample(others, int((1 - train_frac) * num_signals * test_frac))
        test_indices = list(set(others).difference(val_indices))
        save_data_split(train_indices, val_indices, test_indices, data_file)
    return train_indices, val_indices, test_indices

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def r_square_loss(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (1 - SS_res / (SS_tot + K.epsilon()))


def get_model():
    ## define model
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=64, kernel_size=5, activation='relu'), input_shape=(None, 9910, 1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(7))
    model.compile(
        optimizer='adam', loss='mse', metrics=[r_square])
    #return model_seq
    return model


if __name__ == '__main__':
    file_dir = Path(__file__, '..', 'files').resolve()
    experiment_dir = file_dir.joinpath('first_attempt').resolve()
    data_file = file_dir.joinpath('poly_6_coeff_data_split.json').resolve()
    force_coeff_annotation_path = file_dir.joinpath('poly_6_coeffs.csv').resolve()
    force_scaler_path = file_dir.joinpath('poly_6_coeffs_scaler.pkl').resolve()
    train_indices, val_indices, test_indices = split_data(132, data_file)
    #train_loader = SignalLoader(train_indices, force_coeff_annotation_path, force_scaler_path)
    #val_loader = SignalLoader(val_indices, force_coeff_annotation_path, force_scaler_path)
    test_loader= SignalLoader(test_indices, force_coeff_annotation_path, force_scaler_path)
    #x_train, y_train = train_loader.get_all_signals()
    #x_val, y_val = val_loader.get_all_signals()
    x_test, y_test = test_loader.get_all_signals()
    x_test = x_test.reshape((x_test.shape[0], 1, 9910, 1))
    model = get_model()
    callbacks = create_callbacks(experiment_dir)
    model.fit(x_test, y_test, epochs=500, callbacks=callbacks)
