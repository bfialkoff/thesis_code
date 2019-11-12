import json
from random import sample
from pathlib import Path

from keras.callbacks import ModelCheckpoint
from nd_standard_scaler import NDStandardScaler
from core.model.model import create_model
from core.bll.signal_generator import SignalGenerator

from wave_net import WaveNet

def save_data_split(train_indices, val_indices, file_path):
    data_split = {'train_indices': train_indices, 'val_indices': val_indices}
    with open(file_path, 'w') as f:
        json.dump(data_split, f)


def load_split_data(data_file):
    with open(data_file, 'r') as f:
        data_json = json.load(f)
    train_indices, val_indices = data_json['train_indices'], data_json['val_indices']
    return train_indices, val_indices


def split_data(num_signals, data_file, train_frac=0.8):
    if data_file.exists():
        train_indices, val_indices = load_split_data(data_file)
    else:
        signals = range(num_signals)
        train_indices = sample(signals, int(train_frac * num_signals))
        val_indices = list(set(signals).difference(train_indices))
        save_data_split(train_indices, val_indices, data_file)
    return train_indices, val_indices


if __name__ == '__main__':
    batch_size = 32
    epochs = 100
    file_dir = Path(__file__, '..', 'files').resolve()
    e_scaler_path = file_dir.joinpath('e_scaler.pkl').resolve()
    f_scaler_path = file_dir.joinpath('f_scaler.pkl').resolve()
    signal_root_dir = file_dir.joinpath('signals').resolve()
    weights_dir = file_dir.joinpath('weights', 'wavenet').resolve()
    weights_file = weights_dir.joinpath('emg_to_force_{epoch:02d}.hdf5').resolve()
    data_file = weights_dir.joinpath('data_split.json').resolve()
    if not weights_dir.exists():
        weights_dir.mkdir()
    weights_file = str(weights_file)
    callback_list = [ModelCheckpoint(weights_file)]
    num_signals = len(list(signal_root_dir.glob('*/emg.csv')))
    train_indices, val_indices = split_data(num_signals, data_file, train_frac=0.7)
    train_steps = len(train_indices) // batch_size
    val_steps = len(val_indices) // batch_size
    train_generator = SignalGenerator(train_indices, batch_size, e_scaler_path, f_scaler_path)
    val_generator = SignalGenerator(val_indices, batch_size, e_scaler_path, f_scaler_path)

    #    for i, t in enumerate(train_generator.flow()):
    #        print(i)
    #model = create_model((9909, 8), weights_file.format(epoch=initial_epoch))
    initial_epoch = 0
    wvnet = WaveNet(9909, 1)
    model = wvnet.get_model()
    for i, (a, b) in enumerate(train_generator.flow()):
        a, b = a[0], b[0]
        if i == 1:
            break
    wvnet.predict_and_plot(a.reshape(1, -1, 1), b.reshape(1, -1, 1), 0)
    model.fit_generator(train_generator.flow(), steps_per_epoch=train_steps, callbacks=callback_list,
                        epochs=epochs, validation_data=val_generator.flow(), validation_steps=val_steps,
                        validation_freq=3, initial_epoch=initial_epoch)
