import json
from random import sample
from pathlib import Path

from keras.callbacks import ModelCheckpoint

from core.utils.plot import eval_seq2seq_prediction
from core.model.model import create_model
from core.bll.signal_generator import SignalGenerator


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
    batch_size = 16
    epochs = 100
    file_dir = Path(__file__, '..', 'files').resolve()
    e_scaler_path = file_dir.joinpath('e_scaler.pkl').resolve()
    f_scaler_path = file_dir.joinpath('f_scaler.pkl').resolve()
    signal_root_dir = file_dir.joinpath('signals').resolve()
    weights_dir = file_dir.joinpath('weights', 'no_attention').resolve()
    weights_file = weights_dir.joinpath('emg_to_force_{epoch:02d}.hdf5').resolve()
    data_file = weights_dir.joinpath('data_split.json').resolve()
    weights_file = str(weights_file)
    _, val_indices = load_split_data(data_file)
    val_steps = len(val_indices) // batch_size
    val_generator = SignalGenerator(val_indices, batch_size, e_scaler_path, f_scaler_path)

    model = create_model((9909, 8), weights_file.format(epoch=60))
    gt_ = []
    for i, d in enumerate(val_generator.flow()):
        if i == val_steps:
            break
        _, gt = d
        gt_.append(gt)
    p = model.predict_generator(val_generator.val_flow(), steps=val_steps)
    eval_seq2seq_prediction(gt[4], p[4])
    print('done')
