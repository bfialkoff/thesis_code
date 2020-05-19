from pathlib import Path
from random import sample
import json
import pickle

import numpy as np
import pandas as pd

from core.bll.data_reader import DataReader
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor

class SignalLoader:
    """
    todo should recieve path to csv file dump, batch size
     and should have a stream class that yields (batch_size, *signal.shape)
    """
    def __init__(self, signal_indices, coeff_annotations_path, f_scaler_path, sampling_frequency=1980):
        self.sampling_frequency = sampling_frequency
        self.preprocessor = Preprocessor()
        self.signal_indices = signal_indices
        self.annotations = pd.read_csv(coeff_annotations_path)
        self.scaler = self.load_scaler(f_scaler_path)

    def load_scaler(self, scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    def get_signal_coeff_pairs(self, signal_numbers):
        emg_signals = []
        force_coeffs = self.annotations[self.annotations['signal_number'].isin(signal_numbers)].drop(columns=['signal_number'], axis=1)
        force_coeffs = self.scaler.transform(force_coeffs.values).tolist()
        for s in signal_numbers:
            signal_loader = DataReader(s)
            emg_ = signal_loader.get_emg_signal(channel=5)
            processed_emg = self.preprocessor.process_emg_signal(emg_, self.sampling_frequency)
            emg_signals.append(processed_emg)
        return emg_signals, force_coeffs

    def get_all_signals(self):
        emg_signals, force_coeffs = self.get_signal_coeff_pairs(self.signal_indices)
        num_samples = len(self.signal_indices)
        emg_signals = np.array(emg_signals)
        force_coeffs = np.array(force_coeffs)
        emg_signals = emg_signals.reshape((num_samples, emg_signals.shape[1], -1))
        force_coeffs = force_coeffs.reshape((num_samples, -1, force_coeffs.shape[1]))
        return emg_signals, force_coeffs

def save_data_split(train_indices, val_indices, test_indices, file_path):
    data_split = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices}
    with open(file_path, 'w') as f:
        json.dump(data_split, f)


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


if __name__ == '__main__':
    file_dir = Path(__file__, '..', '..',  '..', 'files').resolve()
    data_file = file_dir.joinpath('poly_6_coeff_data_split.json').resolve()
    force_coeff_annotation_path = file_dir.joinpath('poly_6_coeffs.csv').resolve() 
    force_scaler_path = file_dir.joinpath('poly_6_coeffs_scaler.pkl').resolve()
    train_indices, val_indices, test_indices = split_data(132, data_file)
    train_loader = SignalLoader(train_indices, force_coeff_annotation_path, force_scaler_path)
    print(train_loader.get_all_signals())