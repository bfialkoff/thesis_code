from random import sample
import pickle

import numpy as np

from core.bll.data_reader import DataReader
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor

class SignalGenerator:
    """
    todo should recieve path to csv file dump, batch size
     and should have a stream class that yields (batch_size, *signal.shape)
    """
    def __init__(self, signal_indices, batch_size, f_scaler_path, sampling_frequency=1980):
        self.sampling_frequency = sampling_frequency
        self.batch_size = batch_size
        self.preprocessor = Preprocessor()
        self.converter = Converter()
        self.signal_indices = signal_indices
        self.decoder_inputs = np.zeros((self.batch_size, 9909, 1))
        self.annotations = self.flow()
        self.f_scaler = self.load_scaler(f_scaler_path)

    def load_scaler(self, scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    def get_signals(self, signal_numbers):
        emg_signals = []
        force_signals = []
        for s in signal_numbers:
            signal_loader = DataReader(s)
            emg_ = signal_loader.get_emg_signal(channel=5)
            fsr_voltage_ = signal_loader.get_fsr_voaltage_signal()
            processed_emg = self.preprocessor.process_emg_signal(emg_, self.sampling_frequency)
            force = self.converter.fsr_voltage_to_force(fsr_voltage_)
            emg_signals.append(processed_emg)
            force_signals.append(force)
        return emg_signals, force_signals

    def get_signal_coeff_pairs(self, signal_numbers)
        emg_signals = []
        force_coeffs = []
        for s in signal_numbers:
            signal_loader = DataReader(s)
            emg_ = signal_loader.get_emg_signal(channel=5)
            processed_emg = self.preprocessor.process_emg_signal(emg_, self.sampling_frequency)
            emg_signals.append(processed_emg)
            force_coeffs.append(force)
        
        return emg_signals, force_signals

    def get_all_signals(self):
        emg_signals, force_signals = self.get_signals(self.signals)
        emg_signals = np.array(emg_signals)
        emg_signals = emg_signals.reshape((self.batch_size, 9909, -1))
        force_signals = np.array(force_signals).reshape((self.batch_size, -1, 1))
        return emg_signals, force_signals

    def flow(self):
        while True:
            signals = sample(self.signal_indices, self.batch_size)
            emg_signals, force_signals = self.get_signals(signals)
            emg_signals = np.array(emg_signals)
            emg_signals = emg_signals.reshape((self.batch_size, 9909, -1))
            force_signals = np.array(force_signals).reshape((self.batch_size, -1, 1))

            # emg_signals = self.e_scaler.fit_transform(emg_signals)
            # force_signals = self.f_scaler.fit_transform(force_signals)
            yield emg_signals, force_signals

    def _flow(self):
        while True:
            signals = sample(self.signal_indices, self.batch_size)
            emg_signals, force_signals = self.get_signals(signals)
            emg_signals = np.array(emg_signals)
            force_signals = np.array(force_signals).reshape(self.batch_size, -1, 1)

            # emg_signals = self.e_scaler.fit_transform(emg_signals)
            # force_signals = self.f_scaler.fit_transform(force_signals)
            yield [emg_signals, self.decoder_inputs], force_signals

    def val_flow(self):
        while True:
            signals = sample(self.signal_indices, self.batch_size)
            emg_signals, force_signals = self.get_signals(signals)
            emg_signals = np.array(emg_signals)
            yield [emg_signals, self.decoder_inputs]

