from pathlib import Path

import numpy as np

class DataReader:
    path_to_signals = Path(__file__).joinpath('..', '..', '..', 'files', 'signals').resolve()
    signal_dir = 'signals'
    signal_prefix = 'signal'
    force_file_name = 'force.csv'
    emg_file_name = 'emg.csv'

    def __init__(self, signal_number):
        self._signal = self.path_to_signals.joinpath(f'{self.signal_prefix}{signal_number}').resolve()

    def get_fsr_voaltage_signal(self):
        signal_path = self._signal.joinpath(self.force_file_name).resolve()
        fsr_voltage = np.genfromtxt(signal_path, delimiter=',')
        return fsr_voltage

    def get_emg_signal(self, channel=slice(0, 8)):
        """ return the emg signal from the channel given, supports array indexing"""
        signal_path = self._signal.joinpath(self.emg_file_name).resolve()
        emg = np.genfromtxt(signal_path, delimiter=',')[:, channel]
        return emg
