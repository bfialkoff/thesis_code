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
        fsr_voltage = np.genfromtxt(self._signal.joinpath(self.force_file_name).resolve(), delimiter=',')
        return fsr_voltage

    def get_emg_signal(self, channel):
        """ return the emg signal from the channel given, supports array indexing"""
        emg = np.genfromtxt(self._signal.joinpath(self.emg_file_name).resolve(), delimiter=',')[:, channel]
        return emg
