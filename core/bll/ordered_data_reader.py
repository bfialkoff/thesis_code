from pathlib import Path

import numpy as np

class OrderedDataReader:
    path_to_signals = Path(__file__).joinpath('..', '..', '..', 'files', 'ordered_signals').resolve()
    signal_dir = 'ordered_signals'
    signal_prefix = 'subject'
    force_file_name = 'force_{}.csv'
    emg_file_name = 'emg_{}.csv'

    def __init__(self, subject_number):
        self._signal = self.path_to_signals.joinpath(f'{self.signal_prefix}{subject_number}').resolve()

    def get_fsr_voaltage_signal(self, signal_number):
        signal_path = self._signal.joinpath(self.force_file_name.format(signal_number)).resolve()
        fsr_voltage = np.genfromtxt(signal_path, delimiter=',')
        return fsr_voltage

    def get_emg_signal(self, signal_number, channel=slice(0, 8)):
        """ return the emg signal from the channel given, supports array indexing"""
        signal_path = self._signal.joinpath(self.emg_file_name.format(signal_number)).resolve()
        emg = np.genfromtxt(signal_path, delimiter=',')[:, channel]
        return emg
