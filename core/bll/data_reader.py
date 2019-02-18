from pathlib import Path

import numpy as np

class DataReader:
    path_to_signals = Path(__file__).joinpath('..', '..', '..', 'files', 'signals').resolve()
    my_signal = path_to_signals.joinpath('signal1').resolve()

    def __init__(self, signal_number):
        self._signal = self.path_to_signals.joinpath(f'signal{signal_number}').resolve()

    def get_fsr_voaltage_signal(self):
        fsr_voltage = np.genfromtxt(self._signal.joinpath('force.csv').resolve(), delimiter=',')
        return fsr_voltage

    def get_emg_signal(self, channel):
        emg = np.genfromtxt(self._signal.joinpath('emg.csv').resolve(), delimiter=',')[:, channel]
        return emg
