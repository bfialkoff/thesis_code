from pathlib import Path

import numpy as np
from scipy.signal import detrend

from core.bll.features import Features
from core.bll.preprocessing import Preprocessor
from core.utils.plot import *

path_to_signals = Path(__file__).joinpath('..', 'signals').resolve()
my_signal = path_to_signals.joinpath('signal1').resolve()

# TODO, experiment with emg rectification
if __name__ == '__main__':
    emg = np.genfromtxt(my_signal.joinpath('emg.csv').resolve(), delimiter=',')[:, 0]
    force = np.genfromtxt(my_signal.joinpath('force.csv').resolve(), delimiter=',')
    sampling_frequency = 1980
    features = Features()
    preprocessor = Preprocessor()

    processed_emg = preprocessor.process_emg_signal(emg, sampling_frequency)
    processed_force = preprocessor.process_force_signal(force, sampling_frequency)

    time_frequency, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, detrend(emg))
    _, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, force)
    line(emg_dft, time_frequency, title='lpf_emg frequency spectrum', xlim=[-600, 600])
    line(force_dft, time_frequency, title='force signal frequency spectrum', xlim=[-20, 20])

    vpp = features.get_vpp_signal(processed_emg)
    average_force = features.get_average_force_signal(processed_force)
    scatter(average_force, np.log(vpp), 'scatter log(Vpp) vs averaged force', 'mean force', 'log(Vpp)')

    """
    sampling_duration = 20
    sampling_time_period = 0.01
    sampling_frequency = 1/sampling_time_period
    n = np.arange(0, sampling_duration, sampling_time_period)
    test_signal = np.sin(two_pi * 20 * n) + np.sin(two_pi * 10 * n)
    line(test_signal)
    time_frequency, dft = features.get_shifted_fft_and_frequency(sampling_frequency, test_signal)
    line(dft, time_frequency)
    """
