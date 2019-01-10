from pathlib import Path
import numpy as np

from core.bll.features import Features
from core.utils.plot import *

path_to_signals = Path(__file__).joinpath('..', 'signals').resolve()
my_signal = path_to_signals.joinpath('signal1').resolve()

if __name__ == '__main__':
    emg = np.genfromtxt(my_signal.joinpath('emg.csv').resolve(), delimiter=',')[:, 0]
    force = np.genfromtxt(my_signal.joinpath('force.csv').resolve(), delimiter=',')
    sampling_frequency = 1900
    features = Features()
    vpp = features.get_vpp_signal(emg)
    average_force = features.get_average_force_signal(force)

    time_frequency, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, vpp)
    _, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, average_force)
    line(emg_dft, time_frequency)
    line(force_dft, time_frequency)

    # scatter(np.log(vpp), average_force, 'scatter log(Vpp) vs averaged force', 'log(Vpp)', 'mean force')

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
