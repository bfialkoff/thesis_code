from pathlib import Path
from math import pi
import numpy as np

from core.bll.features import Features
from core.utils.plot import *

# todo init as git repo, make a bunch of plots for preliminary analysis fft, scatter power etc

path_to_signals = Path(__file__).joinpath('..', 'signals').resolve()
my_signal = path_to_signals.joinpath('signal1').resolve()


if __name__ == '__main__':
    emg = np.genfromtxt(my_signal.joinpath('emg.csv').resolve(),delimiter=',')[:,0]
    force = np.genfromtxt(my_signal.joinpath('force.csv').resolve(),delimiter=',')

    features = Features()
    vpp = features.get_vpp_signal(emg)
    average_force = features.get_average_force_signal(force)

    scatter(np.log(vpp), average_force, 'scatter log(Vpp) vs averaged force', 'log(Vpp)', 'mean force')

    n = np.array(range(1000))
    test_signal = np.sin(2* pi * n/6)
    line(test_signal)
    frequency_spectrum(test_signal) # todo incomplete




