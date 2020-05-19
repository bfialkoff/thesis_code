from pathlib import Path

import numpy as np

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.data_reader import DataReader
from core.bll.ordered_data_reader import OrderedDataReader
from core.utils.plot import *
from core.utils.array import permute_axes_subtract

path_to_signals = Path(__file__).joinpath('..', 'files', 'signals').resolve()
my_signal = path_to_signals.joinpath('signal1').resolve()

# TODO write a script that will generate a subplot for each channel containing plot_regression_line and
#  the frequency spectrum

if __name__ == '__main__':
    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    data_reader = OrderedDataReader(1)
    emg = data_reader.get_emg_signal(1, 1)
    fsr_voltage = data_reader.get_fsr_voaltage_signal(6)
    sampling_frequency = 1980


    processed_emg = preprocessor.process_emg_signal(emg, sampling_frequency)
    force = converter.fsr_voltage_to_force(fsr_voltage)

    time_frequency, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, processed_emg)
    _, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, fsr_voltage)

    line(processed_emg, title='processed_emg')
    line(emg_dft, time_frequency, title='lpf_emg frequency spectrum', xlim=[-600, 600])
    line(force_dft, time_frequency, title='force signal frequency spectrum', xlim=[-20, 20])

    vpp = features.get_vpp_signal(processed_emg)
    average_force = features.get_average_force_signal(force)

    scatter(average_force, np.log(vpp), 'scatter log(Vpp) vs averaged force', 'mean force', 'log(Vpp)')
    a, b, r_squared = features.linear_regression(np.log(vpp), average_force)
    print(r_squared)
    plot_regression_line((a, b), np.log(vpp), average_force)

