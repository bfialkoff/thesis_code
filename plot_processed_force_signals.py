from pathlib import Path

import numpy as np
from scipy.signal import medfilt
import matplotlib
matplotlib.use('Agg')

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.data_reader import DataReader
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
    sampling_frequency = 1980
    path = Path(__file__).joinpath('..', 'files', 'processed_force_signals').resolve()
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)
    med_filter_kernel = 501
    rms_filter_kernel = 500
    med_filter_seconds = med_filter_kernel / sampling_frequency
    rms_filter_seconds = rms_filter_kernel / sampling_frequency
    for signal in [4]:#range(132):
        save_path = path.joinpath(f'signal_{signal}.png').resolve()
        data_reader = DataReader(signal)
        fsr_voltage = data_reader.get_fsr_voaltage_signal()
        

        raw_force = converter.fsr_voltage_to_force(fsr_voltage)
        #force = preprocessor.process_force_signal(raw_force, sampling_frequency)
        med_force = medfilt(raw_force, med_filter_kernel)
        rms_force = preprocessor.rms_filter(med_force, kernel_length=rms_filter_kernel)
        
        time_frequency, raw_force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, raw_force)
        _, med_force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, med_force)
        _, rms_force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, rms_force)

        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(raw_force, 'k', label='raw')
        ax1.plot(med_force, 'g', label=f'median filtered {med_filter_seconds:.2f}s')
        ax1.plot(rms_force, 'r', label=f'rms filtered {rms_filter_seconds:.2f}s')
        ax1.legend()
        
        ax2.set_xlim([-10, 10])
        ax2.plot(time_frequency, raw_force_dft, 'k-', label='raw fft')
        ax2.plot(time_frequency, med_force_dft, 'g^', label=f'median filtered {med_filter_seconds:.2f}s fft')
        ax2.plot(time_frequency, rms_force_dft, 'r*', label=f'rms filtered {rms_filter_seconds:.2f}s fft')
        ax2.legend()
        plt.savefig(save_path)
        plt.close(f)