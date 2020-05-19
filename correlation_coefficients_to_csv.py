from pathlib import Path

from tqdm import tqdm
import numpy as np

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.data_reader import DataReader

# todo output a csv contatining the r2 for all channels of all signals

if __name__ == '__main__':
    features = Features()
    preprocessor = Preprocessor()

    converter = Converter()
    num_signals = 132
    low_freq, hi_freq = 120, 180
    sampling_frequency = 1980 // 2
    windowed_results_path = Path(__file__).joinpath('..', 'files', f'windowed_correlation_coefficients.csv').resolve()
    baseline_results_path = Path(__file__).joinpath('..', 'files', f'baseline_correlation_coefficients.csv').resolve()
    windowed_results_csv = open(windowed_results_path, 'w')
    baseline_results_csv = open(baseline_results_path, 'w')
    header = 'signal_num,channel_1,channel_2,channel_3,channel_4,channel_5,channel_6,channel_7,channel_8\n'
    windowed_results_csv.write(header)
    baseline_results_csv.write(header)
    for signal in tqdm(range(num_signals)):
        data_reader = DataReader(signal)
        windowed_row = baseline_row = f'{signal}'
        fsr_voltage = data_reader.get_fsr_voaltage_signal()
        force = converter.fsr_voltage_to_force(fsr_voltage)
        force = preprocessor.process_force_signal(force, sampling_frequency)
        average_force = features.get_average_force_signal(force)
        for emg_channel in range(8):
            emg = data_reader.get_emg_signal(emg_channel)
            processed_emg = preprocessor.process_emg_signal(emg, sampling_frequency)
            vpp = features.get_vpp_signal(processed_emg)
            baseline_a, baseline_b, baseline_r_squared = features.linear_regression(np.log(vpp), average_force)
            windowed_emg = preprocessor.bandpass_filter(processed_emg, low_freq, hi_freq, sampling_frequency)
            time_frequency, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, windowed_emg)
            _, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, fsr_voltage)
            vpp = features.get_vpp_signal(windowed_emg)
            a, b, r_squared = features.linear_regression(np.log(vpp), average_force)
            windowed_row += f',{r_squared}'
            baseline_row += f',{baseline_r_squared}'
        windowed_row += '\n'
        baseline_row += '\n'
        windowed_results_csv.write(windowed_row)
        baseline_results_csv.write(baseline_row)

    windowed_results_csv.close()
    baseline_results_csv.close()