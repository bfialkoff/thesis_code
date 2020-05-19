# todo this is just a opy of the other, update it to work with the ordered signals
from pathlib import Path

from scipy.signal import lfilter
from tqdm import tqdm

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.ordered_data_reader import OrderedDataReader
from core.utils.plot import *

if __name__ == '__main__':
    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    channels = 8
    n_best = 5
    bandwidth = 20
    stride = bandwidth // 2
    max_freq = 350
    num_windows = (max_freq - bandwidth) // stride + 1  # +1 is because of the offset
    frequency_windows = [(i * stride, i * stride + bandwidth) for i in range(1, num_windows)]
    tick_labels = ['{}-{}'.format(w1, w2) for (w1, w2) in frequency_windows]
    sampling_frequency = 1980 // 2

    windowed_results_path = Path(__file__).joinpath('..', 'files',
                                                    f'{n_best}_best_ordered_windowed_correlation_coefficients.csv').resolve()
    if not windowed_results_path.parents[0]:
        windowed_results_path.parents[0].mkdir(parents=True)

    header = ''
    for c in range(channels):
        header += f',baseline_r2_channel_{c + 1},baseline_rmse_channel_{c + 1}'
        for n in range(n_best):
            header += f',{n + 1}_best_r2_channel_{c + 1},' \
                      f'{n + 1}_best_rmse_channel_{c + 1},' \
                      f'{n + 1}_best_window_channel_{c + 1}'
    header = 'subject,signal' + header + '\n'
    if not windowed_results_path:
        windowed_results_path.mkdir(parents=True)
    results_csv = open(windowed_results_path, 'w')
    results_csv.write(header)
    for subject in range(1, 8):
        for signal in tqdm(range(1, 21)):
            all_rs = []
            row = f'{subject},{signal}'
            for emg_channel in range(channels):
                data_reader = OrderedDataReader(subject)
                fsr_voltage = data_reader.get_fsr_voaltage_signal(signal)

                force = converter.fsr_voltage_to_force(fsr_voltage)
                force = preprocessor.process_force_signal(force, sampling_frequency)
                emg = data_reader.get_emg_signal(signal, emg_channel)
                average_force = features.get_average_force_signal(force)
                processed_emg = preprocessor.process_emg_signal(emg, sampling_frequency)
                vpp = features.get_vpp_signal(processed_emg)
                baseline_a, baseline_b, baseline_r_squared = features.linear_regression(np.log(vpp), average_force)
                baseline_rmse = (((average_force - (baseline_a * np.log(vpp) + baseline_b)) ** 2).sum() / len(
                    average_force)) ** 0.5
                rs = []
                rmses = []
                row += f',{baseline_r_squared:.3f},{baseline_rmse:.3f}'
                for i, (low_freq, hi_freq) in enumerate(frequency_windows):
                    windowed_emg = preprocessor.bandpass_filter(processed_emg, low_freq, hi_freq, sampling_frequency)
                    time_frequency, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, windowed_emg)
                    _, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, fsr_voltage)
                    vpp = features.get_vpp_signal(windowed_emg)
                    a, b, r_squared = features.linear_regression(np.log(vpp), average_force)
                    rmse = (((average_force - (a * np.log(vpp) + b)) ** 2).sum() / len(average_force)) ** 0.5
                    rs.append(r_squared)
                    # regression_line_fourier_plot((a, b), np.log(vpp), average_force, time_frequency, emg_dft, r_squared)
                    rmses.append(rmse)

                rs = np.array(rs)
                rmses = np.array(rmses)
                arg_n_best = rs.argsort()[::-1][:n_best]
                n_best_rs = rs[arg_n_best]
                n_best_rmse = rmses[arg_n_best]
                n_best_windows = np.array(tick_labels)[arg_n_best]
                for r2, rmse, window in zip(n_best_rs, n_best_rmse, n_best_windows):
                    row += f',{r2:.3f},{rmse:.3f},{window}'
            row = row + '\n'
            results_csv.write(row)
    results_csv.close()
