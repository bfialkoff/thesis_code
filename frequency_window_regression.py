from pathlib import Path

from scipy.signal import lfilter
from tqdm import tqdm

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.data_reader import DataReader
from core.utils.plot import *

if __name__ == '__main__':
    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    bandwidth = 20
    stride = bandwidth // 2
    max_freq = 350
    num_windows = (max_freq - bandwidth) // stride + 1 # +1 is because of the offset
    frequency_windows = [(i * stride, i * stride + bandwidth) for i in range(1, num_windows)]
    tick_labels = ['{}-{}'.format(w1, w2) for (w1, w2) in frequency_windows]
    sampling_frequency = 1980 // 2
    header = 'signal_num,' \
             'base_line_channel_1,baseline_rmse_channel_1,best_channel_1,best_rmse_channel_1,window_channel_1,' \
             'base_line_channel_2,baseline_rmse_channel_2,best_channel_2,best_rmse_channel_2,window_channel_2,' \
             'base_line_channel_3,baseline_rmse_channel_3,best_channel_3,best_rmse_channel_3,window_channel_3,' \
             'base_line_channel_4,baseline_rmse_channel_4,best_channel_4,best_rmse_channel_4,window_channel_4,' \
             'base_line_channel_5,baseline_rmse_channel_5,best_channel_5,best_rmse_channel_5,window_channel_5,' \
             'base_line_channel_6,baseline_rmse_channel_6,best_channel_6,best_rmse_channel_6,window_channel_6,' \
             'base_line_channel_7,baseline_rmse_channel_7,best_channel_7,best_rmse_channel_7,window_channel_7,' \
             'base_line_channel_8,baseline_rmse_channel_8,best_channel_8,best_rmse_channel_8,window_channel_8\n'
    num_signals = 132


    windowed_results_path = Path(__file__).joinpath('..', 'files', 'windowed_correlation_coefficients_17.csv').resolve()
    if not windowed_results_path.parents[0]:
        windowed_results_path.parents[0].mkdir(parents=True)
    windowed_results_csv = open(windowed_results_path, 'w')
    windowed_results_csv.write(header)

    for signal in tqdm(range(17, 18)):
        all_rs = []
        baseline_mean_r = 0
        row = f'{signal},'
        for emg_channel in range(8):
            path = Path(__file__).joinpath('..', 'files',
                                           f'channel_{emg_channel + 1}_windowed_frequency_regression_channel_{emg_channel + 1}_{bandwidth}Hz_stride_{stride}_sampling_freq_{sampling_frequency}').resolve()
            if not path.exists():
                path.mkdir(exist_ok=True, parents=True)
            save_path = path.joinpath(f'signal_{signal}.png')
            data_reader = DataReader(signal)
            fsr_voltage = data_reader.get_fsr_voaltage_signal()

            force = converter.fsr_voltage_to_force(fsr_voltage)
            force = preprocessor.process_force_signal(force, sampling_frequency)
            emg = data_reader.get_emg_signal(emg_channel)
            average_force = features.get_average_force_signal(force)
            processed_emg = preprocessor.process_emg_signal(emg, sampling_frequency)
            vpp = features.get_vpp_signal(processed_emg)
            baseline_a, baseline_b, baseline_r_squared = features.linear_regression(np.log(vpp), average_force)
            baseline_rmse = (((average_force - (baseline_a * np.log(vpp) + baseline_b)) ** 2).sum() / len(average_force)) ** 0.5
            baseline_mean_r += baseline_r_squared / num_signals
            rs = []
            rmses = []
            as_, bs_ = [], []
            row += f'{baseline_r_squared},{baseline_rmse},'
            for i, (low_freq, hi_freq) in enumerate(frequency_windows):
                windowed_emg = preprocessor.bandpass_filter(processed_emg, low_freq, hi_freq, sampling_frequency)
                time_frequency, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, windowed_emg)
                _, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, fsr_voltage)
                vpp = features.get_vpp_signal(windowed_emg)
                a, b, r_squared = features.linear_regression(np.log(vpp), average_force)
                rmse = (((average_force - (a * np.log(vpp) + b)) ** 2).sum() / len(average_force)) ** 0.5
                rs.append(r_squared)
                #regression_line_fourier_plot((a, b), np.log(vpp), average_force, time_frequency, emg_dft, r_squared)
                rmses.append(rmse)
            best_rmse = np.max(rs)
            arg_best = int(np.argmax(rs))
            row += f'{best_rmse},{rmses[arg_best]},{tick_labels[arg_best]},'

        row = row[:-1] + '\n'
        windowed_results_csv.write(row)
    windowed_results_csv.close()
    """
            all_rs.append(rs)
            
            f, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(rs)
            labels_to_tick = range(0, len(rs), 2)
            ax.set_xticks(labels_to_tick)
            ax.set_xticklabels([t for i, t in enumerate(tick_labels) if i in list(labels_to_tick)])
            ax.xaxis.set_tick_params(rotation=90)
            ax.set_title(f'baseline regression $R^2$={baseline_r_squared:.4f}\nbest windowed $R^2$={np.max(rs):.4f}, {tick_labels[int(np.argmax(rs))]}Hz')
            plt.subplots_adjust(bottom=0.15, top=0.86)
            #plt.show()
            plt.savefig(save_path)
            plt.close(f)
            
        all_rs = np.array(all_rs)
        mean_rs = all_rs.mean(axis=0)
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(mean_rs)
        labels_to_tick = range(0, len(tick_labels), 2)
        ax.set_xticks(labels_to_tick)
        ax.set_xticklabels([t for i, t in enumerate(tick_labels) if i in list(labels_to_tick)])
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_title(f'baseline regression $R^2$={baseline_mean_r:.4f}\nbest windowed $R^2$={np.max(mean_rs):.4f}, {tick_labels[int(np.argmax(mean_rs))]}Hz')
        plt.subplots_adjust(bottom=0.15, top=0.86)
        plt.show()
        plt.savefig(path.joinpath(f'average_regression_channel_{emg_channel}.png'))
    """
