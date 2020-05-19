"""
this will need to loop over all subjects, all signals, all channels and all windows
for each window save a row containing:
subject,signal,channel,window,rmse,r^2

"""
# todo this is just a opy of the other, update it to work with the ordered signals
from pathlib import Path

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.ordered_data_reader import OrderedDataReader

graph_dir = Path(__file__).joinpath('..', 'files', '3d_surfaces').resolve()
root_mean_square_error = lambda y1, y2: (((y1 - y2) ** 2).sum() / len(y2)) ** 0.5

def get_z_value(df, signal_num, channel_num, col):
    sdf = df.loc[(df['signal_num'] == signal_num) & (df['channel'] == channel_num)]
    z = sdf.iloc[sdf['r_squared'].argmax()][col]
    return z

def plot_opt_window_surface(df, subject_num, graph_dir, overwrite=False):
    graph_dir = graph_dir.joinpath('optimal_mid_window')
    array_path = graph_dir.joinpath('arrays', f'subject_num_{subject_num}')
    if not array_path.exists():
        array_path.mkdir(parents=True)
    x_values = array_path.joinpath('x_values.npy')
    y_values = array_path.joinpath('y_values.npy')
    z_values = array_path.joinpath('z_values.npy')
    graph_path = graph_dir.joinpath(f'subject_num_{subject_num}')
    if not x_values.exists() or overwrite:
        signal_number_axis = np.arange(1, 21)
        channel_number_axis = np.arange(8)
        mid_windows = []
        for s in signal_number_axis:
            for c in channel_number_axis:
                mw = get_z_value(df, s, c, 'mid_window')
                mid_windows.append(mw)
        signal_grid, channel_grid = np.meshgrid(signal_number_axis, channel_number_axis)
        mid_windows = np.array(mid_windows).reshape(signal_grid.shape)
        np.save(x_values, signal_grid)
        np.save(y_values, channel_grid)
        np.save(z_values, mid_windows)
    else:
        signal_grid = np.load(x_values)
        channel_grid = np.load(y_values)
        mid_windows = np.load(z_values)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('Signal Num')
    ax1.set_ylabel('Channel Num')
    ax1.set_zlabel('Middle of Optimal Window')
    ax1.plot_surface(signal_grid, channel_grid, mid_windows)
    plt.savefig(graph_path)
    plt.close(fig)

def plot_opt_r2_surface(df, subject_num, graph_dir, overwrite=False):
    graph_dir = graph_dir.joinpath('optimal_r2')
    array_path = graph_dir.joinpath('arrays', f'subject_num_{subject_num}')
    if not array_path.exists():
        array_path.mkdir(parents=True)
    x_values = array_path.joinpath('x_values.npy')
    y_values = array_path.joinpath('y_values.npy')
    z_values = array_path.joinpath('z_values.npy')
    graph_path = graph_dir.joinpath(f'subject_num_{subject_num}')
    if not x_values.exists() or overwrite:
        signal_number_axis = np.arange(1, 21)
        channel_number_axis = np.arange(8)
        mid_windows = []
        for s in signal_number_axis:
            for c in channel_number_axis:
                mw = get_z_value(df, s, c, 'r_squared')
                mid_windows.append(mw)
        signal_grid, channel_grid = np.meshgrid(signal_number_axis, channel_number_axis)
        mid_windows = np.array(mid_windows).reshape(signal_grid.shape)
        np.save(x_values, signal_grid)
        np.save(y_values, channel_grid)
        np.save(z_values, mid_windows)
    else:
        signal_grid = np.load(x_values)
        channel_grid = np.load(y_values)
        mid_windows = np.load(z_values)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('Signal Num')
    ax1.set_ylabel('Channel Num')
    ax1.set_zlabel(r'$R^2$')
    ax1.plot_surface(signal_grid, channel_grid, mid_windows)
    plt.savefig(graph_path)
    plt.close(fig)


if __name__ == '__main__':
    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    bandwidth = 20
    stride = bandwidth // 2
    max_freq = 350
    num_windows = (max_freq - bandwidth) // stride + 1  # +1 is because of the offset
    frequency_windows = [(i * stride, i * stride + bandwidth) for i in range(1, num_windows)]
    print(frequency_windows)
    tick_labels = ['{}-{}'.format(w1, w2) for (w1, w2) in frequency_windows]
    sampling_frequency = 1980 // 2
    header = 'subject,signal_num,channel,b_rmse,b_r_squared,frequency_window,rmse,r_squared\n'

    n_best = 5
    windowed_results_path = Path(__file__).joinpath('..', 'files',
                                                 'windowed_regression_full_data.csv').resolve()
    if not windowed_results_path.exists():
        if not windowed_results_path.parents[0]:
            windowed_results_path.parents[0].mkdir(parents=True)
        windowed_results_csv = open(windowed_results_path, 'w')
        windowed_results_csv.write(header)
        for subject in range(1, 8):
            for signal in tqdm(range(1, 21)):
                all_rs = []
                baseline_mean_r = 0
                for emg_channel in range(8):
                    data_reader = OrderedDataReader(subject)
                    fsr_voltage = data_reader.get_fsr_voaltage_signal(signal)

                    force = converter.fsr_voltage_to_force(fsr_voltage)
                    force = preprocessor.process_force_signal(force, sampling_frequency)
                    emg = data_reader.get_emg_signal(signal, emg_channel)
                    average_force = features.get_average_force_signal(force)
                    processed_emg = preprocessor.process_emg_signal(emg, sampling_frequency)
                    vpp = features.get_vpp_signal(processed_emg)
                    baseline_a, baseline_b, baseline_r_squared = features.linear_regression(np.log(vpp), average_force)
                    baseline_rmse = root_mean_square_error(average_force, (baseline_a * np.log(vpp) + baseline_b))
                    for i, (low_freq, hi_freq) in enumerate(frequency_windows):
                        windowed_emg = preprocessor.bandpass_filter(processed_emg, low_freq, hi_freq, sampling_frequency)
                        time_frequency, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, windowed_emg)
                        vpp = features.get_vpp_signal(windowed_emg)
                        a, b, r_squared = features.linear_regression(np.log(vpp), average_force)
                        rmse = root_mean_square_error(average_force, (a * np.log(vpp) + b))
                        row = f'{subject},{signal},{emg_channel},{baseline_rmse},{baseline_r_squared},{i},{rmse},{r_squared}\n'
                        windowed_results_csv.write(row)
        windowed_results_csv.close()

    df = pd.read_csv(windowed_results_path)
    mid_window = [20.0,
                  30.0,
                  40.0,
                  50.0,
                  60.0,
                  70.0,
                  80.0,
                  90.0,
                  100.0,
                  110.0,
                  120.0,
                  130.0,
                  140.0,
                  150.0,
                  160.0,
                  170.0,
                  180.0,
                  190.0,
                  200.0,
                  210.0,
                  220.0,
                  230.0,
                  240.0,
                  250.0,
                  260.0,
                  270.0,
                  280.0,
                  290.0,
                  300.0,
                  310.0,
                  320.0]
    mid_window = pd.DataFrame({'frequency_window': range(len(mid_window)), 'mid_window': mid_window})
    df = pd.merge(df, mid_window, on='frequency_window')
    for subject in range(1, 8):
        s_df = df[df['subject'] == subject]
        plot_opt_window_surface(s_df, subject, graph_dir)
        plot_opt_r2_surface(s_df, subject, graph_dir)

