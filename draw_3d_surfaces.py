from pathlib import Path

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

graph_dir = Path(__file__).joinpath('..', 'files', '3d_surfaces').resolve()


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
    #plt.savefig(graph_path)
    plt.show()
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
    windowed_results_path = Path(__file__).joinpath('..', 'files',
                                                    'windowed_regression_full_data.csv').resolve()
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
