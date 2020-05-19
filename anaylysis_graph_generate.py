"""
this script generates graphs that shows how location of the OW changes the i^th signal of the j^th subject, for the top 5
OWs for each 8 channels. Accordingly this script generates 20 graphs per channel. Each grap shows the top 5 OWs of each
subject. Each subject is color coded, and the OW locations on numbered, 1 indicates its the best OW, 2 indicates its the 2nd best
OW, etc.
"""
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import  matplotlib.pyplot as plt

if __name__ == '__main__':
    colors = ['r', 'g', 'b', 'k', 'c', 'y', 'm']
    colors = colors[:5]
    bandwidth = 20
    stride = bandwidth // 2
    max_freq = 350
    n_best = 5
    num_signals = 20
    num_windows = (max_freq - bandwidth) // stride + 1  # +1 is because of the offset
    frequency_windows = [(i * stride, i * stride + bandwidth) for i in range(1, num_windows)]
    tick_labels = ['{}-{}'.format(w1, w2) for (w1, w2) in frequency_windows]
    tick_values = {t: i for i, t in enumerate(tick_labels)}

    windowed_results_path = Path(__file__).joinpath('..', 'files',
                                                    f'{n_best}_best_ordered_windowed_correlation_coefficients.csv').resolve()
    df = pd.read_csv(windowed_results_path)
    for col in df.columns:
        if 'window' in col:
            df[col] = df[col].str.replace('Oct-30', '10-30')

    signal_groups = df.groupby('signal')
    graph_root_dir = Path(__file__).joinpath('..', 'files', f'all_subjects{n_best}_best_windows').resolve()
    for c in tqdm(range(8)):
        graph_dir = graph_root_dir.joinpath(f'channel_{c}').resolve()
        if not graph_dir.exists():
            graph_dir.mkdir(parents=True)
        columnn_format = ''
        for n in range(n_best):
            columnn_format += f'{n + 1}_best_window_channel_{c + 1},'
        columns = columnn_format.split(',')[:-1]
        for signal_num, signal_df in signal_groups:
            graph_path = graph_dir.joinpath(f'signal_{signal_num}.png').resolve()
            f, ax = plt.subplots(1, 1)
            subject_groups = signal_df.groupby('subject')
            for color, (subject, subject_df) in zip(colors, subject_groups):
                for i, col in enumerate(columns):
                    tick_inds = subject_df[col].map(tick_values)
                    ax.plot([i + 1], tick_inds, f'o{color}', marker=f'${i + 1}$')
            labels_to_tick = range(0, len(tick_values), 1)
            ax.set_yticklabels([t for i, t in enumerate(tick_labels) if i in list(labels_to_tick)])
            ax.yaxis.set_tick_params()
            plt.savefig(graph_path)
            plt.close(f)
            #from sys import exit
            #exit()
