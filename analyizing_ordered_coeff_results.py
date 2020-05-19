"""
this script produces the distribution of optimal windows for all the signals for all the subjects
"""
from pathlib import Path

import pandas as pd
import  matplotlib.pyplot as plt

if __name__ == '__main__':
    bandwidth = 20
    stride = bandwidth // 2
    max_freq = 350
    n_best = 5
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

    g = df.groupby('signal')
    colors = ['r', 'g', 'b', 'k', 'c', 'y', 'm']
    for signal_num, g_ in g:
        f, ax = plt.subplots(1, 1)
        window_cols = []
        for col in g_.columns:
            if 'window' in col:
                window_cols.append(col)
                g_[col] = g_[col].map(tick_values)

        for c, col in zip(colors, window_cols):
            y_axis = g_[col].values.reshape(-1)
            ax.hist(y_axis, bins=len(tick_labels))
        labels_to_tick = range(0, len(tick_labels), 2)
        ax.set_xticks(labels_to_tick)
        ax.set_xticklabels([t for i, t in enumerate(tick_labels) if i in list(labels_to_tick)])
        ax.xaxis.set_tick_params(rotation=45)
        graph_path = Path(__file__).joinpath('..', 'files', 'window_analysis', f'{signal_num}_scatter_optimal_window.png').resolve()
        plt.savefig(graph_path)
        plt.close()
