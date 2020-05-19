"""
some interesting things to explore:

for each subject and for each channel check how the optimal windows are distributed
1)that is load subject one, look at channel 1, does the optimal signal stay in the same place for all 20 signals
put another way, how does the best window for the i^th channel change over time

2) load subject 1, how does the best window change over each channel. Is there a relationship? Does it change over time?

3) is the behavior seen in (1), (2) consistent for each subject.

fixme, not ready:
 this script produces the distribution of optimal windows for all the signals for all the subjects
"""
from pathlib import Path

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
    g = df.groupby('subject')
    c = 0
    columnn_format = ''
    for n in range(n_best):
        columnn_format += f'{n + 1}_best_window_channel_{c + 1},'
    columns = columnn_format.split(',')[:-1]

    for subject, g_ in g:
        f, ax = plt.subplots(1, 1)
        window_cols = []
        for col, c in zip(columns, colors):
            g_[col] = g_[col].map(tick_values)
            ax.plot(range(20), g_[col], f'o{c}')
        graph_path = Path(__file__).joinpath('..', 'files', 'window_analysis', f'{subject}_scatter_optimal_window.png').resolve()
        #plt.savefig(graph_path)
        plt.show()
        break
        #plt.close()
