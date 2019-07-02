import os
from pathlib import Path
import re

from tqdm import tqdm
import numpy as np

from core.bll.data_reader import DataReader

def get_latest_signal(signal_out_path):
    signals = [p for p in os.listdir(signal_out_path) if 'signal' in p]
    latest_signal = max(re.split('(\d+)', s)[1] for s in signals)
    latest_signal = int(latest_signal)
    return latest_signal

signal_dir = DataReader.signal_dir

path_to_in_signals = Path(__file__).joinpath('..', '..', 'Harel', signal_dir).resolve()
path_to_out_signals = Path(__file__).joinpath('..', 'files', signal_dir).resolve()
latest_signal = get_latest_signal(path_to_out_signals)
i = 1
for root, dirs, files in tqdm(os.walk(path_to_in_signals)):
    out_path = path_to_out_signals.joinpath(f'{signal_dir}{latest_signal + i}')
    if not Path.exists(out_path):
        Path.mkdir(out_path)
        i += 1
    else:
        continue

    for name in files:
        data = np.genfromtxt(os.path.join(root, name), delimiter=',')
        if 'EMG' in name:
            data = data[:, :7]
            np.savetxt(out_path.joinpath('emg.csv'), data)
        else:
            data = data[:, 0]
            np.savetxt(out_path.joinpath('force.csv'), data)
