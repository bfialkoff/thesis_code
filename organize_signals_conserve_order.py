from pathlib import Path

from tqdm import tqdm
import numpy as np

from core.bll.data_reader import DataReader




signal_dir = 'ordered_' + DataReader.signal_dir
signal_file_name = DataReader.signal_prefix

path_to_in_signals = Path(__file__).joinpath('..', 'files', 'signals_from_harel').resolve()
path_to_out_signals = Path(__file__).joinpath('..', 'files', signal_dir).resolve()

for nivdak in tqdm(range(1, 8)):
    for signal_num in range(1, 21):
        signals_path = path_to_in_signals.joinpath(f'nivdak{nivdak}', str(signal_num)).resolve()
        out_path = path_to_out_signals.joinpath(f'subject{nivdak}').resolve()
        if not out_path.exists():
            out_path.mkdir(parents=True)
        emg_path = signals_path.joinpath('AD1EMG.txt')
        force_path = signals_path.joinpath('AD2FORCE.txt')
        emg_data = np.genfromtxt(emg_path, delimiter=',')
        force_data = np.genfromtxt(force_path, delimiter=',')
        emg_data = emg_data[:, :-1]
        force_data = force_data[:, 0]
        emg_outpath = out_path.joinpath(f'emg_{signal_num}.csv').resolve()
        force_outpath = out_path.joinpath(f'force_{signal_num}.csv').resolve()
        np.savetxt(force_outpath, force_data, delimiter=',')
        np.savetxt(emg_outpath, emg_data, delimiter=',')

