from pathlib import Path

from tqdm import tqdm
import pandas as pd

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.ordered_data_reader import OrderedDataReader
from core.utils.plot import *



if __name__ == '__main__':
    csv_path = Path(__file__, '..', 'files', 'annotations.csv')
    all_df = pd.DataFrame()
    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    subject = 3
    signal = 2
    num_subjects_gen = range(2, 8)
    num_signals_gen = range(1, 21)
    num_channels_gen = range(8)

    sampling_frequency = 1980 // 2

    for subject in tqdm(num_subjects_gen):
        data_reader = OrderedDataReader(subject)
        row = {}
        for signal in num_signals_gen:
            fsr_voltage = data_reader.get_fsr_voaltage_signal(signal)
            force = converter.fsr_voltage_to_force(fsr_voltage)
            force = preprocessor.process_force_signal(force, sampling_frequency)
            row.update({'force': force, 'signal_num': len(force) * [signal]}) # this doesnt count the signal number properly

            for channel in num_channels_gen:
                raw_emg = data_reader.get_emg_signal(signal, channel)
                emg = raw_emg.copy()
                processed_emg = preprocessor.process_emg_signal(raw_emg, sampling_frequency)
                row.update({f'channel_{channel}': processed_emg})
            row.update({'subject': len(row['force']) * [subject]})
            curr_df = pd.DataFrame(row)
            all_df = pd.concat([all_df, curr_df])
    all_df.to_csv(csv_path, index=False)