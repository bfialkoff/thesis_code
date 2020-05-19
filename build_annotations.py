from pathlib import Path
from random import sample

import pandas as pd

def train_val_split(index_list, train_frac=0.7):
    num_samples = len(index_list)
    train_samples = int(train_frac * num_samples)
    train_indices = sample(index_list, train_samples)
    val_indices = list(set(index_list).difference(train_indices))
    return train_indices, val_indices


if __name__ == '__main__':
    csv_path = Path(__file__, '..', 'files', 'annotations.csv')
    train_path = Path(__file__, '..', 'files', 'train_annotations.csv')
    val_path = Path(__file__, '..', 'files', 'val_annotations.csv')
    data_set = pd.read_csv(csv_path)
    subject_values = data_set['subject'].unique().tolist()
    train_indices, val_indices = train_val_split(subject_values)

    train_df = data_set[data_set['subject'].isin(train_indices)]
    val_df = data_set[data_set['subject'].isin(val_indices)]
    print()
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)