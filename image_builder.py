from random import shuffle, sample

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

from core.utils.gauss_rank_scaler import GaussRankScaler
from core.utils.array import permute_axes_subtract

class EmgImageGenerator:
    def __init__(self, csv_path, batch_size):
        self.input_columns = [f'channel_{c}' for c in range(8)]
        self.output_column = 'force'
        annotation = pd.read_csv(csv_path)
        self.num_samples = len(annotation)
        self.annotation = self.scale_rows(annotation)
        self.batch_size = batch_size
        self.index_list = list(range(self.num_samples))
        shuffle(self.index_list)

    def scale_rows(self, df):
        self.gauss_scaler = GaussRankScaler()
        self.min_max_scaler = MinMaxScaler()
        df[self.output_column] = self.gauss_scaler.fit_transform(df[self.output_column].values.reshape(-1, 1)) # transforms the outputs to a normal distribution
        df[self.input_columns] = self.gauss_scaler.fit_transform(df[self.input_columns])
        return df

    def generator(self):
        # something like
        counter = 0
        while True:
            if counter == self.num_samples:
                shuffle(self.index_list)
                counter = 0
            batch_indices = sample(self.index_list, self.batch_size)
            input_rows = self.annotation.iloc[batch_indices]

            inputs, outputs = input_rows[self.input_columns].values, input_rows[self.output_column].values
            input_images = permute_axes_subtract(inputs)
            input_images = self.min_max_scaler.fit_transform(input_images.reshape(-1, self.batch_size))
            input_images = input_images.reshape(self.batch_size, 8, 8)
            input_images = np.repeat(input_images[:, :, :, np.newaxis], 3, axis=3)
            yield input_images, outputs

def show_2_images(img1, img2):
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()

if __name__ == '__main__':
    from pathlib import Path
    train_path = Path(__file__, '..', 'files', 'train_annotations.csv')
    val_path = Path(__file__, '..', 'files', 'val_annotations.csv')
    emg_gen = EmgImageGenerator(train_path, 16)

    for d in emg_gen.generator():
        print('apple')