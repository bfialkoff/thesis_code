import numpy as np
import matplotlib.pyplot as plt

from core.bll.data_reader import DataReader
from core.utils.array import permute_axes_subtract

if __name__ == '__main__':
    data_reader = DataReader(1)
    emg = data_reader.get_emg_signal(np.arange(7))
    p_diffs = permute_axes_subtract(emg)
    fftd = np.fft.fft2(p_diffs)
    for mat in p_diffs:
        mat = np.abs(mat)
        im = plt.imshow(mat)
        im.set_data(mat)
        plt.pause(0.002)
    pass