import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning, SparseCoder

from core.bll.data_reader import DataReader

"""
todo 
1) experiment with different filters to get a more matlabby looking signal, look into IIR like Harel did
2) attempt to get a more visual image, try a log transform
3) read about muscle firings

"""

# see here http://www.seaandsailor.com/dictlearning.html, and use `jupyter notebook` in the conda prompt to launch
if __name__ == '__main__':
    data_reader1 = DataReader(1)
    emg = data_reader1.get_emg_signal(np.arange(7))
    dico = DictionaryLearning(n_components=1024)
    D = dico.fit(emg[:, 1]).components_

    coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=20,
                        transform_alpha=None, transform_algorithm="omp")
    result = coder.transform(emg[:, 2])

    plt.plot(range(len(emg), emg[:, 2]), 'r')

    pass
