import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import fft
import pandas as pd

from core.bll.data_reader import DataReader
from core.bll.preprocessing import Preprocessor

from core.utils.plot import *

def plot_wavelet(time, signal, scales,
                 waveletname='mexh',
                 cmap=plt.cm.seismic,
                 title='Wavelet Transform (Power Spectrum) of signal',
                 ylabel='Period (years)',
                 xlabel='Time'):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)

    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots()
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap=cmap)

    ax.set_title(title,)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()


data_reader = DataReader(1)
preprocessor = Preprocessor()
raw_signal = data_reader.get_emg_signal(4)
sampling_frequency = 1980

processed_signal = preprocessor.process_emg_signal(raw_signal, sampling_frequency)

time = np.arange(len(processed_signal))
scales = sampling_frequency * np.linspace(1, 20, 20)

# todo the frequency spectrum isnt correct see preliminary_emg_analysis
plot_wavelet(time, processed_signal, scales)