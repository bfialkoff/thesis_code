import numpy as np
import matplotlib.pyplot as plt
import pywt

from core.bll.data_reader import DataReader
from core.bll.preprocessing import Preprocessor
from core.bll.features import Features
from core.bll.converter import Converter

def wavelet_filter(raw_signal, max_level=6, wavelet_type='sym4'):
    coeffs = pywt.wavedec(raw_signal, wavelet_type, level=max_level)
    thresholds = len(coeffs) * [15]
    thresholds[:3] = [0.1, .2, 0.4]
    thresholded_coeffs = [pywt.threshold(c, t*max(c), mode='soft') for c, t in zip(coeffs, thresholds)]
    datarec = pywt.waverec(thresholded_coeffs, wavelet_type)
    return datarec, thresholded_coeffs, coeffs
for signal in range(1,2): # [7, 10, 14, 17]:
    print(signal)
    data_reader = DataReader(signal)
    preprocessor = Preprocessor()
    converter = Converter()
    raw_signal = data_reader.get_emg_signal(5)
    fsr_signal = data_reader.get_fsr_voaltage_signal()
    raw_force = converter.fsr_voltage_to_force(fsr_signal)
    force = preprocessor.rms_filter(raw_force)
    sampling_frequency = 1980
    """
    y = a e^bx --> log(y) = bx + log(a) --> y = mx + b ; a = e^b, m = b
    """
    x_data = np.arange(len(force))
    a, b = np.polyfit(np.log(force), x_data, 1)
    b = np.log(b)
    y = a * np.exp(b * x_data)

    data = preprocessor.process_emg_signal(raw_signal, sampling_frequency)

    # Decompose into wavelet components, to the level selected:
    datarec, thresholded_coeffs, coeffs = wavelet_filter(data)
    """
    f, ax = plt.subplots(7, 1)
    for i in range(len(coeffs)):
        ax[i].plot(coeffs[i])
        ax[i].plot(thresholded_coeffs[i])
    """
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)

    time_frequency, emg_dft = Features.get_shifted_fft_and_frequency(sampling_frequency, data)
    _, emg_dft_wavelet = Features.get_shifted_fft_and_frequency(sampling_frequency, datarec)
    # todo plot the force here as well add the
    ax1.plot(data)
    ax1.set_title("Raw signal")
    ax2.plot(time_frequency, emg_dft)
    ax2.set_title("Raw signal - \nFourier Spectrum")
    ax2.set_ylim([0, 10])
    ax3.plot(raw_force)
    ax3.set_title('raw force')

    ax4.plot(datarec)
    ax4.set_title("Wavelet filtered")
    ax5.plot(time_frequency, emg_dft_wavelet[:-1])
    ax5.set_title("Wavelet filtered - \nFourier Spectrum")
    ax5.set_ylim([0, 10])
    ax6.plot(force, label='signal')
    # ax6.plot(x_data, y, 'r', label='regressed')
    ax6.set_title('rms force')
    plt.tight_layout()
    plt.show()
