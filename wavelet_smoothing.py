import matplotlib.pyplot as plt
import pywt

from core.bll.data_reader import DataReader
from core.bll.preprocessing import Preprocessor

def wavelet_filter(raw_signal, max_level=6, threshold=0.5, wavelet_type='sym4'):
    coeffs = pywt.wavedec(raw_signal, wavelet_type, level=max_level)
    thresholds = len(coeffs) * [threshold]
    thresholds[:2] = [0, 0]
    thresholds[2] = 0.4
    thresholded_coeffs = [pywt.threshold(c, t*max(c), mode='soft') for c, t in zip(coeffs, thresholds)]
    datarec = pywt.waverec(thresholded_coeffs, wavelet_type)
    return datarec, thresholded_coeffs, coeffs

data_reader = DataReader(7)
preprocessor = Preprocessor()
raw_signal = data_reader.get_emg_signal(5)
sampling_frequency = 1980

data = preprocessor.process_emg_signal(raw_signal, sampling_frequency)

# Decompose into wavelet components, to the level selected:
datarec, thresholded_coeffs, coeffs = wavelet_filter(data)
f, ax = plt.subplots(7, 1)
for i in range(len(coeffs)):
    ax[i].plot(coeffs[i])
    ax[i].plot(thresholded_coeffs[i])


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(data)
plt.xlabel('time [n]')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
plt.plot(datarec)
plt.xlabel('time (s)')
plt.title("De-noised signal using wavelet techniques")

plt.tight_layout()
plt.show()
