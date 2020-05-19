from pathlib import Path

import pywt
import numpy as np
from scipy.signal import butter, lfilter, detrend, iirnotch, medfilt

from core.utils.plot import *

class Preprocessor:
    # usage: y = butter_lowpass_filter(data, cutoff, fs, order)
    def _butterworth_filter(self, noisy_signal, sampling_freq, cutoff, order=8):
        '''
        this function recieves a signal as input and returns the signal after passing it through
        a butterworth low pass filter
        :param noisy_signal: an np array, the signal to be filtered
        :param cutoff: a float, the cutoff frequency in Hz
        :param sampling_freq: a float, the sampling frequency in Hz
        :param order: an int, the order of the butterworth filter
        :return: the noisy_signal after passing it through the lp butterworth filter
        '''
        nyq = 0.5 * sampling_freq
        normal_cutoff = cutoff / nyq
        filter_numerator, filter_denominator = butter(order, normal_cutoff)
        y = lfilter(filter_numerator, filter_denominator, noisy_signal)
        return y

    def _notch_filter(self, signal, sampling_freq, cutoff):
        '''
        this function recieves a signal as input and returns the signal after passing it through
        a butterworth low pass filter
        :param signal: an np array, the signal to be filtered
        :param cutoff: a float, the frequency to be cut out of the signal
        :param sampling_freq: a float, the sampling frequency in Hz
        :return: the signal after passing it through the notch filter
        '''
        nyq = 0.5 * sampling_freq
        normal_cutoff = cutoff / nyq
        filter_numerator, filter_denominator = iirnotch(normal_cutoff, 30)
        y = lfilter(filter_numerator, filter_denominator, signal)
        return y

    def process_force_signal(self, force, sampling_freq, median_kernel=501, rms_kernel=500):
        med_force = medfilt(force, median_kernel)
        rms_force = self.rms_filter(med_force, kernel_length=rms_kernel)
        return rms_force

    def wavelet_filter(self, raw_signal, max_level=6, wavelet_type='sym4'):
        coeffs = pywt.wavedec(raw_signal, wavelet_type, level=max_level)
        thresholds = len(coeffs) * [15]
        thresholds[:3] = [0.1, .2, 0.4]
        thresholded_coeffs = [pywt.threshold(c, t * max(c), mode='soft') for c, t in zip(coeffs, thresholds)]
        datarec = pywt.waverec(thresholded_coeffs, wavelet_type)
        return datarec



    def process_emg_signal(self, emg, sampling_freq, path=None, cutoff=400):
        detrended_emg = detrend(emg)
        lpf_emg = self._butterworth_filter(detrended_emg, sampling_freq=sampling_freq, cutoff=cutoff)
        notched_emg = self._notch_filter(lpf_emg, sampling_freq, 50)
        notched_emg = self._notch_filter(notched_emg, sampling_freq, 100)
        #wavelet_smoothed = self.wavelet_filter(notched_emg)[:-1]
        #notched_emg = self.bandpass_filter(notched_emg, low_freq=5, hi_freq=25, sampling_freq=sampling_freq)
        #return wavelet_smoothed # ruins the regression
        return notched_emg

    def bandpass_filter(self, signal, low_freq, hi_freq, sampling_freq, order=5):
        nyq = 0.5 * sampling_freq
        low_freq = low_freq / nyq
        hi_freq = hi_freq / nyq
        filter_numerator, filter_denominator = butter(order, [low_freq, hi_freq], btype='band')
        filtered_signal = lfilter(filter_numerator, filter_denominator, signal)
        return filtered_signal

        

    def rms_filter(self, raw_signal, kernel_length=100):
        filtered_signal = lfilter(np.ones(kernel_length) / kernel_length, 1, raw_signal)
        return filtered_signal

