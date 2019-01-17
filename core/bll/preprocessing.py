import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, detrend, iirnotch
from core.utils.linear_regression import linear_regression

class Preprocessor:
    # usage: y = butter_lowpass_filter(data, cutoff, fs, order)
    def _butter_lowpass_filter(self, noisy_signal, sampling_freq, cutoff, order=8):
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

    def _notch_filter(self, signal, sampling_freq, cutoff=50,):
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


    def process_emg_signal(self, emg, sampling_freq):
        detrended_emg = detrend(emg)
        lpf_emg = self._butter_lowpass_filter(detrended_emg, sampling_freq=sampling_freq, cutoff=450)
        notched_emg = self._notch_filter(lpf_emg, sampling_freq)
        return notched_emg

    def process_force_signal(self, force, sampling_freq):
        lpf_force = self._butter_lowpass_filter(force, cutoff=10, sampling_freq=sampling_freq)
        return lpf_force


    def _get_line_coeffs(self):
        path_to_line_coefficients = Path(__file__).joinpath('..', '..', '..', 'files', 'line_coeffs.pkl').resolve()
        if not path_to_line_coefficients.exists():
            force = np.array([50, 100, 262.5, 500, 1000, 2000, 4000, 7000, 10000])
            resistance = np.array([10, 6.5, 3.33, 2.182, 1.227, 0.77, 0.471, 0.335, 0.25])
            a, b = linear_regression(np.log10(resistance), np.log10(force))
            with open(path_to_line_coefficients, 'wb') as f:
                pickle.dump((a, b), f)
        else:
            with open(path_to_line_coefficients, 'rb') as f:
                a, b = pickle.load(f)
        return a, b

    def convert_fsr_voltage_to_force(self, fsr_voltage):
        a, b = self._get_line_coeffs()
        log_force = a * np.log10(fsr_voltage) + b
        return np.power(10, log_force)
