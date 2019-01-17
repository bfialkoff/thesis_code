import numpy as np


class Features:
    # TODO add various feature getting functions form literature

    def _get_sliding_window_signal(self, arr, func):
        window_size = 9000
        index_points = range(len(arr) - window_size)
        signal = np.array([func(arr[i:i + window_size]) for i in index_points])
        return signal

    def get_vpp_signal(self, emg):
        get_vpp = lambda x: np.max(x) - np.min(x)
        #vpp_signal = self._get_sliding_window_signal(emg, get_vpp)
        emg_split_list = np.array_split(emg, 15)
        vpp_signal = np.array([get_vpp(emg) for emg in emg_split_list])
        return vpp_signal

    def get_average_force_signal(self, force):
        #averaged_force_signal = self._get_sliding_window_signal(force, np.mean)
        force_split_list = np.array_split(force, 15)
        averaged_force_signal = np.array([np.mean(force) for force in force_split_list])
        return averaged_force_signal

    def get_shifted_fft_and_frequency(self, sampling_frequency, signal):
        dft = np.abs(np.fft.fft(signal))
        len_dft = len(dft)
        discrete_frequency = np.arange(0, len_dft)
        discrete_frequency[round(len_dft / 2):] = discrete_frequency[round(len_dft / 2):] - len_dft
        time_frequency = (sampling_frequency / len_dft) * discrete_frequency
        return time_frequency, dft
