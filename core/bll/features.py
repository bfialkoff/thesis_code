import numpy as np


class Features:
    # TODO add various feature getting functions form literature

    def _get_sliding_window_signal(self, arr, func):
        window_size = 50
        index_points = range(len(arr) - window_size)
        signal = np.array([func(arr[i:i + window_size]) for i in index_points])
        return signal

    def get_vpp_signal(self, emg):
        get_vpp = lambda x: np.max(x) - np.min(x)
        vpp_signal = self._get_sliding_window_signal(emg, get_vpp)
        return vpp_signal

    def get_average_force_signal(self, force):
        averaged_force_signal = self._get_sliding_window_signal(force, np.mean)
        return averaged_force_signal