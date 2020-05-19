import numpy as np

from core.utils.smallestenclosingcircle import make_circle


class Features:
    # TODO add various feature getting functions form literature
    MAX_FORCE = 10
    feature_length = 15
    @classmethod
    def _get_sliding_window_signal(cls, arr, func):
        # fixme adjust this to allow for stride
        window_size = 9000
        index_points = range(len(arr) - window_size)
        signal = np.array([func(arr[i:i + window_size]) for i in index_points])
        return signal

    @classmethod
    def get_shifted_fft_and_frequency(cls, sampling_frequency, signal):
        dft = np.abs(np.fft.fft(signal))
        len_dft = len(dft)
        discrete_frequency = np.arange(0, len_dft)
        discrete_frequency[round(len_dft / 2):] = discrete_frequency[round(len_dft / 2):] - len_dft
        time_frequency = (sampling_frequency / len_dft) * discrete_frequency
        dft = dft / dft.max()
        return time_frequency, dft

    def get_cole_cole_params(self, emg, sampling_frequency=1980):
        """
        input processed_emg signal
        1) get dft
        2) get positive frequencies
        3) get real_part, imag_part of dft
        4) get circle params
        5) get cole_cole params
        """
        def get_cole_cole(emg):
            emg_dft = np.fft.fft(emg)
            emg_dft = emg_dft[:len(emg_dft) // 2]
            r_dft, i_dft = emg_dft.real, emg_dft.imag
            points = zip(r_dft, i_dft)
            xc, yc, r = make_circle(points)
            discr = np.sqrt(r ** 2 - yc ** 2) **0.5
            x1 = xc + discr
            x2 = xc - discr

            eps_inf = min(x1, x2)
            delta = abs(x1 - x2)
            tau = None
            cc_params = eps_inf, delta, tau
            return cc_params

        emg_split_list = np.array_split(emg, self.feature_length)
        cole_cole_signal = np.array([get_cole_cole(emg) for emg in emg_split_list])
        return cole_cole_signal

    def get_vpp_signal(self, emg):
        get_vpp = lambda x: np.max(x) - np.min(x)
        #vpp_signal = self._get_sliding_window_signal(emg, get_vpp)
        emg_split_list = np.array_split(emg, self.feature_length)
        vpp_signal = np.array([get_vpp(emg) for emg in emg_split_list])
        return vpp_signal

    @classmethod
    def get_average_force_signal(cls, force):
        #averaged_force_signal = cls._get_sliding_window_signal(force, np.mean)
        force_split_list = np.array_split(force, cls.feature_length)
        averaged_force_signal = np.array([np.mean(force) for force in force_split_list])
        return averaged_force_signal



    def linear_regression(self, x, y):
        a, b = np.polyfit(x, y, 1)
        r_squared = np.corrcoef(x, y)[0, 1]
        return a, b, r_squared

if __name__ == '__main__':
    from core.bll.preprocessing import Preprocessor
    from core.bll.ordered_data_reader import OrderedDataReader

    preprocessor = Preprocessor()
    data_reader = OrderedDataReader(1)
    fs = 1980//2
    emg = data_reader.get_emg_signal(1, 1)
    emg = -preprocessor.process_emg_signal(emg, sampling_freq=fs)
    a = Features().get_cole_cole_params(emg, fs)
