from pathlib import Path

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.ordered_data_reader import OrderedDataReader
from core.utils.plot import *

if __name__ == '__main__':
    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    data_reader = OrderedDataReader(2)
    fsr_voltage = data_reader.get_fsr_voaltage_signal(6)

    sampling_frequency = 1980 // 2

    for channel in range(8):
        emg = data_reader.get_emg_signal(6, channel)
        path = Path(__file__).joinpath('..', 'files', 'results_graphs').resolve()
        """
        detrended_emg = detrend(emg)
        lpf_emg = self._butterworth_filter(detrended_emg, sampling_freq=sampling_freq, cutoff=cutoff)
        notched_emg = self._notch_filter(lpf_emg, sampling_freq, 50)
        notched_emg = self._notch_filter(notched_emg, sampling_freq, 100)
        wavelet_smoothed = self.wavelet_filter(notched_emg)[:-1]
        #notched_emg = self.bandpass_filter(notched_emg, low_freq=5, hi_freq=25, sampling_freq=sampling_freq)
        #return wavelet_smoothed # ruins the regression
        """
        processed_emg = preprocessor.process_emg_signal(emg, sampling_frequency, path=path)
        force = converter.fsr_voltage_to_force(fsr_voltage)
        rms_force = preprocessor.process_force_signal(force, sampling_frequency)
        time_frequency, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, processed_emg)
        _, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, fsr_voltage)
        vpp = features.get_vpp_signal(processed_emg)
        average_force = features.get_average_force_signal(rms_force)
        a, b, r_squared = features.linear_regression(np.log(vpp), average_force)
        print(r_squared)
        regression_line_fourier_plot((a, b), np.log(vpp), average_force, time_frequency, emg_dft, r_squared)

