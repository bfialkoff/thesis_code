
from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.data_reader import DataReader
from core.utils.plot import *

if __name__ == '__main__':
    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    data_reader = DataReader(1)
    fsr_voltage = data_reader.get_fsr_voaltage_signal()
    sampling_frequency = 1980

    for channel in range(8):
        emg = data_reader.get_emg_signal(channel)
        processed_emg = preprocessor.process_emg_signal(emg, sampling_frequency)
        force = converter.convert_fsr_voltage_to_force(fsr_voltage)

        time_frequency, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, processed_emg)
        _, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, fsr_voltage)

        vpp = features.get_vpp_signal(processed_emg)
        average_force = features.get_average_force_signal(force)
        a, b, r_squared = features.linear_regression(np.log(vpp), average_force)
        print(r_squared)
        regression_line_fourier_plot((a, b), np.log(vpp), average_force, time_frequency, emg_dft, r_squared)

