from pathlib import Path

from scipy.signal import detrend, medfilt


from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.data_reader import DataReader
from core.utils.plot import *

def get_emg_axes():
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Voltage [V]')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel(r'Normalized Amplitude')
    ax1.set_title('Time Domain')
    ax2.set_title('Frequency Domain')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=0.6)
    return f, (ax1, ax2)

def get_force_axes():
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Force [N/g]')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel(r'Normalized Amplitude')
    ax1.set_title('Time Domain')
    ax2.set_title('Frequency Domain')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.88, wspace=None, hspace=0.6)
    return f, (ax1, ax2)

if __name__ == '__main__':
    path = Path(__file__).joinpath('..', 'files', 'results_graphs_signal_17').resolve()
    if not path.exists():
        path.joinpath('emg').mkdir(parents=True)
        path.joinpath('force').mkdir(parents=True)

    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    data_reader = DataReader(17)
    sampling_frequency = 1980 // 2


    fsr_voltage = data_reader.get_fsr_voaltage_signal()

    force = converter.fsr_voltage_to_force(fsr_voltage)
    time_axis = np.arange(len(force)) / 1980
    time_frequency, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, force)
    f, (ax1, ax2) = get_force_axes()
    ax1.plot(time_axis, force)
    ax2.plot(time_frequency, force_dft)
    f.suptitle('Raw Force Signal')
    plt.savefig(path.joinpath('force', 'raw_force.eps'))
    plt.savefig(path.joinpath('force', 'raw_force.png'))
    plt.savefig(path.joinpath('force', 'raw_force.tiff'))

    med_force = medfilt(force, 501)
    _, med_force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, med_force)
    f, (ax1, ax2) = get_force_axes()
    ax1.plot(time_axis, med_force)
    ax2.plot(time_frequency, med_force_dft)
    f.suptitle('Median Filtered Force Signal')
    plt.savefig(path.joinpath('force','med_filt.eps'))
    plt.savefig(path.joinpath('force','med_filt.png'))
    plt.savefig(path.joinpath('force','med_filt.tiff'))

    rms_force = preprocessor.rms_filter(med_force, kernel_length=500)
    _, rms_force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, rms_force)
    f, (ax1, ax2) = get_force_axes()
    ax1.plot(time_axis, rms_force)
    ax2.plot(time_frequency, rms_force_dft)
    f.suptitle('RMS Force Signals')
    plt.savefig(path.joinpath('force', 'rms_force.eps'))
    plt.savefig(path.joinpath('force', 'rms_force.png'))
    plt.savefig(path.joinpath('force', 'rms_force.tiff'))

    for channel in range(8):
        emg = data_reader.get_emg_signal(channel)
        time_axis = np.arange(len(emg)) / 1980

        f, (ax1, ax2) = get_emg_axes()
        detrended_emg = detrend(emg)
        time_frequency, detrended_emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, detrended_emg)
        ax1.plot(time_axis, emg)
        ax2.plot(time_frequency, detrended_emg_dft)
        f.suptitle('Raw EMG Signal')
        plt.savefig(path.joinpath('emg', 'raw.eps'))
        plt.savefig(path.joinpath('emg', 'raw.png'))
        plt.savefig(path.joinpath('emg', 'raw.tiff'))
        f, (ax1, ax2) = get_emg_axes()
        lpf_emg = preprocessor._butterworth_filter(detrended_emg, sampling_freq=sampling_frequency, cutoff=400)
        _, lpf_emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, lpf_emg)
        ax1.plot(time_axis, lpf_emg)
        ax2.plot(time_frequency, lpf_emg_dft)
        f.suptitle('Low Pass Filtered EMG Signal')
        plt.savefig(path.joinpath('emg', 'lpf.eps'))
        plt.savefig(path.joinpath('emg', 'lpf.png'))
        plt.savefig(path.joinpath('emg', 'lpf.tiff'))
        f, (ax1, ax2) = get_emg_axes()
        notched_emg = preprocessor._notch_filter(lpf_emg, sampling_frequency, 50)
        notched_emg = preprocessor._notch_filter(notched_emg, sampling_frequency, 100)
        _, notched_emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, notched_emg)
        ax1.plot(time_axis, notched_emg)
        ax2.plot(time_frequency, notched_emg_dft)
        f.suptitle('Notched EMG Signal')
        plt.savefig(path.joinpath('emg', 'notched.eps'))
        plt.savefig(path.joinpath('emg', 'notched.png'))
        plt.savefig(path.joinpath('emg', 'notched.tiff'))
        f, (ax1, ax2) = get_emg_axes()
        narrow_band = preprocessor.bandpass_filter(notched_emg, low_freq=5, hi_freq=25, sampling_freq=sampling_frequency)
        _, narrow_band_emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, narrow_band)
        ax1.plot(time_axis, narrow_band)
        ax2.plot(time_frequency, narrow_band_emg_dft)
        f.suptitle('Narrow Bandpass Filtered EMG Signal')
        plt.savefig(path.joinpath('emg', 'narrow_band.eps'))
        plt.savefig(path.joinpath('emg', 'narrow_band.png'))
        plt.savefig(path.joinpath('emg', 'narrow_band.tiff'))
        break	
        """
        
        rms_force = preprocessor.process_force_signal(force, sampling_frequency)
        _, emg_dft = features.get_shifted_fft_and_frequency(sampling_frequency, processed_emg)
        _, force_dft = features.get_shifted_fft_and_frequency(sampling_frequency, fsr_voltage)
        vpp = features.get_vpp_signal(processed_emg)
        average_force = features.get_average_force_signal(rms_force)
        a, b, r_squared = features.linear_regression(np.log(vpp), average_force)
        print(r_squared)
        regression_line_fourier_plot((a, b), np.log(vpp), average_force, time_frequency, emg_dft, r_squared)
        """
