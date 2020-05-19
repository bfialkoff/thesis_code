from pathlib import Path

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.ordered_data_reader import OrderedDataReader
from core.utils.plot import *

from tqdm import tqdm

def get_shifted_fft_and_frequency(sampling_frequency, signal):
    dft = np.fft.fft(signal)
    abs_dft = np.abs(dft)
    len_dft = len(dft)
    discrete_frequency = np.arange(0, len_dft)
    discrete_frequency[round(len_dft / 2):] = discrete_frequency[round(len_dft / 2):] - len_dft
    time_frequency = (sampling_frequency / len_dft) * discrete_frequency
    return time_frequency, dft, abs_dft

if __name__ == '__main__':
    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    subject = 3
    signal = 2
    num_subjects_gen = range(2, 8)
    num_signals_gen = range(1, 21)
    num_channels_gen= range(8)


    sampling_frequency = 1980 // 2


    for subject in tqdm(num_subjects_gen):
        data_reader = OrderedDataReader(subject)
        path = Path(__file__).joinpath('..', 'files', 'cole_cole_graphs',
                                       f'subject_{subject}')
        for signal in num_signals_gen:
            subject_path = path.joinpath(f'signal_{signal}').resolve()
            if not subject_path.exists():
                subject_path.mkdir(parents=True)
            for channel in num_channels_gen:
                raw_emg = data_reader.get_emg_signal(signal, channel)
                emg = raw_emg.copy()
                processed_emg = preprocessor.process_emg_signal(raw_emg, sampling_frequency, path=subject_path)
                #processed_emg = preprocessor.bandpass_filter(processed_emg, low_freq, hi_freq, sampling_frequency)
                full_time_frequency, emg_dft, abs_dft = get_shifted_fft_and_frequency(sampling_frequency, processed_emg)
                abs_dft = abs_dft[:len(emg_dft) // 2]
                emg_dft = emg_dft[:len(emg_dft) // 2]
                time_frequency = full_time_frequency[:len(full_time_frequency) // 2]
                r_dft, i_dft = emg_dft.real, emg_dft.imag
                pr_dft, pi_dft = r_dft, i_dft
                """
                inds = r_dft.argsort()
                pr_dft = r_dft[inds]
                pi_dft = i_dft[inds]
                """
                f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

                ax1.plot(time_frequency, abs_dft)
                ax2.plot(np.arange(len(full_time_frequency)) / sampling_frequency, emg)
                ax3.plot(time_frequency, r_dft, 'r', linewidth=3, label='real')
                ax3.plot(time_frequency, i_dft, 'b', alpha=0.5, label='imag')
                ax4.plot(pr_dft, pi_dft, label='imag vs real')

                ax1.set_title('Fourier Spectrum\processed signal')
                ax2.set_title('Raw EMG signal')
                ax3.set_title('real and imaginary\n parts of fft vs freq.')
                ax4.set_title('imag part of spectrum\n as a func. of real part')

                ax1.set_xlabel('f[Hz]')
                ax1.set_ylabel('||F{emg(t)}(f)||')
                ax2.set_xlabel('t[s]')
                ax2.set_ylabel('EMG')
                ax3.set_xlabel('f[Hz]')
                ax3.set_ylabel('F{emg(t)}(f)')
                ax4.set_xlabel('real(F{emg(t)}(f))')
                ax4.set_ylabel('imag(F{emg(t)}(f))')

                ax3.legend()
                ax4.legend()
                plt.subplots_adjust(hspace=0.9, wspace=0.31,
                                    right=0.96, left=0.11)
                #plt.savefig(subject_path.joinpath(f'channel_{channel}.png'))
                plt.show()
                plt.close()
