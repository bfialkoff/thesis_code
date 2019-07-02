import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
# from tqdm import tqdm todo install tqdm
from core.bll.converter import Converter
from core.bll.data_reader import DataReader
from core.bll.preprocessing import Preprocessor
from core.utils.array import permute_axes_subtract
from core.bll.features import Features

"""
todo 
1) experiment with different filters to get a more matlabby looking signal, look into IIR like Harel did
2) attempt to get a more visual image, try a log transform
3) read about muscle firings

"""

if __name__ == '__main__':
    const = 1000
    sampling_frequency = 1980
    data_reader = DataReader(1)

    raw_emg = data_reader.get_emg_signal(np.arange(7))
    fsr_voltage = data_reader.get_fsr_voaltage_signal()
    force = Converter().convert_fsr_voltage_to_force(fsr_voltage)

    emg = Preprocessor().process_emg_signal(raw_emg, sampling_frequency)
    rms_emg = const * np.sqrt((emg ** 2) / len(emg))

    _, emg_dft = Features().get_shifted_fft_and_frequency(sampling_frequency, emg)
    _, rms_emg_dft = Features().get_shifted_fft_and_frequency(sampling_frequency, rms_emg)

    nrms = permute_axes_subtract(emg)
    rms = permute_axes_subtract(rms_emg)

    nrms_fft = permute_axes_subtract(emg_dft) * const
    rms_fft = permute_axes_subtract(rms_emg_dft) * const
    force_mat = np.repeat(force, 2 * 49, axis=0).reshape((-1, 7, 14))

    # normalize and convert to uint8 for video writing
    rms = (255 * np.abs(rms / np.max(rms))).astype(np.uint8)
    rms_fft = (255 * np.abs(rms_fft / np.max(rms_fft))).astype(np.uint8)
    force_mat = (255 * np.abs(force_mat / np.max(force))).astype(np.uint8)

    # need a better method for videoing....
    shape = nrms.shape[0], 2 * nrms.shape[1], 2 * nrms.shape[1]
    combined = np.zeros(shape, dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), sampling_frequency,
                          (2 * nrms.shape[1], 2 * nrms.shape[1]), False)
    for i, (nr, r, fnr, fr, fmat) in enumerate(zip(nrms, rms, nrms_fft, rms_fft, force_mat)):
        combined[i, :7, :7] = r
        combined[i, :7, 7:] = fr
        combined[i, 7:, :] = fmat
        frame = combined[i]
        out.write(frame)
    out.release()
