from math import pi

import matplotlib.pyplot as plt
import numpy as np

from core.bll.features import Features

two_pi = 2 * pi


def multi_plot(*lines, ls='-'):
    fig, ax = plt.subplots(len(lines), 1)
    for i, line in enumerate(lines):
        ax[i].plot(line, ls)
    plt.show()


def line(array, x_axis=None, title=None, x_label=None, y_label=None, xlim=None, ls='-'):
    fig, axes1 = plt.subplots(1, 1)

    if xlim is not None:
        axes1.set_xlim(xlim)
    if x_axis is not None:
        axes1.plot(x_axis, array, ls)
    else:
        axes1.plot(array, ls)
    if title:
        axes1.set_title(title)
    if x_label:
        axes1.set_xlabel(x_label)
    if y_label:
        axes1.set_ylabel(y_label)
    plt.show()


def scatter(force, emg, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(force, emg, marker='o')

    if title:
        ax.set_title(title, fontsize=12)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)

    plt.show()


def plot_regression_line(coeffs, x_data, y_data):
    a, b = coeffs
    min_x = np.min(x_data) - 2
    max_x = np.max(x_data) + 2

    x_line_data = np.arange(min_x, max_x, 0.1)
    y_line_data = a * x_line_data + b

    fig, ax = plt.subplots(1, 1)
    ax.plot(x_data, y_data, 'or', label='True Signal')
    ax.plot(x_line_data, y_line_data, '-b', label='Regression Line')
    ax.set_xlim([min_x, max_x])
    ax.legend()
    plt.show()


def frequency_spectrum(sampling_frequency, array):
    time_frequency, dft = Features.get_shifted_fft_and_frequency(sampling_frequency, array)
    fig, axes1 = plt.subplots(1, 1)
    axes1.plot(time_frequency, np.abs(dft))
    plt.show()

def frequency_spectrum_and_power(sampling_frequency, array):
    time_frequency, dft = Features.get_shifted_fft_and_frequency(sampling_frequency, array)
    variance = np.std(array) ** 2
    print(variance)
    variance = 1
    fft_power = variance * dft ** 2  # FFT power spectrum

    fig, axes1 = plt.subplots(1, 1)
    axes1.plot(time_frequency, dft, 'r-', label='Fourier Transform')
    axes1.plot(time_frequency, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum', alpha=0.5)
    axes1.set_xlabel('Frequency [Hz / year]', fontsize=18)
    axes1.set_ylabel('Amplitude', fontsize=18)
    axes1.legend()
    plt.show()



def regression_line_fourier_plot(coeffs, vpp_data, force_data, time_freq, emg_dft, r_squared=None):
    a, b = coeffs
    min_x = np.min(vpp_data) - 2
    max_x = np.max(vpp_data) + 2
    x_line_data = np.arange(min_x, max_x, 0.1)
    y_line_data = a * x_line_data + b

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(vpp_data, force_data, 'or', label='True Signal')
    ax1.plot(x_line_data, y_line_data, '-b', label='Regression Line')
    ax1.set_xlabel('Peak-to-Peak Voltage [mV]')
    ax1.set_ylabel('Mean Force [grams]')
    ax1.set_xlim([min_x, max_x])
    title = 'EMG Vs Force Semi-log Regression'
    if r_squared is not None:
        title += f' $R^2$={r_squared:.2f}'
    ax1.set_title(title)

    ax2.plot(time_freq, emg_dft, '.y', label='Frequency Spectrum')
    ax2.set_xlabel('frequency [Hz]')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim([-600, 600])
    ax2.set_title('EMG Fourier Transform')

    ax1.legend()
    ax2.legend()
    plt.subplots_adjust(top=0.895, bottom=0.1, left=0.11, right=0.9, hspace=0.68, wspace=0.2)

    plt.show()


def plot_emg_force(force, emg):
    time = range(len(force))

    fig, axes1 = plt.subplots(1, 1)

    axes1.plot(time, force, 'r', label='force graph')
    axes1.tick_params('y', colors='r')
    axes1.set_xlabel('samples')
    axes1.set_ylabel('force [N]')

    axes2 = axes1.twinx()
    axes2.tick_params('y', colors='b')
    axes2.plot(time, emg, 'b', label=f"emg graph")
    axes2.set_ylabel('emg [mV]')

    axes1.legend()
    axes2.legend()
    plt.show()


def plot_emg_force_all_sensors(force, emg):
    time = range(len(force))

    fig, axes = plt.subplots(2, 4)

    for i, ax in enumerate(axes.flatten()):
        ax.plot(time, force, 'r', label='force graph')
        ax.plot(time, emg[:, i], 'b', label=f"emg graph sensor {i}")

    # TODO make legend bigger
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.show()

def eval_seq2seq_prediction(ground_truth_signal, predicted_signal):
    fig, ax = plt.subplots(1, 1)
    discrete_time = range(len(ground_truth_signal))
    ax.plot(discrete_time, ground_truth_signal, 'r', label='ground truth')
    ax.plot(discrete_time, predicted_signal, 'b', label='prediction')
    ax.legend()
    plt.show()