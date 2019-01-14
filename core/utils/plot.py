from math import pi

import matplotlib.pyplot as plt
import numpy as np

from core.bll.features import Features

two_pi = 2 * pi


def line(array, x_axis=None, title=None, x_label=None, y_label=None, xlim=None):
    fig, axes1 = plt.subplots(1, 1)

    if xlim is not None:
        plt.xlim(xlim)
    if x_axis is not None:
        axes1.plot(x_axis, array)
    else:
        axes1.plot(array)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.show()


def scatter(force, emg, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(force, emg, marker='o')

    if title:
        plt.title(title, fontsize=12)
    if x_label:
        plt.xlabel(x_label, fontsize=12)
    if y_label:
        plt.ylabel(y_label, fontsize=12)

    plt.show()


def frequency_spectrum(sampling_frequency, array):
    time_frequency, dft = Features().get_shifted_fft_and_frequency(sampling_frequency, array)
    fig, axes1 = plt.subplots(1, 1)
    axes1.plot(time_frequency, np.abs(dft))
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
