import matplotlib.pyplot as plt
import numpy as np

def line(array, title=None, x_label=None, y_label=None):
    fig, axes1 = plt.subplots(1, 1)
    axes1.plot(array)

    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.show()

def scatter(emg, force, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(emg, force)

    if title:
        plt.title(title, fontsize=12)
    if x_label:
        plt.xlabel(x_label, fontsize=12)
    if y_label:
        plt.ylabel(y_label, fontsize=12)

    plt.show()

def frequency_spectrum(array, x_values=None):
    dft = np.fft.fft(array)
    fig, axes1 = plt.subplots(1, 1)
    if x_values is None:
        axes1.plot(np.abs(dft))
    else:
        axes1.plot(x_values, (dft))
    plt.show()

def plot_emg_force(force, emg):
    time = range(len(force))

    fig, axes1 = plt.subplots(1,1)

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

    fig, axes = plt.subplots(2,4)

    for i, ax in enumerate(axes.flatten()):
        ax.plot(time, force, 'r', label='force graph')
        ax.plot(time, emg[:,i], 'b', label=f"emg graph sensor {i}")

    # TODO make legend bigger
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.show()
