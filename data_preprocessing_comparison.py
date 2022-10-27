import os

import numpy as np
from xspec import Spectrum, Plot
from matplotlib import pyplot as plt

from data_preprocessing import spectrum_data


def diff_plot(
        x_data_1: np.ndarray,
        y_data_1: np.ndarray,
        x_data_2: np.ndarray,
        y_data_2: np.ndarray):
    """
    Plots the ratio between two data sets (set 1 / set 2)

    Parameters
    ----------
    x_data_1 : ndarray
        x values for first data set
    y_data_1 : ndarray
        y values for first data set
    x_data_2 : ndarray
        x values for second data set
    y_data_2 : ndarray
        y values for second data set
    """
    matching_indices = np.array((), dtype=int)

    for i in x_data_1:
        matching_indices = np.append(matching_indices, np.argmin(np.abs(x_data_2 - i)))

    diff = y_data_1 / y_data_2[matching_indices]

    plt.title('PyXspec compared to fits', fontsize=24)
    plt.scatter(x_data_1, diff)
    plt.xlabel('Energy (keV)', fontsize=20)
    plt.ylabel('PyXspec / fits data', fontsize=20)
    plt.text(0.05, 0.2,
             f'Average ratio: {round(np.mean(diff), 3)}',
             fontsize=16, transform=plt.gca().transAxes
             )


def spectrum_plot(x_bin: np.ndarray, y_bin: np.ndarray, x_px: np.ndarray, y_px: np.ndarray):
    """
    Plots the spectrum of PyXspec & fits data

    Parameters
    ----------
    x_bin : ndarray
        Binned x data from fits file
    y_bin : ndarray
        Binned y data from fits file
    x_px : ndarray
        x data from PyXspec
    y_px : ndarray
        y data from PyXspec
    """
    plt.title('Spectrum of PyXspec & fits', fontsize=24)
    plt.xlabel('Energy (keV)', fontsize=20)
    plt.ylabel(r'Counts $s^{-1}$ $detector^{-1}$ $keV^{-1}$', fontsize=20)
    plt.scatter(x_bin, y_bin, label='Fits data', marker='x')
    plt.scatter(x_px, y_px, label='PyXspec data')
    plt.xlim([0.15, 14.5])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=20)


def main():
    """
    Main function for preprocessing the data
    """
    # Variables
    plot_diff = True
    data_root = ''
    spectrum = './data/synth_spectra/synth_03.fits'

    # Constant
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # PyXspec data & plotting
    os.chdir(data_root)
    Spectrum(spectrum)
    Plot.device = '/NULL'
    Plot.xAxis = 'keV'
    # Plot.yLog = True
    # Plot.addCommand('rescale x 0.15 15')
    # Plot.addCommand('rescale y 1e-3 60')
    Plot('data')
    os.chdir(root_dir)

    # PyXspec data
    x_px = np.array(Plot.x())[:-1]
    y_px = np.array(Plot.y())[:-1] / 49

    x_bin, y_bin, _ = spectrum_data(spectrum, '')

    # Pyplot plotting of binned fits data & PyXspec data
    if plot_diff:
        plt.figure(constrained_layout=True, figsize=(16, 18))
        plt.subplot(2, 1, 2)
        diff_plot(x_bin, y_bin, x_px, y_px)
        plt.subplot(2, 1, 1)
    else:
        plt.figure(constrained_layout=True, figsize=(16, 12))

    spectrum_plot(x_bin, y_bin, x_px, y_px)

    plt.show()


if __name__ == '__main__':
    main()
