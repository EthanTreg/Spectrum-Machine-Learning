import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt
from xspec import Spectrum, Plot


def channel_kev(channel: np.ndarray) -> np.ndarray:
    """
    Convert units of channel to keV

    Parameters
    ----------
    channel : ndarray
        Detector channels

    Returns
    -------
    ndarray
        Channels in units of keV
    """
    return (channel * 10 + 5) / 1e3


def binning(
        x_data: np.ndarray,
        y_data: np.ndarray,
        clow: float,
        chi: float,
        nchan: int) -> (np.ndarray, np.ndarray):
    """
    Bins x & y data

    Removes data of bad quality defined as 1 [Not yet implemented]

    Parameters
    ----------
    x_data : ndarray
        x data that will be averaged per bin
    y_data : ndarray
        y data that will be summed per bin
    clow : float
        Lower limit of data that will be binned
    chi : float
        Upper limit of datat that will be binned
    nchan : int
        Number of channels per bin

    Returns
    -------
    ndarray
        binned x data
    ndarray
        binned y data
    """

    x_data = x_data[clow:chi]
    y_data = y_data[clow:chi]

    x_data = np.mean(x_data.reshape(-1, int(nchan)), axis=1)
    y_data = np.sum(y_data.reshape(-1, int(nchan)), axis=1)

    return x_data, y_data


def diff_plot(x_data_1: np.ndarray, y_data_1: np.ndarray, x2: np.ndarray, y2: np.ndarray):
    """
    Plots the ratio between two data sets (set 1 / set 2)

    Length of set 1 >= length of set 2

    Parameters
    ----------
    x_data_1 : ndarray
        x values for first data set
    y_data_1 : ndarray
        y values for first data set
    x2 : ndarray
        x values for second data set
    y2 : ndarray
        y values for second data set
    """
    matching_indices = np.array((), dtype=int)

    for i in x_data_1:
        matching_indices = np.append(matching_indices, np.argmin(np.abs(x2 - i)))

    diff = y_data_1 / y2[matching_indices]

    plt.title('PyXspec compared to fits', fontsize=24)
    plt.scatter(x_data_1, diff)
    plt.xlabel('Energy (keV)', fontsize=20)
    plt.ylabel('PyXspec / fits data', fontsize=20)
    plt.text(0.05, 0.2,
             f'Average ratio: {round(np.mean(diff), 2)}',
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
    plt.xlim([0.15, 15])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=20)


def main():
    """
    Main function for preprocessing the data
    """
    # Initialize variables
    plot_diff = True
    spectra = 'js_ni2101010101_0mpu7_goddard_GTI0'
    # spectra = 'js_ni0001020103_0mpu7_goddard_GTI0'
    x_bin = np.array(())
    y_bin = np.array(())

    # Binning method starting from 1
    # bins = np.array([[0, 1, 21, 249, 601, 1201, 1495, 1500], [1, 2, 3, 4, 5, 6, 1, 1]], dtype=int)
    # Binning method starting from 0 & including 1494 - 1500
    bins = np.array([[0, 20, 248, 600, 1200, 1500], [2, 3, 4, 5, 6, 1]], dtype=int)

    # Fetch spectrum & background fits files
    with fits.open('./data/' + spectra + '.jsgrp') as f:
        spectrum_info = f[1].header
        spectrum = pd.DataFrame(f[1].data)
        detectors = int(f[1].header['RESPFILE'][7:9])

    with fits.open('./data/' + spectra + '.bg') as f:
        background_info = f[1].header
        background = pd.DataFrame(f[1].data)

    # Pre binned data
    x_data = channel_kev(spectrum.CHANNEL.to_numpy())
    y_data = (
             spectrum.COUNTS /
             spectrum_info['EXPOSURE'] - background.COUNTS /
             background_info['EXPOSURE']
             ).to_numpy() / detectors

    # Bin data
    for i in range(bins.shape[1] - 1):
        x_new, y_new = binning(x_data, y_data, bins[0, i], bins[0, i + 1], bins[1, i])
        y_new /= bins[1, i] * 1e-2
        x_bin = np.append(x_bin, x_new)
        y_bin = np.append(y_bin, y_new)

    # PyXspec data & plotting
    Spectrum('./data/' + spectra + '.jsgrp')
    Plot.device = '/NULL'
    Plot.xAxis = 'keV'
    # Plot.yLog = True
    # Plot.addCommand('rescale x 0.15 15')
    # Plot.addCommand('rescale y 1e-3 60')
    Plot('data')

    # PyXspec data
    x_px = np.array(Plot.x())
    y_px = np.array(Plot.y()) / detectors

    # Pyplot plotting of binned fits data & PyXspec data
    if plot_diff:
        plt.figure(constrained_layout=True, figsize=(16, 18))
        plt.subplot(2, 1, 2)
        diff_plot(x_px, y_px, x_bin, y_bin)
        plt.subplot(2, 1, 1)
    else:
        plt.figure(constrained_layout=True, figsize=(16, 12))

    spectrum_plot(x_bin, y_bin, x_px, y_px)

    plt.show()


if __name__ == '__main__':
    main()
