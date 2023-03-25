"""
Compares data_preprocessing with PyXspec processing
This is for testing purposes only
"""
import os

import numpy as np
from xspec import Spectrum, Plot
from matplotlib import pyplot as plt

from fspnet.data_preprocessing import correct_spectrum
from fspnet.utils.plots import diff_plot, spectrum_plot


def main():
    """
    Main function for preprocessing the data
    """
    # Variables
    plot_diff = True
    # data_root = '../'
    # spectrum = './data/synth_spectra/synth_000000.fits'
    data_root = '../../../Documents/Nicer_Data/spectra/'
    spectrum = './js_ni2101010101_0mpu7_goddard_GTI0.jsgrp'

    # Constant
    root_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(data_root)
    detectors, x_bin, (y_bin, _) = correct_spectrum(spectrum, '')

    # PyXspec data & plotting
    Spectrum(spectrum)
    Plot.device = '/NULL'
    Plot.xAxis = 'keV'
    Plot('data')
    os.chdir(root_dir)

    # PyXspec data
    x_px = np.array(Plot.x())[:-1]
    y_px = np.array(Plot.y())[:-1] / detectors


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
