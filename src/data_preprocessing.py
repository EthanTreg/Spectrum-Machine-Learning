import os
import numpy as np
import pandas as pd
from astropy.io import fits

from src.utils.utils import progress_bar


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
        nchan: int) -> tuple[np.ndarray, np.ndarray]:
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
    (ndarray, ndarray)
        Binned (x, y) data
    """
    x_data = x_data[clow:chi]
    y_data = y_data[clow:chi]

    x_data = np.mean(x_data.reshape(-1, int(nchan)), axis=1)
    y_data = np.sum(y_data.reshape(-1, int(nchan)), axis=1)

    return x_data, y_data


def spectra_names(data_dir: str) -> np.ndarray:
    """
    Fetches the file names of all spectra up to the defined data amount

    Parameters
    ----------
    data_dir : string
        Directory of the spectra dataset

    Returns
    -------
    ndarray
        Array of spectra file names
    """
    # Fetch all files within directory
    all_files = os.listdir(data_dir)

    # Remove all files that aren't spectra
    return np.delete(all_files, np.char.find(all_files, '.jsgrp') == -1)


def spectrum_data(
        spectrum_path: str,
        background_dir: str = '',
        cut_off: list = None) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Fetches binned data from spectrum

    Returns the binned x, y spectrum data as a 2D array

    Parameters
    ----------
    spectrum_path : string
        File path to the spectrum
    background_dir : string, default = ''
        Path to the root directory where the background is located
    cut_off : list, default = [0.3, 10]
        Range of accepted data in keV

    Returns
    -------
    (ndarray, ndarray, int)
        Binned spectrum data and number of detectors
    """
    # Initialize variables
    x_bin = np.array(())
    y_bin = np.array(())
    bins = np.array([[0, 20, 248, 600, 1200, 1494, 1500], [2, 3, 4, 5, 6, 2, 1]], dtype=int)

    if not cut_off:
        cut_off = [0.3, 10]

    # Fetch spectrum & background fits files
    with fits.open(spectrum_path) as f:
        spectrum_info = f[1].header
        spectrum = pd.DataFrame(f[1].data)
        response = f[1].header['RESPFILE']
        detectors = int(response[response.find('_d') + 2:response.find('_d') + 4])

    with fits.open(background_dir + spectrum_info['BACKFILE']) as f:
        background_info = f[1].header
        background = pd.DataFrame(f[1].data)

    # Pre binned data
    x_data = channel_kev(spectrum.CHANNEL.to_numpy())
    if 'COUNTS' in spectrum and 'COUNTS' in background:
        y_data = (
                 spectrum.COUNTS /
                 spectrum_info['EXPOSURE'] - background.COUNTS /
                 background_info['EXPOSURE']
                 ).to_numpy() / detectors
    elif 'COUNTS' in spectrum:
        y_data = (
                 spectrum.COUNTS /
                 spectrum_info['EXPOSURE'] - background.RATE
         ).to_numpy() / detectors
    else:
        y_data = (spectrum.RATE - background.RATE).to_numpy() / detectors

    # Bin data
    for i in range(bins.shape[1] - 1):
        x_new, y_new = binning(x_data, y_data, bins[0, i], bins[0, i + 1], bins[1, i])
        y_new /= bins[1, i] * 1e-2
        x_bin = np.append(x_bin, x_new)
        y_bin = np.append(y_bin, y_new)

    cut_indices = np.argwhere((x_bin < cut_off[0]) | (x_bin > cut_off[1]))
    x_bin = np.delete(x_bin, cut_indices)
    y_bin = np.delete(y_bin, cut_indices)

    return x_bin, y_bin, detectors


def preprocess():
    """
    Preprocess spectra and save to file
    """
    # Initialize variables
    data_dir = '../../../Documents/Nicer_Data/ethan/'
    labels_path = '../data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    spectra = []

    # Fetch spectra names & labels
    # spectra_files = spectra_names(data_dir)
    labels = np.loadtxt(labels_path, skiprows=6, dtype=str)
    spectra_files = labels[:, 6]

    # Fetch spectra data
    for i, spectrum_name in enumerate(spectra_files):
        spectra.append(spectrum_data(data_dir + spectrum_name, data_dir)[1])
        progress_bar(i, spectra_files.size)

    # Save spectra data
    spectra = np.vstack(spectra)
    np.save('./data/preprocessed_spectra', spectra)


if __name__ == '__main__':
    preprocess()
