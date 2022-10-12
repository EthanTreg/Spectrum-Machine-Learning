import os
import numpy as np
import pandas as pd
from astropy.io import fits


def progress_bar(i: int, total: int):
    """
    Terminal progress bar

    Parameters
    ----------
    i : int
        Current progress
    total : int
        Completion number
    """
    length = 50
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = '█' * filled + '-' * (length - filled)
    print(f'\rProgress: |{bar_fill}| {int(percent)}%\t', end='')

    if i == total:
        print()


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


def spectrum_data(data_dir: str, spectrum: str, cut_off: list = None) -> np.ndarray:
    """
    Fetches binned data from spectrum

    Returns the binned x, y spectrum data as a 2D array

    Parameters
    ----------
    data_dir : string
        Path to the root directory where spectrum and backgrounds are located
    spectrum : string
        Name of the spectrum to fetch
    cut_off : list, default=[0.3, 10]
        Range of accepted data in keV

    Returns
    -------
    ndarray, shape=(2, N)
        Binned spectrum data
    """
    # Initialize variables
    x_bin = np.array(())
    y_bin = np.array(())
    bins = np.array([[0, 20, 248, 600, 1200, 1494, 1500], [2, 3, 4, 5, 6, 2, 1]], dtype=int)
    # bins = np.array([
    #     [0, 14, 16, 20, 248, 600, 1200, 1248, 1250, 1254, 1494, 1500],
    #     [2, 1, 2, 3, 4, 5, 6, 2, 4, 6, 2, 1]
    # ], dtype=int)

    if not cut_off:
        cut_off = [0.3, 10]

    # Fetch spectrum & background fits files
    with fits.open(data_dir + '/' + spectrum) as f:
        spectrum_info = f[1].header
        spectrum = pd.DataFrame(f[1].data)
        detectors = int(f[1].header['RESPFILE'][7:9])

    with fits.open(data_dir + '/' + spectrum_info['BACKFILE']) as f:
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

    cut_indices = np.argwhere((x_bin < cut_off[0]) | (x_bin > cut_off[1]))
    x_bin = np.delete(x_bin, cut_indices)
    y_bin = np.delete(y_bin, cut_indices)

    return np.array((x_bin, y_bin))


def preprocess():
    """
    Preprocess spectra and save to file
    """
    # Initialize variables
    data_dir = '../../Documents/Nicer_Data/ethan'
    labels_path = './data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    spectra = []

    # Fetch spectra names & labels
    spectra_files = spectra_names(data_dir)
    labels = np.loadtxt(labels_path, skiprows=6, dtype=str)

    # Remove data that doesn't have a label
    bad_indices = np.invert(np.in1d(spectra_files, labels[:, 6]))
    spectra_files = np.delete(spectra_files, bad_indices)

    # Fetch spectra data
    for i, spectrum_name in enumerate(spectra_files):
        spectra.append(spectrum_data(data_dir, spectrum_name)[1])
        progress_bar(i, spectra_files.size)

    # Save spectra data
    spectra = np.vstack(spectra)
    np.save('./data/preprocessed_spectra', spectra)


if __name__ == '__main__':
    preprocess()
