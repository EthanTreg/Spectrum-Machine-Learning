"""
Utility functions for preprocessing including fetching spectra names and corrects spectrum
"""
import numpy as np
import pandas as pd
from numpy import ndarray
from astropy.io import fits


def _channel_kev(channel: ndarray) -> ndarray:
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


def _create_bin(
        x_data: ndarray,
        y_data: ndarray,
        clow: float,
        chi: float,
        nchan: int) -> tuple[ndarray, ndarray]:
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


def _binning(x_data: ndarray, y_data: ndarray, bins: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    Bins data to match binning performed in Xspec

    Parameters
    ----------
    x_data : ndarray
        x data to be binned
    y_data : ndarray
        y data to be binned
    bins : ndarray
        Array of bin change index & bin size

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Binned x & y data & bin energy per data point
    """
    # Initialize variables
    x_bin = np.array(())
    y_bin = np.array(())
    energy_bin = np.array(())

    # Bin data
    for i in range(bins.shape[1] - 1):
        x_new, y_new = _create_bin(x_data, y_data, bins[0, i], bins[0, i + 1], bins[1, i])
        x_bin = np.append(x_bin, x_new)
        y_bin = np.append(y_bin, y_new)
        energy_bin = np.append(energy_bin, [bins[1, i] * 1e-2] * y_new.size)

    return x_bin, y_bin, energy_bin


def _corrected_spectrum(
        detectors: int,
        spectrum_exposure: int,
        back_exposure: int,
        spectrum: pd.DataFrame,
        background: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corrects spectrum using background and normalizes using exposure and number of detectors

    Parameters
    ----------
    detectors : integer
        Number of detectors
    spectrum_exposure : integer
        Exposure time for spectrum
    back_exposure : integer
        Exposure time for background
    spectrum : DataFrame
        Original spectrum in either counts or rate
    background : DataFrame
        Background in either counts or rate

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Corrected normalized spectrum and spectrum counts
    """
    if 'COUNTS' in spectrum:
        spectrum['RATE'] = spectrum['COUNTS'] / spectrum_exposure

    if 'COUNTS' in background:
        background['RATE'] = background['COUNTS'] / back_exposure

    y_data = (spectrum['RATE'] - background['RATE']) / detectors

    return y_data, spectrum['RATE'] * spectrum_exposure


def _spectrum_data(
        detectors: int,
        back_exposure: int,
        spectrum: pd.DataFrame,
        background: pd.DataFrame,
        spectrum_info: fits.fitsrec,
        cut_off: tuple[float, float] = None) -> tuple[ndarray, ndarray, ndarray]:
    """
    Corrects and normalizes the spectrum using background, exposure time and number of detectors
    Also calculates the normalized uncertainty from data counts

    Parameters
    ----------
    detectors : integer
        Number of detectors
    back_exposure : integer
        Exposure time for background
    spectrum : DataFrame
        Original spectrum in either counts or rate
    background : DataFrame
        Background in either counts or rate
    spectrum_info : fitsrec
        Fits information on the spectrum
    cut_off : tuple[float, float], default = (0.3, 10)
        Lower and upper limit of accepted data in keV

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Binned x values, y values and uncertainty
    """
    # Initialize variables
    bins = np.array([[0, 20, 248, 600, 1200, 1494, 1500], [2, 3, 4, 5, 6, 2, 1]], dtype=int)

    if not cut_off:
        cut_off = (0.3, 10)

    # Pre binned data
    x_data = _channel_kev(spectrum['CHANNEL'].to_numpy())
    y_data, counts = _corrected_spectrum(
        detectors,
        spectrum_info['EXPOSURE'],
        back_exposure,
        spectrum,
        background
    )

    # Bin data
    y_bin = _binning(x_data, y_data.to_numpy(), bins)[1]
    x_bin, counts_bin, energy_bin = _binning(x_data, counts.to_numpy(), bins)
    uncertainty = np.maximum(np.sqrt(np.maximum(counts_bin, 0)), 1) / \
                  (spectrum_info['EXPOSURE'] * detectors)

    # Energy normalization
    y_bin /= energy_bin
    uncertainty /= energy_bin

    # Energy range cut-off
    cut_indices = np.argwhere((x_bin < cut_off[0]) | (x_bin > cut_off[1]))
    x_bin = np.delete(x_bin, cut_indices)
    y_bin = np.delete(y_bin, cut_indices)
    uncertainty = np.delete(uncertainty, cut_indices)

    return x_bin, y_bin, uncertainty


def correct_spectrum(
        spectrum_path: str,
        background_dir: str = '',
        aug_count: int = 0,
        cut_off: tuple[float, float] = None) -> tuple[int, ndarray, list[tuple[ndarray, ndarray]]]:
    """
    Fetches spectrum, corrects, normalizes and bins data

    Parameters
    ----------
    spectrum_path : string
        File path to the spectrum
    background_dir : string, default = ''
        Path to the root directory where the background is located
    aug_count : int, default = 0
        Number of augmentations to perform
    cut_off : tuple[float, float], default = (0.3, 10)
        Lower and upper limit of accepted data in keV

    Returns
    -------
    tuple[integer, ndarray, list[tuple[ndarray, ndarray]]]
        Number of detectors, binned energies and list of binned spectrum data & binned uncertainties
        for original + augmentations
    """
    data = []

    # Fetch spectrum & background fits files
    with fits.open(spectrum_path) as file:
        spectrum_info = file[1].header
        spectrum = pd.DataFrame(file[1].data)

        if 'FKRSP001' in spectrum_info:
            response = spectrum_info['FKRSP001']
        else:
            response = spectrum_info['RESPFILE']

        detectors = int(response[response.find('_d') + 2:response.find('_d') + 4])

    with fits.open(background_dir + spectrum_info['BACKFILE']) as file:
        back_exposure = file[1].header['EXPOSURE']
        background = pd.DataFrame(file[1].data).iloc[:, 1].to_frame()

    # Normalize and bin spectrum
    x_bin, y_bin, uncertainty = _spectrum_data(
        detectors,
        back_exposure,
        spectrum,
        background,
        spectrum_info,
        cut_off,
    )

    data.append((y_bin, uncertainty))
    scaling = np.random.uniform(0.5, 2, aug_count)

    # Randomly scale background for augmentation
    for scale in scaling:
        y_bin = _spectrum_data(
            detectors,
            back_exposure,
            spectrum,
            background * scale,
            spectrum_info,
            cut_off,
        )[1]

        data.append((np.maximum(0, y_bin), uncertainty))

    return detectors, x_bin, data
