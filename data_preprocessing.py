import numpy as np
import pandas as pd
from astropy.io import fits


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


def spectrum_data(data_dir: str, spectrum: str) -> np.ndarray:
    """
    Fetches binned data from spectrum

    Returns the binned x, y spectrum data as a 2D array

    Parameters
    ----------
    data_dir : string
        Path to the root directory where spectrum and backgrounds are located
    spectrum : string
        Name of the spectrum to fetch

    Returns
    -------
    ndarray, shape=(2, N)
        Binned spectrum data
    """
    # Initialize variables
    x_bin = np.array(())
    y_bin = np.array(())
    bins = np.array([[0, 20, 248, 600, 1200, 1500], [2, 3, 4, 5, 6, 1]], dtype=int)

    # Fetch spectrum & background fits files
    with fits.open(data_dir + '/spectra/' + spectrum) as f:
        spectrum_info = f[1].header
        spectrum = pd.DataFrame(f[1].data)
        detectors = int(f[1].header['RESPFILE'][7:9])

    with fits.open(data_dir + '/backgrounds/' + spectrum_info['BACKFILE']) as f:
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

    return np.array((x_bin, y_bin))
