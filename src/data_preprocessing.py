import os
from time import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from numpy import ndarray
from astropy.io import fits

from src.utils.utils import progress_bar


def channel_kev(channel: ndarray) -> ndarray:
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


def create_bin(
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


def binning(x_data: ndarray, y_data: ndarray, bins: ndarray) -> tuple[ndarray, ndarray, ndarray]:
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
        x_new, y_new = create_bin(x_data, y_data, bins[0, i], bins[0, i + 1], bins[1, i])
        x_bin = np.append(x_bin, x_new)
        y_bin = np.append(y_bin, y_new)
        energy_bin = np.append(energy_bin, [bins[1, i] * 1e-2] * y_new.size)

    return x_bin, y_bin, energy_bin


def spectrum_data(
        spectrum_path: str,
        background_dir: str = '',
        cut_off: list = None) -> tuple[ndarray, ndarray, ndarray, int]:
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
    (ndarray, ndarray, ndarray, integer)
        Binned energies, binned spectrum data, binned uncertainties & number of detectors
    """
    bins = np.array([[0, 20, 248, 600, 1200, 1494, 1500], [2, 3, 4, 5, 6, 2, 1]], dtype=int)

    if not cut_off:
        cut_off = [0.3, 10]

    # Fetch spectrum & background fits files
    with fits.open(spectrum_path) as file:
        spectrum_info = file[1].header
        spectrum = pd.DataFrame(file[1].data)
        response = file[1].header['RESPFILE']
        detectors = int(response[response.find('_d') + 2:response.find('_d') + 4])

    try:
        with fits.open(background_dir + spectrum_info['BACKFILE']) as file:
            background_info = file[1].header
            background = pd.DataFrame(file[1].data)
    except FileNotFoundError:
        spectrum_info['BACKFILE'] = spectrum_info['BACKFILE'].replace(
            'spectra/synth_',
            'spectra/synth_0'
        )

        with fits.open(background_dir + spectrum_info['BACKFILE']) as file:
            background_info = file[1].header
            background = pd.DataFrame(file[1].data)


    # Pre binned data
    x_data = channel_kev(spectrum.CHANNEL.to_numpy())
    if 'COUNTS' in spectrum and 'COUNTS' in background:
        y_data = (
                 spectrum.COUNTS /
                 spectrum_info['EXPOSURE'] - background.COUNTS /
                 background_info['EXPOSURE']
                 ).to_numpy() / detectors

        counts_bin = binning(x_data, spectrum.COUNTS.to_numpy(dtype=float), bins)[1]
    elif 'COUNTS' in spectrum:
        y_data = (
                 spectrum.COUNTS /
                 spectrum_info['EXPOSURE'] - background.RATE
         ).to_numpy() / detectors

        counts_bin = binning(x_data, spectrum.COUNTS.to_numpy(dtype=float), bins)[1]
    else:
        y_data = (spectrum.RATE - background.RATE).to_numpy() / detectors

        counts_bin = binning(
            x_data,
            (spectrum.RATE * spectrum_info['EXPOSURE']).to_numpy(dtype=float),
            bins
        )[1]

    # Bin data
    x_bin, y_bin, energy_bin = binning(x_data, y_data, bins)
    uncertainty = np.sqrt(counts_bin) / (spectrum_info['EXPOSURE'] * detectors)

    # Energy normalization
    y_bin /= energy_bin
    uncertainty /= energy_bin

    # Energy range cut-off
    cut_indices = np.argwhere((x_bin < cut_off[0]) | (x_bin > cut_off[1]))
    x_bin = np.delete(x_bin, cut_indices)
    y_bin = np.delete(y_bin, cut_indices)
    uncertainty = np.delete(uncertainty, cut_indices)

    return x_bin, y_bin, uncertainty, detectors


def worker(
        worker_id: int,
        total: int,
        spectra_paths: ndarray,
        counter: mp.Value,
        queue: mp.Queue,
        background_dir: str = '',
        cut_off: list = None):
    """
    Worker used to parallelize spectrum_data

    Parameters
    ----------
    worker_id : integer
        Worker ID
    total : integer
        Total number of iterations
    spectra_paths : ndarray
        File path to the spectra
    counter : Value
        Number of spectra processed
    queue : Queue
        Multiprocessing queue to add spectra to
    background_dir : string, default = ''
        Path to the root directory where the background is located
    cut_off : list, default = [0.3, 10]
        Range of accepted data in keV
    """
    spectra = []

    for spectrum_path in spectra_paths:
        spectra.append(spectrum_data(spectrum_path, background_dir, cut_off=cut_off)[1:3])

        # Increase progress
        with counter.get_lock():
            counter.value += 1

        progress_bar(counter.value, total)

    queue.put([worker_id, np.stack(spectra, axis=0)])


def spectra_names(data_dir: str, blacklist: str = None, whitelist: str = None) -> ndarray:
    """
    Fetches the file names of all spectra up to the defined data amount

    Parameters
    ----------
    data_dir : string
        Directory of the spectra dataset
    blacklist : string, default = None
        Exclude all files with substring
    whitelist : string, default = None
        Require all files have the substring

    Returns
    -------
    ndarray
        Array of spectra file names
    """
    # Fetch all files within directory
    files = np.sort(np.array(os.listdir(data_dir)))

    # Remove all files that aren't whitelisted
    if whitelist:
        files = np.delete(files, np.char.find(files, whitelist) == -1)

    # Remove all files that are blacklisted
    if blacklist:
        files = np.delete(files, np.char.find(files, blacklist) != -1)

    return np.char.add(data_dir, files)


def preprocess():
    """
    Preprocess spectra and save to file
    """
    # Variables
    # data_dir = '../../../Documents/Nicer_Data/spectra/'
    # background_dir = data_dir
    # labels_path = '../data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    # processed_data_path = '../data/preprocessed_spectra.npy'
    data_dir = '../data/synth_spectra/'
    background_dir = '../'
    labels_path = None
    processed_data_path = '../data/synth_spectra.npy'

    # Constants
    processes = []
    initial_time = time()
    counter = mp.Value('i', 0)
    queue = mp.Queue()

    # Fetch spectra names & labels
    if labels_path:
        labels = np.loadtxt(labels_path, skiprows=6, dtype=str)
        spectra_files = np.char.add(data_dir, labels[:, 6])
    else:
        spectra_files = spectra_names(data_dir, blacklist='bkg')

    spectra_groups = np.array_split(spectra_files, mp.cpu_count())

    # Fetch spectra data
    for i, group in enumerate(spectra_groups):
        processes.append(mp.Process(
            target=worker,
            args=(
                i,
                spectra_files.size,
                group,
                counter,
                queue,
            ),
            kwargs={'background_dir': background_dir}
        ))

    # Start multiprocessing
    for process in processes:
        process.start()

    # Collect results
    output = np.array([queue.get() for _ in processes], dtype=object)

    # End multiprocessing
    for process in processes:
        process.join()

    # Save spectra data
    worker_order = np.argsort(output[:, 0])
    spectra = output[worker_order, 1]
    spectra = np.concatenate(spectra, axis=0)
    np.save(processed_data_path, spectra)
    print(f'\nProcessing time: {time() - initial_time:.1f}')

if __name__ == '__main__':
    preprocess()
