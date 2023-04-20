"""
Normalises and bins spectra and can perform background augmentation and synthetic splicing
"""
import pickle
import multiprocessing as mp
from time import time

import numpy as np
from numpy import ndarray

from fspnet.utils.preprocessing import correct_spectrum_file
from fspnet.utils.utils import progress_bar, file_names, open_config
from fspnet.utils.multiprocessing import check_cpus


def _worker(
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
        spectra.append(correct_spectrum_file(
            spectrum_path,
            background_dir,
            cut_off=cut_off,
        )[-2:])

        # Increase progress
        with counter.get_lock():
            counter.value += 1

        progress_bar(counter.value, total)

    queue.put([worker_id, np.stack(spectra, axis=0)])


def preprocess(config_path: str = '../config.yaml'):
    """
    Preprocess spectra from fits files and save to numpy file

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        File path to the configuration file
    """
    # If run by command line, optional argument can be used
    _, config = open_config('data-preprocessing', config_path)

    # Initialize variables
    cpus = config['preprocessing']['cpus']
    data_dir = config['data']['spectra-directory']
    background_dir = config['data']['background-directory']
    names_path = config['data']['names-path']
    processed_path = config['output']['processed-path']

    # Constants
    blacklist = ['bkg', '.bg', '.rmf', '.arf']
    initial_time = time()
    processes = []
    counter = mp.Value('i', 0)
    queue = mp.Queue()

    cpus = check_cpus(cpus)

    # Fetch spectra names and parameters
    if names_path and '.pickle' in names_path:
        with open(names_path, 'rb') as file:
            spectra_files = pickle.load(file)['names']

        spectra_files = np.char.add(data_dir, spectra_files)
    elif '.npy' in names_path:
        spectra_files = np.load(names_path)
    else:
        spectra_files = np.char.add(data_dir, file_names(data_dir, blacklist=blacklist))

    # Create worker jobs
    spectra_groups = np.array_split(spectra_files, cpus)

    # Correct & normalize spectra
    for i, group in enumerate(spectra_groups):
        processes.append(mp.Process(
            target=_worker,
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
    np.save(processed_path, spectra)
    print(f'\nProcessing time: {time() - initial_time:.1f}')


if __name__ == '__main__':
    preprocess()
