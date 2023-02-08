"""
Normalises and bins spectra and can perform background augmentation and synthetic splicing
"""
import argparse
from time import time
import multiprocessing as mp

import yaml
import numpy as np
from numpy import ndarray

from fspnet.utils.utils import progress_bar, file_names
from fspnet.utils.preprocessing_utils import correct_spectrum


def _worker(
        worker_id: int,
        total: int,
        spectra_paths: ndarray,
        counter: mp.Value,
        queue: mp.Queue,
        aug_count: int = 0,
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
    aug_count : integer, default = 0
        Number of augmentations to perform
    background_dir : string, default = ''
        Path to the root directory where the background is located
    cut_off : list, default = [0.3, 10]
        Range of accepted data in keV
    """
    spectra = []

    for spectrum_path in spectra_paths:
        spectra.extend(correct_spectrum(
            spectrum_path,
            background_dir,
            aug_count=aug_count,
            cut_off=cut_off,
        )[-1])

        # Increase progress
        with counter.get_lock():
            counter.value += 1

        progress_bar(counter.value, total)

    queue.put([worker_id, np.stack(spectra, axis=0)])


def preprocess(config_path: str = '../config.yaml'):
    """
    Preprocess spectra and save to file

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        File path to the configuration file
    """
    # If run by command line, optional argument can be used
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default=config_path)
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, 'r', encoding='utf-8') as file:
        config = list(yaml.safe_load_all(file))[1]

    # Initialize variables
    aug_count = config['augmentation']['augmentation_number']
    data_dir = config['data']['spectra-directory']
    background_dir = config['data']['background-directory']
    params_path = config['data']['parameters-path']
    processed_path = config['output']['processed-path']

    # Constants
    params = None
    cpus = mp.cpu_count()
    initial_time = time()
    processes = []
    counter = mp.Value('i', 0)
    queue = mp.Queue()

    # Fetch spectra names & labels
    if '.npy' in params_path:
        spectra_files = file_names(data_dir, blacklist='bkg')
        params = np.load(params_path)
    elif params_path:
        labels = np.loadtxt(params_path, skiprows=6, dtype=str)
        spectra_files = np.char.add(data_dir, labels[:, 6])
        params = labels[:, 9:].astype(float)
    else:
        spectra_files = file_names(data_dir, blacklist='bkg')

    # Duplicate parameters if provided and augmentation is used
    if aug_count and params:
        params = np.repeat(params, aug_count + 1, axis=0)
        np.save(processed_path.replace('spectra', 'params'), params)

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
            kwargs={'aug_count': aug_count, 'background_dir': background_dir}
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
