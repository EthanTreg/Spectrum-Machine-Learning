"""
Generates synthetic data using PyXspec
"""
import os
import pickle

import numpy as np

from fspnet.utils.data import delete_data
from fspnet.utils.utils import file_names, open_config
from fspnet.utils.multiprocessing import check_cpus, mpi_multiprocessing


def main(config_path: str = '../config.yaml'):
    """
    Main function for generating synthesized data

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        File path to the configuration file
    """
    # If run by command line, optional argument can be used
    config_path, config = open_config('synthesize-spectra', config_path)

    # Variables
    cpus = config['synthesize']['cpus']
    names_path = config['data']['names-path']
    synth_path = config['output']['synthetic-path']
    worker_dir = config['output']['worker-directory']

    # Constants
    blacklist = ['bkg', '.bg', '.rmf', '.arf']
    synth_data = {
        'params': [],
        'spectra': [],
        'uncertainties': [],
        'info': [],
    }
    worker_data = {
        'synth_num_total': config['synthesize']['synthetic-number'],
        'config': config,
    }

    cpus = check_cpus(cpus)

    if not os.path.exists(worker_dir):
        os.mkdir(worker_dir)

    # Manage existing synthetic spectra
    delete_data(files=[f'{worker_dir}worker_data.pickle'])

    if config['synthesize']['clear-spectra']:
        delete_data(directory=worker_dir, files=[synth_path])

    # Load previously saved synthetic spectra & worker spectra
    if os.path.exists(synth_path):
        with open(synth_path, 'rb') as file:
            synth_data = pickle.load(file)

        worker_data['synth_num_total'] -= len(synth_data['spectra'])

    if worker_data['synth_num_total'] <= 0:
        print('Synthetic spectra already finished')
        return

    for worker_file in os.listdir(worker_dir):
        with open(worker_dir + worker_file, 'rb') as file:
            worker_data['synth_num_total'] -= len(pickle.load(file)['spectra'])

    # Get spectra file names to randomly sample
    if '.pickle' in names_path:
        with open(names_path, 'rb') as file:
            worker_data['spectra_names'] = pickle.load(file)['names']
    elif '.npy' in names_path:
        worker_data['spectra_names'] = np.load(names_path)
    else:
        worker_data['spectra_names'] = file_names(
            config['data']['spectra-directory'],
            blacklist=blacklist,
        )

    with open(f'{worker_dir}worker_data.pickle', 'wb') as file:
        pickle.dump(worker_data, file)

    mpi_multiprocessing(
        cpus,
        worker_data['synth_num_total'],
        f'fspnet.utils.synthesize_worker {worker_dir}',
    )

    print('\nGeneration complete, merging worker data...')

    delete_data(files=[f'{worker_dir}worker_data.pickle'])

    # Load worker data and if previous data exists, stack together, then save data
    for worker_file in os.listdir(worker_dir):
        with open(worker_dir + worker_file, 'rb') as file:
            worker_data = pickle.load(file)

        synth_data['params'].extend(worker_data['params'])
        synth_data['spectra'].extend(worker_data['spectra'])
        synth_data['uncertainties'].extend(worker_data['uncertainties'])
        synth_data['info'].extend(worker_data['info'])

        os.remove(worker_dir + worker_file)

    with open(synth_path, 'wb') as file:
        pickle.dump(synth_data, file)


if __name__ == '__main__':
    main()
