"""
Generates synthetic data using PyXspec
"""
import os
import argparse
from time import time

import yaml
import xspec
import numpy as np
from astropy.io import fits

from fspnet.utils.utils import progress_bar, file_names
from fspnet.utils.pyxspec_worker import initialize_pyxspec
from fspnet.utils.preprocessing_utils import correct_spectrum


def _delete_synth(synth_dir: str, synth_data: str):
    """
    Either clears existing synthetic spectra or
    renames and counts the number of existing synthetic spectra

    Parameters
    ----------
    synth_dir : string
        Directory of synthetic spectra
    synth_data : string
        File path to save synthetic spectra data
    """
    # Clear all existing synthetic spectra
    for synth in os.listdir(synth_dir):
        os.remove(synth_dir + synth)

    if os.path.exists(synth_data):
        os.remove(synth_data)
        os.remove(synth_data[:-4] + '_params.npy')


def _generate_synth(current_num: int,
                    exposure: float = 1e3,
                    config_path: str = './config.yaml') -> np.ndarray:
    """
    Generates synthetic spectra using PyXspec

    Parameters
    ----------
    current_num : integer
        Number of existing synthetic spectra
    exposure : float, default = 1000
        Exposure time of synthetic spectra
    config_path : string, default = './config.yaml'
        File path to the configuration file

    Returns
    -------
    ndarray
        Synthetic params
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = list(yaml.safe_load_all(file))[2]

    # Load variables
    synth_num = config['synthesize']['spectra-per-batch']
    load_frequency = config['synthesize']['fake-per-background']
    data_dir = config['data']['spectra-directory']
    labels_path = config['data']['spectra-names-file']
    synth_dir = config['output']['synthetic-directory']
    fixed_params = config['model']['fixed-parameters']
    param_limits = config['model']['parameter-limits']

    # Initialize the model
    model = initialize_pyxspec(
        config['model']['model-name'],
        custom_model=config['model']['custom-model-name'],
        model_dir=config['model']['model-directory'],
    )

    params = dict(zip(
        range(1, model.nParameters + 1),
        np.zeros(model.nParameters)
    )) | fixed_params
    synth_params = np.empty((0, len(params) - len(fixed_params)))

    # Get spectra file names to randomly sample
    if labels_path:
        spectra_names = np.loadtxt(labels_path, skiprows=6, usecols=6, dtype=str)
    else:
        spectra_names = file_names(data_dir, blacklist='bkg')

    for i in range(synth_num):
        # Retrieve background and response spectrum
        if i % load_frequency == 0:
            # Choose spectrum to base synthetic off
            spectrum = np.random.choice(spectra_names)
            xspec.AllData.clear()

            with fits.open(data_dir + spectrum) as f:
                spectrum_info = f[1].header
                background = data_dir + spectrum_info['BACKFILE']
                response = data_dir + spectrum_info['RESPFILE']
                aux = data_dir + spectrum_info['ANCRFILE']

            # Generate base fake settings
            fake_base = xspec.FakeitSettings(
                response=response,
                arf=aux,
                background=background,
                exposure=exposure,
            )

        # Generate random model parameters for synthetic spectrum
        for param in param_limits:
            if param['log']:
                params[param['id']] = 10 ** np.random.uniform(
                    np.log10(param['low']), np.log10(param['high'])
                )
            else:
                params[param['id']] = np.random.uniform(param['low'], param['high'])

        model.setPars(params)
        synth_params = np.vstack((
            synth_params,
            np.delete(
                np.fromiter(params.values(), dtype=float),
                np.fromiter(fixed_params.keys(), dtype=int) - 1
            )
        ))

        # Generate fake spectrum
        fake_setting = xspec.FakeitSettings(fake_base)
        fake_setting.fileName = \
            f'{synth_dir}synth_{i + current_num:0{int(np.log10(synth_num)) + 1}d}.fits'

        xspec.AllData.fakeit(1, fake_setting)
        progress_bar(i, synth_num)

    return synth_params


def _clean_synth(
        current_num: int,
        synth_dir: str,
        synth_data: str,
        params: np.ndarray,
        min_counts: float = 1) -> int:
    """
    Remove bad synthetic spectra

    Parameters
    ----------
    current_num : integer
        Number of existing synthetic spectra
    synth_dir : string
        Directory of synthetic spectra
    synth_data : string
        File path to save synthetic spectra data
    params : ndarray
        Synthetic params to clean
    min_counts : float
        Minimum number of counts to be accepted

    Returns
    -------
    integer
        Number of successful spectra
    """
    synth = []
    detectors = np.array(())

    # Retrieve synthetic spectra names
    synth_names = np.sort(np.array(os.listdir(synth_dir)))
    synth_names = np.delete(
        synth_names,
        np.char.find(synth_names, '_bkg') != -1
    )[current_num:]
    synth_num = synth_names.size

    # Retrieve synthetic spectra data
    for i, spectrum in enumerate(synth_names):
        detector, _, data = correct_spectrum(synth_dir + spectrum)
        synth.extend(data)
        detectors = np.append(detectors, detector)
        progress_bar(i, synth_num)

    synth = np.stack(synth, axis=0)

    # Find synthetic spectra that has integer overflow errors or S/N is too low
    max_counts = np.max(np.abs(synth[:, 0]), axis=-1) * detectors
    overflow = np.argwhere(np.min(synth[:, 0], axis=-1) < 0)[:, 0]
    bad_indices = np.unique(np.append(overflow, np.argwhere(max_counts < min_counts)))

    # Remove bad synthetic spectra
    for bad_synth in bad_indices:
        os.remove(synth_dir + synth_names[bad_synth])
        os.remove(synth_dir + synth_names[bad_synth][:-5] + '_bkg.fits')

    synth = np.delete(synth, bad_indices, axis=0)
    params = np.delete(params, bad_indices, axis=0)

    # Save synthetic spectra data to file
    if os.path.exists(synth_data):
        synth = np.vstack((np.load(synth_data), synth))
        params = np.vstack((np.load(synth_data.replace('_spectra.npy', '_params.npy')), params))

    np.save(synth_data, synth)
    np.save(synth_data.replace('_spectra.npy', '_params.npy'), params)

    return synth_num - bad_indices.size


def _fix_names(current_num: int, synth_num: int, synth_dir: str):
    """
    Fix names of synthetic spectra to be in order

    Parameters
    ----------
    current_num : integer
        Number of existing synthetic spectra
    synth_num : integer
        Number of new synthetic spectra to be created
    synth_dir : string
        Directory of synthetic spectra
    """
    # Find existing synthetic spectra names
    synth_names = file_names(synth_dir, blacklist='bkg')

    if synth_names.size > synth_num:
        synth_num = synth_names.size

    synth_names = synth_names[current_num:]

    # Rename each synthetic spectra so spectrum number is correct
    for i, synth in enumerate(synth_names):
        new_synth = f'{synth_dir}synth_{i + current_num:0{int(np.log10(synth_num)) + 1}d}.fits'
        new_background = new_synth[:-5] + '_bkg.fits'

        with fits.open(synth_dir + synth, mode='update') as f:
            f[1].header['BACKFILE'] = new_background
            f.flush()

        os.rename(synth_dir + synth, new_synth)
        os.rename(synth_dir + synth[:-5] + '_bkg.fits', new_background)

        progress_bar(i, synth_names.size)


def main(config_path: str = './config.yaml'):
    """
    Main function for generating synthesized data

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        File path to the configuration file
    """
    # Raise directory otherwise file path length is too long for fits file, not ideal solution
    root_dir = os.getcwd()
    os.chdir('../')

    # If run by command line, optional argument can be used
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default=config_path)
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, 'r', encoding='utf-8') as file:
        config = list(yaml.safe_load_all(file))[2]

    # Variables
    total_synth_num = config['synthesize']['synthetic-number']
    synth_num = config['synthesize']['spectra-per-batch']
    synth_dir = config['output']['synthetic-directory']
    synth_data = config['output']['synthetic-data']

    # Constant
    success = int(len(os.listdir(synth_dir)) / 2)

    if not os.path.exists(synth_dir):
        os.makedirs(synth_dir)

    # Manage existing synthetic spectra
    if config['synthesize']['clear-spectra']:
        _delete_synth(synth_dir, synth_data)

    # TODO: Multiprocessing
    while success < total_synth_num:
        initial_time = time()
        current_num = int(len(os.listdir(synth_dir)) / 2)

        # Generate synthetic spectra
        print('\nGenerating synthetic spectra...')
        params = _generate_synth(current_num, config_path=config_path)

        # Remove bad synthetic spectra
        print('\nRemoving bad synthetic spectra...')
        success += _clean_synth(current_num, synth_dir, synth_data, params)

        if success != current_num:
            # Rename synthetic spectra in order
            print('\nRenaming synthetic spectra...')
            _fix_names(current_num, total_synth_num, synth_dir)

        print(f'\nCount: {success} / {total_synth_num} {100 * success / total_synth_num} %'
              f'\tSuccess: {success - current_num} / {synth_num} '
              f'{100 * (success - current_num) / synth_num:.1f} %'
              f'\tTime: {time() - initial_time:.2f} s')

    os.chdir(root_dir)


if __name__ == '__main__':
    main()
