# TODO: Does passing orbit information help with accuracy?
import os
from time import time

import xspec
import numpy as np
from astropy.io import fits

from data_preprocessing import spectrum_data


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


def initialise_xspec(model_type: str = 'tbabs(simplcutx(ezdiskbb))') -> xspec.Model:
    """
    Initialise Xspec and create model

    Parameters
    ----------
    model_type : string, default = tbabs(simplcutx(ezdiskbb))
        Model type to load into Xspec

    Returns
    -------
    Model
        Xspec model
    """
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0

    # Load model
    xspec.AllModels.lmod('simplcutx', dirPath='../../Documents/Xspec_Models/simplcutx/')
    model = xspec.Model(model_type)
    xspec.AllModels.setEnergies('0.002 500 1000 log')

    return model


def delete_synth(synth_dir: str, synth_data: str):
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


def generate_synth(current_num: int,
                   synth_num: int,
                   synth_dir: str,
                   param_limits: list[dict],
                   model: xspec.Model,
                   exposure: float = 1e3) -> np.ndarray:
    """
    Generates synthetic spectra using PyXspec

    Parameters
    ----------
    current_num : integer
        Number of existing synthetic spectra
    synth_num : integer
        Number of synthetic spectra to generate
    synth_dir : string
        Directory of synthetic spectra
    param_limits : list[dictionary]
        Parameter ID, limits and if logarithmic space should be used
    model : Model
        Xspec model to be used
    exposure : float, default = 1000
        Exposure time of synthetic spectra

    Returns
    -------
    ndarray
        Synthetic params
    """
    load_frequency = 100
    data_dir = '../../Documents/Nicer_Data/ethan/'
    labels_path = './data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    fixed_params = {4: 0, 5: 100}
    params = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0} | fixed_params
    synth_params = np.empty((0, len(params) - len(fixed_params)))

    for i in range(synth_num):
        # Retrieve background and response spectrum
        if i % load_frequency == 0:
            # Choose spectrum to base synthetic off
            spectrum = np.random.choice(np.loadtxt(labels_path, skiprows=6, usecols=6, dtype=str))
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
                exposure=exposure
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


def clean_synth(
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
        _, synth_spectrum, detector = spectrum_data(synth_dir + spectrum)
        synth.append(synth_spectrum)
        detectors = np.append(detectors, detector)
        progress_bar(i, synth_num)

    synth = np.vstack(synth)

    # Find synthetic spectra that has integer overflow errors or S/N is too low
    max_counts = np.max(np.abs(synth), axis=1) * detectors
    overflow = np.argwhere(np.min(synth, axis=1) < 0)[:, 0]
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
        params = np.vstack((np.load(synth_data[:-4] + '_params.npy'), params))

    np.save(synth_data, synth)
    np.save(synth_data[:-4] + '_params.npy', params)

    return synth_num - bad_indices.size


def fix_names(current_num: int, synth_num: int, synth_dir: str):
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
    synth_names = np.sort(np.array(os.listdir(synth_dir)))
    synth_names = np.delete(synth_names, np.char.find(synth_names, '_bkg') != -1)

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


def main():
    """
    Main function for generating synthesized data
    """
    # Variables
    clear_synth = False
    batches = 1000
    total_synth_num = 1e5
    synth_dir = './data/synth_spectra/'
    synth_data = './data/synth_spectra.npy'
    param_limits = [
        {'id': 1, 'low': 5e-3, 'high': 75, 'log': True},
        {'id': 2, 'low': 1.3, 'high': 4, 'log': False},
        {'id': 3, 'low': 1e-3, 'high': 1, 'log': True},
        {'id': 6, 'low': 2.5e-2, 'high': 4, 'log': True},
        {'id': 7, 'low': 1e-2, 'high': 1e10, 'log': True},
    ]

    # Constant
    success = 0

    model = initialise_xspec()

    if not os.path.exists(synth_dir):
        os.makedirs(synth_dir)

    # Manage existing synthetic spectra
    if clear_synth:
        delete_synth(synth_dir, synth_data)

    # TODO: Multiprocessing
    for i in range(batches):
        t_initial = time()
        synth_num = int(total_synth_num / batches)
        current_num = int(len(os.listdir(synth_dir)) / 2)

        if i == batches - 1:
            synth_num += total_synth_num % batches

        # Generate synthetic spectra
        print('\nGenerating synthetic spectra...')
        params = generate_synth(current_num, synth_num, synth_dir, param_limits, model)

        # Remove bad synthetic spectra
        print('\nRemoving bad synthetic spectra...')
        success += clean_synth(current_num, synth_dir, synth_data, params)

        if success != current_num:
            # Rename synthetic spectra in order
            print('\nRenaming synthetic spectra...')
            fix_names(current_num, total_synth_num, synth_dir)

        print(f'\nBatch {i + 1} / {batches} {((i + 1) / batches) * 100:.1f} %'
              f'\tSuccess: {success} / {total_synth_num * i / batches}'
              f'{100 * success * batches / (total_synth_num * i):.1f} %'
              f'\tTime: {time() - t_initial:.2f} s')


if __name__ == '__main__':
    main()
