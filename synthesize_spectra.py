# TODO: Does passing orbit information help with accuracy?
import os
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


def existing_synth(new_synth_num: int, synth_dir: str, clear_synth: bool = True) -> int:
    """
    Either clears existing synthetic spectra or
    renames and counts the number of existing synthetic spectra

    Parameters
    ----------
    new_synth_num : int
        Number of new synthetic spectra to be created
    synth_dir : string
        Directory of synthetic spectra
    clear_synth : boolean, default = True
        If the existing data should be cleared or preserved

    Returns
    -------
    integer
        Number of existing synthetic spectra
    """
    if clear_synth:
        # Clear all existing synthetic spectra
        for synth in os.listdir(synth_dir):
            os.remove(synth_dir + synth)

        return 0

    # Find existing synthetic spectra names
    synth_names = np.sort(np.array(os.listdir(synth_dir)))
    synth_names = np.delete(synth_names, np.char.find(synth_names, '_bkg') != -1)

    # Rename each synthetic spectra so spectrum number is correct
    for i, synth in enumerate(synth_names):
        os.rename(
            synth_dir + synth,
            f'{synth_dir}synth_{i:0{int(np.log10(new_synth_num)) + 1}d}.fits'
        )
        os.rename(
            synth_dir + synth[:-5] + '_bkg.fits',
            f'{synth_dir}synth_{i:0{int(np.log10(new_synth_num)) + 1}d}_bkg.fits'
        )

    return int(len(os.listdir(synth_dir)) / 2)


def generate_synth(current_num: int,
                   synth_num: int,
                   synth_dir: str,
                   param_limits: list[dict],
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
    exposure : float, default = 1000
        Exposure time of synthetic spectra

    Returns
    -------
    ndarray
        Synthetic params
    """
    data_dir = '../../Documents/Nicer_Data/ethan/'
    labels_path = './data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    synth_params = []
    params = {1: 0, 2: 0, 3: 0, 4: 0, 5: 100, 6: 0, 7: 0}

    model = initialise_xspec()

    for i in range(synth_num):
        # Choose spectrum to base synthetic off
        spectrum = np.random.choice(np.loadtxt(labels_path, skiprows=6, usecols=6, dtype=str))

        # Retrieve background and response spectrum
        with fits.open(data_dir + spectrum) as f:
            spectrum_info = f[1].header
            background = data_dir + spectrum_info['BACKFILE']
            response = data_dir + spectrum_info['RESPFILE']
            aux = data_dir + spectrum_info['ANCRFILE']

        # Generate random model parameters for synthetic spectrum
        for param in param_limits:
            if param['log']:
                params[param['id']] = 10 ** np.random.uniform(
                    np.log10(param['low']), np.log10(param['high'])
                )
            else:
                params[param['id']] = np.random.uniform(param['low'], param['high'])

        model.setPars(params)
        synth_params.append(params)

        # Generate fake spectrum
        fake_setting = xspec.FakeitSettings(
            response=response,
            arf=aux,
            background=background,
            exposure=exposure,
            fileName=f'{synth_dir}synth_{i + current_num:0{int(np.log10(synth_num)) + 1}d}.fits'
        )

        xspec.AllData.fakeit(1, fake_setting)
        progress_bar(i, synth_num)

    return np.vstack(synth_params)


def clean_synth(current_num: int, synth_dir: str, params: np.ndarray):
    """
    Remove bad synthetic spectra

    Parameters
    ----------
    current_num : integer
        Number of existing synthetic spectra
    synth_dir : string
        Directory of synthetic spectra
    params : ndarray
        Synthetic params to clean
    """
    synth = []

    # Retrieve synthetic spectra names
    synth_names = np.sort(np.array(os.listdir(synth_dir)))
    synth_names = np.delete(
        synth_names,
        np.char.find(synth_names, '_bkg') != -1
    )[current_num:]

    synth_num = synth_names.size

    # Retrieve synthetic spectra data
    for i, spectrum in enumerate(synth_names):
        synth.append(spectrum_data(synth_dir + spectrum, '')[1])
        progress_bar(i, synth_num)

    synth = np.vstack(synth)

    # Find synthetic spectra that has integer overflow errors or S/N is too low
    max_counts = np.max(np.abs(synth), axis=1)
    overflow = np.argwhere(np.min(synth, axis=1) < 0)[:, 0]
    bad_indices = np.unique(np.append(overflow, np.argwhere(max_counts < 1e1)[:, 0]))

    # Remove bad synthetic spectra
    for bad_synth in bad_indices:
        os.remove(synth_dir + synth_names[bad_synth])
        os.remove(synth_dir + synth_names[bad_synth][:-5] + '_bkg.fits')

    # Save synthetic spectra data to file
    np.save('./data/synth_spectra', np.delete(synth, bad_indices, axis=0))
    np.save('./data/synth_spectra_params', np.delete(params, bad_indices))
    print(
        f'Spectra success: '
        f'{synth_num - bad_indices.size} / {synth_num} '
        f'{((synth_num - bad_indices.size) / synth_num) * 100:.1f}%'
    )


def main():
    """
    Main function for generating synthesized data
    """
    # Initialize variables
    clear_synth = True
    synth_num = int(1e2)
    synth_dir = './data/synth_spectra/'
    param_limits = [
        {'id': 1, 'low': 5e-3, 'high': 75, 'log': True},
        {'id': 2, 'low': 1.3, 'high': 4, 'log': False},
        {'id': 3, 'low': 1e-3, 'high': 1, 'log': True},
        {'id': 6, 'low': 2.5e-2, 'high': 4, 'log': True},
        {'id': 7, 'low': 1e-2, 'high': 1e10, 'log': True},
    ]

    current_num = existing_synth(synth_num, synth_dir, clear_synth)

    # Generate synthetic spectra
    params = generate_synth(current_num, synth_num, synth_dir, param_limits)

    # Remove bad synthetic spectra
    clean_synth(current_num, synth_dir, params)


if __name__ == '__main__':
    main()
