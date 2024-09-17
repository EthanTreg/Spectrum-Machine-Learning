"""
Worker for generating synthetic spectra using PyXspec
"""
import os
import re
import pickle
from time import time

import xspec
import numpy as np
import pandas as pd
from numpy import ndarray
from astropy.io import fits
from netloader.utils.utils import progress_bar

from fspnet.utils.preprocessing import spectrum_data
from fspnet.utils.workers import initialize_worker, initialize_pyxspec


def _uniform_sample(
        low: float,
        high: float,
        log: bool = False,
        bias: bool = False,
        data: ndarray = None) -> float:
    """
    Generates a random value from a uniform distribution in either linear or log space
    and can be biased to force distribution to uniform

    Parameters
    ----------
    low : float
        Minimum value
    high : float
        Maximum value
    log : boolean, default = False
        If log space should be used
    bias : boolean, default = False
        If the distribution should be biased towards a uniform distribution,
        requires existing values
    data : ndarray, default = None
        If bias is True, data will be used to bias the distribution towards under-sampled values

    Returns
    -------
    float
        Sampled value
    """
    bin_num = 1000

    if log:
        low = np.log10(low)
        high = np.log10(high)

    if data is not None and log:
        data = np.log10(data)

    # If biased selection should be used, requires minimum number of data
    if bias and data is not None and data.size > bin_num:
        # Bias probabilities from current distribution
        probs, bins = np.histogram(
            data,
            bin_num,
            range=(low, high),
        )
        probs = np.max(probs) - probs

        # Sample random options from uniform distribution,
        # assign probabilities and randomly choose one
        options = np.random.uniform(low, high, size=bin_num)
        option_probs = probs[np.digitize(options, bins) - 1]
        option_probs = option_probs + np.min(
            option_probs,
            where=option_probs > 0,
            initial=np.max(option_probs),
        )
        option_probs = option_probs / np.linalg.norm(option_probs, ord=1)
        value = np.random.choice(options, p=option_probs)
    else:
        value = np.random.uniform(low, high)

    if log:
        return 10 ** value

    return value


def _check_synth(min_counts: int = 1e4) -> tuple[ndarray, ndarray] | None:
    """
    Checks if the generated spectrum is good, if it is, correct, bin and normalise it

    Parameters
    ----------
    min_counts : integer, default = 10
        Minimum summed background subtracted counts for the spectrum to be valid

    Returns
    -------
    tuple[ndarray, ndarray] | None
        Spectrum and uncertainty if it is good; otherwise, None
    """
    # Fetch data from PyXspec
    background = pd.DataFrame(xspec.AllData(1).background.values, columns=['RATE'])
    spectrum = pd.DataFrame(np.stack((
        np.arange(background['RATE'].size),
        xspec.AllData(1).values,
    ), axis=1), columns=['CHANNEL', 'RATE'])
    exposure = float(xspec.AllData(1).exposure)
    detectors = int(re.search(r'_d(\d+)', xspec.AllData(1).response.rmf).group(1))

    # Correct spectrum
    _, counts, uncertainty, energy_bin = spectrum_data(
        detectors,
        exposure,
        exposure,
        spectrum,
        background,
    )

    # Don't save if spectrum is bad
    if np.sum(counts * energy_bin) * exposure * detectors < min_counts or \
            np.min(spectrum['RATE']) < 0:
        return None

    return counts, uncertainty


def _generate_synth(
        config: dict,
        model: xspec.Model,
        fake_base: xspec.FakeitSettings,
        synth_params: list[ndarray] = None) -> ndarray:
    """
    Generates synthetic spectra using PyXspec

    Parameters
    ----------
    config : dict
        Configuration settings
    model : Model
        PyXspec model to base the synthetic spectra off
    fake_base : FakeitSettings
        Settings for PyXspec to base the synthetic spectra off
    synth_params : list[ndarray], default = None
        If flat-distribution-bias in config.yaml is True,
        then parameters will be biased towards under-sampled values from synth_params

    Returns
    -------
    ndarray
        Synthetic params
    """
    # Constants
    bias = config['synthesize']['flat-distribution-bias']
    params_num = config['model']['parameter-number']
    log_params = config['model']['log-parameters']
    param_limits = config['model']['parameter-limits']
    params = config['model']['fixed-parameters'].copy()
    fixed_num = len(params)

    # Get free parameter indices
    param_indices = np.arange(params_num + fixed_num) + 1
    param_indices = np.setdiff1d(param_indices, list(params)).tolist()

    if synth_params:
        synth_params = np.swapaxes(synth_params, 1, 0)
    else:
        synth_params = [None] * len(param_limits)

    # Generate random model parameters for synthetic spectrum
    for i, (
            param,
            synth_param,
            param_idx,
    ) in enumerate(zip(param_limits, synth_params, param_indices)):
        params[param_idx] = _uniform_sample(
            param['low'],
            param['high'],
            log=i in log_params,
            bias=bias,
            data=synth_param,
        )

    # Generate fake spectrum
    model.setPars(params)
    xspec.AllData.fakeit(nSpectra=1, settings=fake_base, applyStats=True, noWrite=True)

    return np.fromiter(params.values(), dtype=float)[fixed_num:]


def _update_base_spectrum(
        data_dir: str,
        spectra_names: ndarray,
        exposure: float = 1e3) -> xspec.FakeitSettings:
    """
    Updates the spectrum that the synthetic spectra are based off

    Parameters
    ----------
    data_dir : string
        Directory of spectra
    spectra_names : ndarray
        Names of possible spectra
    exposure : float, default = 1e3
        Exposure setting for Fakeit settings

    Returns
    -------
    FakeitSettings
        Settings for PyXspec to base the synthetic spectra off
    """
    # Choose spectrum to base synthetic off
    spectrum = np.random.choice(spectra_names)
    xspec.AllData.clear()

    # Load random spectrum
    with fits.open(data_dir + spectrum) as file:
        spectrum_info = file[1].header

    # Generate fake settings
    fake_base = xspec.FakeitSettings(
        response=data_dir + spectrum_info['RESPFILE'],
        arf=data_dir + spectrum_info['ANCRFILE'],
        background=data_dir + spectrum_info['BACKFILE'],
        exposure=exposure,
    )

    return fake_base


def worker():
    """
    Worker for multiprocessing to generate synthetic spectra
    """
    rank, cpus, worker_dir, data = initialize_worker()

    # Constants
    bad_num = bad_num_total = 0
    synth_num_total = data['synth_num_total'] // cpus
    initial_time = time()
    config = data['config']
    save_frequency = config['synthesize']['spectra-per-batch']
    load_frequency = config['synthesize']['spectra-per-background']
    data_dir = config['data']['spectra-directory']
    synth_data = {
        'params': [],
        'spectra': [],
        'uncertainties': [],
        'info': [],
    }

    # Initialize the model
    model = initialize_pyxspec(
        config['model']['model-name'],
        custom_model=config['model']['custom-model-name'],
        model_dir=config['model']['model-directory'],
    )

    if rank < data['synth_num_total'] % cpus:
        synth_num_total += 1

    # If previous worker data exists, load it
    if os.path.exists(f'{worker_dir}{rank}.pkl'):
        with open(f'{worker_dir}{rank}.pkl', 'rb') as file:
            synth_data = pickle.load(file)

            synth_num_total += len(synth_data['spectra'])

    fake_base = _update_base_spectrum(data_dir, data['spectra_names'])

    # Keep generating synthetic spectra until total number for worker is reached
    while len(synth_data['spectra']) < synth_num_total:
        # Update base spectrum
        if len(synth_data['spectra']) % load_frequency == 0:
            fake_base = _update_base_spectrum(data_dir, data['spectra_names'])

        # Save synthetic spectra generated by the worker
        if (
            len(synth_data['spectra']) % save_frequency == 0
            and len(synth_data['spectra']) != 0
            and bad_num == 0
        ):
            with open(f'{worker_dir}worker_{rank}.pickle', 'wb') as file:
                pickle.dump(synth_data, file)

        # Generate synthetic spectrum & check if it is good
        spectrum_params = _generate_synth(
            config,
            model,
            fake_base,
            synth_params=synth_data['params'],
        )
        check_return = _check_synth()

        # If the spectrum is good, save the parameters
        if check_return:
            bad_num_total += bad_num
            synth_data['spectra'].append(check_return[0])
            synth_data['uncertainties'].append(check_return[1])
            synth_data['params'].append(spectrum_params)
            synth_data['info'].append(fake_base)

            if cpus == 1:
                success = len(synth_data['spectra']) * 100 / \
                          (len(synth_data['spectra']) + bad_num_total)
                eta = (time() - initial_time) * (synth_num_total / len(synth_data['spectra']) - 1)

                progress_bar(
                    len(synth_data['spectra']),
                    synth_num_total + 1,
                    text=f'ETA: {eta:.2f} s'
                         f"\tSuccess: {success:.1f} %"
                         f"\tGenerated: {len(synth_data['spectra'])} / {synth_num_total}"
                )
            else:
                print(f'fail_num={bad_num}')
                print('update')

            bad_num = 0
        else:
            bad_num += 1

    # Save final data
    with open(f'{worker_dir}worker_{rank}.pickle', 'wb') as file:
        pickle.dump(synth_data, file)


if __name__ == '__main__':
    worker()
