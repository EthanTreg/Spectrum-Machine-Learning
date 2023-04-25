"""
Misc functions used elsewhere
"""
import os
from argparse import ArgumentParser

import yaml
import torch
import numpy as np


def progress_bar(i: int, total: int, text: str = ''):
    """
    Terminal progress bar

    Parameters
    ----------
    i : integer
        Current progress
    total : integer
        Completion number
    text : string, default = '
        Optional text to place at the end of the progress bar
    """
    length = 50
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = '█' * filled + '-' * (length - filled)
    print(f'\rProgress: |{bar_fill}| {int(percent)}%\t{text}\t', end='')

    if i == total:
        print()


def file_names(data_dir: str, blacklist: list[str] = None, whitelist: str = None) -> np.ndarray:
    """
    Fetches the file names of all spectra that are in the whitelist, if not None,
    or not on the blacklist, if not None

    Parameters
    ----------
    data_dir : string
        Directory of the spectra dataset
    blacklist : list[string], default = None
        Exclude all files with substrings
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
    for substring in blacklist:
        files = np.delete(files, np.char.find(files, substring) != -1)

    return files


def get_device() -> tuple[dict, torch.device]:
    """
    Gets the device for PyTorch to use

    Returns
    -------
    tuple[dictionary, device]
        Arguments for the PyTorch DataLoader to use when loading data into memory and PyTorch device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    return kwargs, device


def open_config(key: str, config_path: str, parser: ArgumentParser = None) -> tuple[str, dict]:
    """
    Opens the configuration file from either the provided path or through command line argument

    Parameters
    ----------
    key : string
        Key of the configuration file
    config_path : string
        Default path to the configuration file
    parser : ArgumentParser, default = None
        Parser if arguments other than config path are required

    Returns
    -------
    tuple[string, dictionary]
        Configuration path and configuration file dictionary
    """
    if not parser:
        parser = ArgumentParser()

    parser.add_argument(
        '--config_path',
        default=config_path,
        help='Path to the configuration file',
        required=False,
    )
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, 'rb') as file:
        config = yaml.safe_load(file)[key]

    return config_path, config
