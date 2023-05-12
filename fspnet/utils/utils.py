"""
Misc functions used elsewhere
"""
import os
from argparse import ArgumentParser

import yaml
import torch
import numpy as np


def _interactive_check() -> bool:
    """
    Checks if the launch environment is interactive or not

    Returns
    -------
    boolean
        If environment is interactive
    """
    if os.getenv('PYCHARM_HOSTED'):
        return True

    try:
        if get_ipython().__class__.__name__:
            return True
    except NameError:
        return False

    return False


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


def closest_factors(num: int) -> tuple[int, int]:
    """
    Finds the closest factors for a given integer

    Parameters
    ----------
    num : integer
        Number to find the closest factors for

    Returns
    -------
    tuple[integer, integer]
        Two closest factors
    """
    factor = int(np.floor(np.sqrt(num)))

    while num % factor:
        factor -= 1

    return factor, int(num / factor)


def subplot_grid(num: int) -> np.ndarray:
    """
    Calculates the most square grid for a given input for mosaic subplots

    Parameters
    ----------
    num : integer
        Total number to split into a mosaic grid

    Returns
    -------
    ndarray
        2D array of indices with relative width for mosaic subplot
    """
    # Constants
    grid = (int(np.sqrt(num)), int(np.ceil(np.sqrt(num))))
    subplot_layout = np.arange(num)
    diff_row = np.abs(num - np.prod(grid))

    # If number is not divisible into a square-ish grid,
    # then the total number will be unevenly divided across the rows
    if diff_row and diff_row != grid[0]:
        shift_num = diff_row * (grid[1] + np.sign(num - np.prod(grid)))

        # Layout of index and repeated values to correspond to the width of the index
        subplot_layout = np.vstack((
            np.repeat(
                subplot_layout[:-shift_num],
                int(shift_num / diff_row)
            ).reshape(grid[0] - diff_row, -1),
            np.repeat(
                subplot_layout[-shift_num:],
                int((num - shift_num) / (grid[0] - diff_row))
            ).reshape(diff_row, -1),
        ))
    # If a close to square grid is found
    elif diff_row:
        subplot_layout = subplot_layout.reshape(grid[0], grid[1] + 1)
    # If grid is square
    else:
        subplot_layout = subplot_layout.reshape(*grid)

    return subplot_layout


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
    if not _interactive_check():
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
