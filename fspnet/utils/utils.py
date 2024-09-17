"""
Misc functions used elsewhere
"""
import os
from argparse import ArgumentParser

import yaml
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


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


def legend_marker(colours: list[str], labels: list[str], markers: list[str] = None) -> ndarray:
    """
    Creates markers for a legend

    Parameters
    ----------
    colours : list[string]
        Colours for the legend
    labels : list[string]
        Labels for the legend
    markers : list[string], default = None
        Markers for the legend

    Returns
    -------
    ndarray
        Legend labels
    """
    legend_labels = []

    if markers is None:
        markers = [None] * len(colours)

    for colour, label, marker in zip(colours, labels, markers):
        legend_labels.append([plt.gca().scatter([], [], color=colour, marker=marker), label])

    return np.array(legend_labels).swapaxes(0, 1)


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


def name_sort(
        names: list[ndarray],
        data: list[ndarray],
        shuffle=True) -> tuple[list[ndarray], list[ndarray]]:
    """
    Sorts names and data so that two arrays contain the same names

    Parameters
    ----------
    names : list[ndarray, ndarray]
        Name arrays to sort
    data : list[ndarray, ndarray]
        Data arrays to sort from corresponding name arrays
    shuffle : boolean, default = True
        If name and data arrays should be shuffled

    Returns
    -------
    tuple[list[ndarray, ndarray], list[ndarray, ndarray]]
        Sorted name and data arrays
    """
    # Sort for longest dataset first
    sort_idx = np.argsort([datum.shape[0] for datum in data])[::-1]
    data = [data[i] for i in sort_idx]
    names = [names[i] for i in sort_idx]

    # Sort target spectra by name
    name_sort_idx = np.argsort(names[0])
    names[0] = names[0][name_sort_idx]
    data[0] = data[0][name_sort_idx]

    # Filter target parameters using spectra that was predicted and log parameters
    target_idx = np.searchsorted(names[0], names[1])
    names[0] = names[0][target_idx]
    data[0] = data[0][target_idx]

    # Shuffle params
    if shuffle:
        shuffle_idx = np.random.permutation(data[0].shape[0])
        names[0] = names[0][shuffle_idx]
        names[1] = names[1][shuffle_idx]
        data[0] = data[0][shuffle_idx]
        data[1] = data[1][shuffle_idx]

    data = [data[i] for i in sort_idx]
    names = [names[i] for i in sort_idx]

    return names, data


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
