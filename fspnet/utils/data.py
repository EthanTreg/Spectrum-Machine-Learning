"""
Loads data and creates data loaders for network training
"""
import os
import pickle

import torch
import numpy as np
from numpy import ndarray
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from fspnet.utils.utils import get_device


class SpectrumDataset(Dataset):
    """
    A dataset object containing spectrum data for PyTorch training

    Attributes
    ----------
    log_params : list
        Index of each free parameter in logarithmic space
    transform : list[tuple[ndarray, ndarray]]
        Min and max spectral range and mean & standard deviation of parameters
    names : ndarray
        Names of each spectrum
    spectra : tensor
        Spectra dataset
    params : tensor
        Parameters for each spectra if supervised
    indices : ndarray, default = None
        Data indices for random training & validation datasets

    Methods
    -------
    downscaler(downscales)
        Downscales input spectra
    """
    def __init__(
            self,
            data_file: str,
            log_params: list[int],
            transform: tuple[tuple[float, float], tuple[ndarray, ndarray]] = None):
        """
        Parameters
        ----------
        data_file : string
            Path to the file with the spectra dataset
        log_params : list[int]
            Index of each free parameter in logarithmic space
        transform : tuple[tuple[float, float], tuple[ndarray, ndarray]], default = [None, None]
            Min and max spectral range and mean & standard deviation of parameters
            used for transformation after log
        """
        self.indices = None
        self.log_params = log_params

        with open(data_file, 'rb') as file:
            data = pickle.load(file)

        self.spectra = np.array(data['spectra'])

        # Get spectra names if available
        if 'names' in data:
            self.names = data['names']
        else:
            self.names = np.arange(self.spectra.shape[0])

        if transform:
            self.transform = transform
        else:
            self.transform = [None, None]

        # Set negative values equal to minimum positive value
        if np.min(self.spectra) <= 0:
            self.spectra = _min_clamp(self.spectra, axis=1)

        # Transform spectra & uncertainty
        self.spectra = np.log10(self.spectra)
        self.spectra, self.transform[0] = data_normalization(
            self.spectra,
            mean=False,
            transform=self.transform[0],
        )

        # Make sure spectra & uncertainty length is even
        self.spectra = _even_length(torch.from_numpy(self.spectra).float())

        # If no parameters are provided
        if 'params' not in data:
            self.params = np.empty(self.spectra.size(0))
            return

        self.params = np.array(data['params'])

        # Transform parameters
        self.params[:, log_params] = np.log10(self.params[:, log_params])
        self.params, self.transform[1] = data_normalization(
            self.params,
            axis=0,
            transform=self.transform[1],
        )
        self.params = torch.from_numpy(self.params).float()

    def __len__(self) -> int:
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str | int]:
        """
        Gets the training data for a given index

        Parameters
        ----------
        idx : integer
            Index of the target spectrum

        Returns
        -------
        tuple[Tensor, Tensor, string | integer]
            Spectrum data, target parameters, and spectrum name/number
        """
        return self.spectra[idx], self.params[idx], self.names[idx]

    def downscaler(self, downscales: int):
        """
        Downscales input spectra

        Parameters
        ----------
        downscales : integer
            Number of times to downscale
        """
        avgpool = nn.AvgPool1d(kernel_size=2)

        self.spectra = self.spectra.unsqueeze(dim=1)

        for _ in range(downscales):
            self.spectra = avgpool(self.spectra)

        self.spectra = self.spectra.squeeze(dim=1)


def _even_length(x: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor of even length in the last
    dimension by merging the last two values

    Parameters
    ----------
    x : Tensor
        Input data

    Returns
    -------
    Tensor
        Output data with even length
    """
    if x.size(-1) % 2 != 0:
        x = torch.cat((
            x[..., :-2],
            torch.mean(x[..., -2:], dim=-1, keepdim=True)
        ), dim=-1)

    return x


def _min_clamp(data: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Clamps all values <= 0 to the minimum non-zero positive value

    Parameters
    ----------
    data : ndarray
        Input data to be clamped
    axis : integer, default = None
        Which axis to take the minimum over, if None, take the global minimum

    Returns
    -------
    ndarray
        Clamped data
    """
    min_count = np.min(
        data,
        where=data > 0,
        initial=np.max(data),
        axis=axis,
    )

    if axis:
        min_count = np.expand_dims(min_count, axis=axis)

    return np.maximum(data, min_count)


def load_data(data_path: str, columns: list[int] | range = None) -> ndarray:
    """
    Loads data from either a csv, pickle or numpy file

    Parameters
    ----------
    data_path : string
        Path to the data
    columns : list[integer] | range, default = None
        If data is a csv file, then columns can be used to specify which columns to load

    Returns
    -------
    ndarray
        Data loaded from the file
    """
    if '.csv' in data_path:
        data = np.loadtxt(
            data_path,
            delimiter=',',
            usecols=columns,
        )
    elif '.pickle' in data_path:
        with open(data_path, 'rb') as file:
            data = np.array(pickle.load(file)['params'])
    else:
        data = np.load(data_path)

    return data


def load_x_data(y_size: int) -> ndarray:
    """
    Fetches x data from file and matches the length to the y data

    Parameters
    ----------
    y_size : int
        Number of y data points

    Returns
    -------
    ndarray
        x data points
    """
    x_data = np.load('../data/spectra_x_axis.npy')

    # Make sure x data size is even
    if x_data.size % 2 != 0:
        x_data = np.append(x_data[:-2], np.mean(x_data[-2:]))

    # Make sure x data size is of the same size as y data
    if x_data.size != y_size and x_data.size % y_size == 0:
        x_data = x_data.reshape(int(x_data.size / y_size), - 1)
        x_data = np.mean(x_data, axis=0)

    return x_data


def data_normalization(
        data: np.ndarray,
        mean: bool = True,
        axis: int = None,
        transform: tuple[float, float] = None) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Transforms data either by normalising or
    scaling between 0 & 1 depending on if mean is true or false.

    Parameters
    ----------
    data : ndarray
        Data to be normalised
    mean : boolean, default = True
        If data should be normalised or scaled between 0 and 1
    axis : integer, default = None
        Which axis to normalise over, if none, normalise over all axes
    transform: tuple[float, float], default = None
        If transformation values exist already

    Returns
    -------
    tuple[ndarray, tuple[float, float]]
        Transformed data & transform values
    """
    if mean and not transform:
        transform = [np.mean(data, axis=axis), np.std(data, axis=axis)]
    elif not mean and not transform:
        transform = [
            np.min(data, axis=axis),
            np.max(data, axis=axis) - np.min(data, axis=axis)
        ]

        transform[1] = _min_clamp(transform[1])

    if axis:
        data = (data - np.expand_dims(transform[0], axis=axis)) /\
               np.expand_dims(transform[1], axis=axis)
    else:
        data = (data - transform[0]) / transform[1]

    return data, transform


def delete_data(directory: str = None, files: list[str] = None):
    """
    Removes all files in the provided directory and/or all files in the list of provided files

    Parameters
    ----------
    directory : string, default = None
        Directory to remove all files within
    files : list[string], default = None
        List of files to remove
    """
    if files:
        for file in files:
            if os.path.exists(file):
                os.remove(file)

    if directory:
        for file in os.listdir(directory):
            os.remove(directory + file)


def data_initialisation(
        spectra_path: str,
        log_params: list,
        val_frac: float = 0.1,
        transform: tuple[tuple[float, float], tuple[ndarray, ndarray]] = None,
        indices: ndarray = None) -> tuple[DataLoader, DataLoader]:
    """
    Initialises training and validation data

    Parameters
    ----------
    spectra_path : string
        Path to synthetic data
    log_params : list
        Index of each free parameter in logarithmic space
    val_frac : float, default = 0.1
        Fraction of validation data
    transform : tuple[tuple[float, float], tuple[ndarray, ndarray]], default = None
        Min and max spectral range and mean & standard deviation of parameters
    indices : ndarray, default = None
        Data indices for random training & validation datasets

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Dataloaders for the training and validation datasets
    """
    batch_size = 120
    kwargs = get_device()[0]

    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(
        spectra_path,
        log_params,
        transform=transform,
    )
    val_amount = max(int(len(dataset) * val_frac), 1)

    # If network hasn't trained on data yet, randomly separate training and validation
    if indices is None or indices.size != len(dataset):
        indices = np.random.choice(len(dataset), len(dataset), replace=False)

    dataset.indices = indices

    train_dataset = Subset(dataset, indices[:-val_amount])
    val_dataset = Subset(dataset, indices[-val_amount:])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    if val_frac == 0:
        val_loader = train_loader
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    print(f'\nTraining data size: {len(train_dataset)}\tValidation data size: {len(val_dataset)}')

    return train_loader, val_loader
