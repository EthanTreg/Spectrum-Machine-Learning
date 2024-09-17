"""
Loads data and creates data loaders for network training
"""
import os
import pickle

import numpy as np
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from netloader.utils.utils import get_device


class SpectrumDataset(Dataset):
    """
    A dataset object containing spectrum data for PyTorch training

    Attributes
    ----------
    log_params : list
        Index of each free parameter in logarithmic space
    names : ndarray
        Names of each spectrum
    spectra : tensor
        Spectra dataset
    params : tensor
        Parameters for each spectra if supervised
    idxs : ndarray, default = None
        Data indices for random training & validation datasets
    """
    def __init__(self, data_file: str, log_params: list[int]):
        """
        Parameters
        ----------
        data_file : string
            Path to the file with the spectra dataset
        log_params : list[int]
            Index of each free parameter in logarithmic space
            used for transformation after log
        """
        self.log_params: list[int] = log_params
        self.idxs: ndarray | None = None

        with open(data_file, 'rb') as file:
            data = pickle.load(file)

        self.spectra = np.array(data['spectra'])[:, :data['spectra'].shape[-1] // 2 * 2]
        self.uncertainty = np.array(
            data['uncertainties'],
        )[:, :data['uncertainties'].shape[-1] // 2 * 2]

        # Get spectra names if available
        if 'names' in data:
            self.names = data['names']
        else:
            self.names = np.arange(len(self.spectra))

        # If no parameters are provided
        if 'params' not in data:
            self.params = np.empty(len(self.spectra))
            return

        self.params = np.array(data['params'])

        if 'param_uncertainty' in data:
            self.param_uncertainty = np.abs(data['param_uncertainty'])
        else:
            self.param_uncertainty = np.ones_like(self.params)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> tuple[ndarray, ndarray | Tensor, ndarray | Tensor]:
        """
        Gets the training data for a given index

        Parameters
        ----------
        idx : integer
            Index of the target spectrum

        Returns
        -------
        tuple[integer | string, Tensor, Tensor, Tensor]
            Spectrum name/number, spectrum data, target parameters, and uncertainty
        """
        # module = torch if isinstance(self.spectra, Tensor) else np
        # return self.names[idx], self.params[idx], module.stack((
        #     self.spectra[idx],
        #     self.uncertainty[idx],
        # ))
        return self.names[idx], self.params[idx], self.spectra[idx][None]


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
    x_data = np.load('../../Spectrum-Machine-Learning/data/spectra_x_axis.npy')

    # Make sure x data size is even
    if x_data.size % 2 != 0:
        x_data = np.append(x_data[:-2], np.mean(x_data[-2:]))

    # Make sure x data size is of the same size as y data
    if x_data.size != y_size and x_data.size % y_size == 0:
        x_data = x_data.reshape(int(x_data.size / y_size), - 1)
        x_data = np.mean(x_data, axis=0)
    elif x_data.size != y_size:
        x_data = np.arange(y_size)

    return x_data


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


def loader_init(
        dataset: SpectrumDataset,
        batch_size: int = 120,
        val_frac: float = 0.1,
        idxs: ndarray = None) -> tuple[DataLoader, DataLoader]:
    """
    Initialises training and validation data loaders

    Parameters
    ----------
    dataset : DarkDataset
        Dataset to generate data loaders for
    batch_size : integer, default = 1024
        Number of data inputs per weight update,
        smaller values update the network faster and requires less memory, but is more unstable
    val_frac : float, default = 0.1
        Fraction of validation data
    idxs : ndarray, default = None
        Data indices for random training & validation datasets

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Dataloaders for the training and validation datasets
    """
    kwargs = get_device()[0]

    # Fetch dataset & calculate validation fraction
    val_amount = max(int(len(dataset) * val_frac), 1)

    # If network hasn't trained on data yet, randomly separate training and validation
    if idxs is None or idxs.size != len(dataset):
        idxs = np.arange(len(dataset))
        np.random.shuffle(idxs)

    dataset.idxs = idxs

    train_dataset = Subset(dataset, idxs[:-val_amount])
    val_dataset = Subset(dataset, idxs[-val_amount:])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    if val_frac == 0:
        val_loader = train_loader
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    print(f'\nTraining data size: {len(train_loader.dataset)}'
          f'\tValidation data size: {len(val_loader.dataset)}')

    return train_loader, val_loader
