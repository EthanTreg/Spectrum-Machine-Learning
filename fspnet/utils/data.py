"""
Loads data and creates data loaders for network training
"""
import torch
import numpy as np
from numpy import ndarray
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from fspnet.utils.utils import even_length


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
    uncertainty : tensor
        Spectral Poisson uncertainty
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
            params_path: str,
            log_params: list[int],
            transform: list[tuple[ndarray, ndarray]] = None):
        """
        Parameters
        ----------
        data_file : string
            Path to the file with the spectra dataset
        params_path : string
            Path to the labels file, if none, then an unsupervised approach is used
        log_params : list[int]
            Index of each free parameter in logarithmic space
        transform : list[tuple[ndarray, ndarray]], default = None
            Min and max spectral range and mean & standard deviation of parameters
            used for transformation after log
        """
        self.indices = None
        self.log_params = log_params
        data = np.load(data_file)

        if len(data.shape) == 3:
            self.spectra, self.uncertainty = np.rollaxis(np.load(data_file), 1)
        else:
            self.spectra = np.load(data_file)
            self.uncertainty = np.empty_like(self.spectra)

        if transform:
            spectra_transform, params_transform = transform
        else:
            spectra_transform = params_transform = None

        # Set negative values equal to minimum positive value
        if np.min(self.spectra) <= 0:
            self.spectra = self._min_clamp(self.spectra)

        if np.min(self.uncertainty) <= 0:
            self.uncertainty = self._min_clamp(self.uncertainty)

        # Transform spectra & uncertainty
        self.uncertainty = np.log10(1 + self.uncertainty / self.spectra)
        self.spectra = np.log10(self.spectra)
        self.spectra, spectra_transform = self._data_normalization(
            self.spectra,
            mean=False,
            transform=spectra_transform,
        )
        self.uncertainty /= spectra_transform[1]

        # Make sure spectra & uncertainty length is even
        self.spectra = even_length(torch.from_numpy(self.spectra).float())
        self.uncertainty = even_length(torch.from_numpy(self.uncertainty).float())

        # If no parameters are provided
        if not params_path:
            self.params = np.empty(self.spectra.size(0))
            self.transform = transform
            self.names = np.arange(self.spectra.size(0))
            return

        # Load spectra parameters and names
        if '.npy' in params_path:
            self.params = np.load(params_path)
            self.names = np.arange(self.spectra.size(0))
        else:
            labels = np.loadtxt(params_path, skiprows=6, dtype=str)
            self.params = labels[:, 9:].astype(float)
            self.names = labels[:, 6]

        # Transform parameters
        self.params[:, log_params] = np.log10(self.params[:, log_params])
        self.params, params_transform = self._data_normalization(
            self.params,
            transform=params_transform,
        )
        self.params = torch.from_numpy(self.params).float()

        # Get parameter mean and standard deviation if transformation not supplied
        self.transform = [spectra_transform, params_transform]

    def __len__(self) -> int:
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, str | int]:
        """
        Gets the training data for a given index

        Parameters
        ----------
        idx : integer
            Index of the target spectrum

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, string | integer]
            Spectrum data, target parameters, spectrum uncertainty and spectrum name/number
        """
        return self.spectra[idx], self.params[idx], self.uncertainty[idx], self.names[idx]

    def _min_clamp(self, data: ndarray) -> ndarray:
        """
        Clamps all values <= 0 to the minimum non-zero positive value

        Parameters
        ----------
        data : ndarray
            Input data to be clamped

        Returns
        -------
        ndarray
            Clamped data
        """
        data = np.swapaxes(data, 0, 1)
        min_count = np.min(
            data,
            where=data > 0,
            initial=np.max(data),
            axis=0,
        )

        return np.swapaxes(np.maximum(data, min_count), 0, 1)

    def _data_normalization(
            self,
            data: ndarray,
            mean: bool = True,
            transform: tuple[float, float] = None) -> tuple[ndarray, tuple[float, float]]:
        """
        Transforms data either by normalising or
        scaling between 0 & 1 depending on if mean is true or false.

        Parameters
        ----------
        data : ndarray
            Data to be normalised
        mean : bool, default = True
            If data should be normalised or scaled between 0 and 1
        transform: tuple[float, float], default = None
            If transformation values exist already

        Returns
        -------
        tuple[ndarray, tuple[float, float]]
            Transformed data & transform values
        """
        if mean and not transform:
            transform = [np.mean(data, axis=0), np.std(data, axis=0)]
        elif not mean and not transform:
            transform = [
                np.min(data),
                np.max(data) - np.min(data)
            ]

        data = (data - transform[0]) / transform[1]

        return data, transform

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


def data_initialisation(
        spectra_path: str,
        params_path: str,
        log_params: list,
        kwargs: dict,
        val_frac: float = 0.1,
        transform: list[list[ndarray]] = None,
        indices: ndarray = None) -> tuple[DataLoader, DataLoader]:
    """
    Initialises training and validation data

    Parameters
    ----------
    spectra_path : string
        Path to synthetic data
    params_path : string
        Path to labels
    log_params : list
        Index of each free parameter in logarithmic space
    kwargs : dictionary
        Keyword arguments for dataloader
    val_frac : float, default = 0.1
        Fraction of validation data
    transform : list[ndarray], default = None
        Min and max spectral range and mean & standard deviation of parameters
    indices : ndarray, default = None
        Data indices for random training & validation datasets

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Dataloaders for the training and validation datasets
    """
    batch_size = 120

    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(spectra_path, params_path, log_params, transform=transform)
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

    print(f'Training data size: {len(train_dataset)}\tValidation data size: {len(val_dataset)}\n')

    return train_loader, val_loader
