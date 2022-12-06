import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, Subset


class SpectrumDataset(Dataset):
    """
    A dataset object containing spectrum data for PyTorch training

    Attributes
    ----------
    spectra : tensor
        Spectra dataset
    params : tensor
        Parameters for each spectra if supervised
    transform : list[list[ndarray]]
        Min and max spectral range and mean & standard deviation of parameters
    names : ndarray
        Names of each spectrum
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
            labels_file: str,
            log_params: list,
            transform: list[list[np.ndarray]] = None):
        """
        Parameters
        ----------
        data_file : string
            Path to the file with the spectra dataset
        labels_file : string
            Path to the labels file, if none, then an unsupervised approach is used
        log_params : list
            Index of each free parameter in logarithmic space
        transform : list[ndarray], default = None
            Min and max spectral range and mean & standard deviation of parameters
            used for transformation after log
        """
        self.transform = transform
        self.indices = None
        spectra = np.load(data_file)

        # Set negative values equal to minimum real value
        if np.min(spectra) < 0:
            spectra = np.swapaxes(spectra, 0, 1)
            min_count = np.min(spectra, where=spectra > 0, initial=np.max(spectra), axis=0)
            spectra = np.swapaxes(np.maximum(spectra, min_count), 0, 1)

        spectra = np.log10(spectra)

        # Get spectra min and max values if transformation not supplied
        if transform:
            spectra_transform = transform[0]
        else:
            spectra_transform = [np.min(spectra), np.max(spectra)]
            self.transform = [spectra_transform]

        # Scale spectra between 0 and 1
        self.spectra = torch.from_numpy(
            (spectra - spectra_transform[0]) / (spectra_transform[1] - spectra_transform[0])
        ).float()

        # Make sure spectra length is even
        if self.spectra.size(1) % 2 != 0:
            self.spectra = torch.cat((
                self.spectra[:, :-2],
                torch.mean(self.spectra[:, -2:], dim=1, keepdim=True)
            ), dim=1)

        # Load spectra parameters and names
        if '.npy' in labels_file:
            params = np.load(labels_file)
            self.names = np.arange(self.spectra.size(0))
        else:
            labels = np.loadtxt(labels_file, skiprows=6, dtype=str)
            params = labels[:, 9:].astype(float)
            self.names = labels[:, 6]

        params[:, log_params] = np.log10(params[:, log_params])

        # Get parameter mean and standard deviation if transformation not supplied
        if transform:
            param_transform = transform[1]
        else:
            param_transform = [np.mean(params, axis=0), np.std(params, axis=0)]
            self.transform.append(param_transform)

        # Normalize parameters with mean of 0 and standard deviation of 1
        params = (params - param_transform[0]) / param_transform[1]
        self.params = torch.from_numpy(params).float()

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str | int]:
        """
        Gets the spectrum data for a given index
        If supervised learning return target parameters of spectrum otherwise returns spectrum name

        Parameters
        ----------
        idx : int
            Index of the target spectrum

        Returns
        -------
        (tensor, tensor, str)
            Spectrum data, target parameters and spectrum name
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


def data_initialisation(
        spectra_path: str,
        labels_path: str,
        log_params: list,
        kwargs: dict,
        val_frac: float = 0.1,
        transform: list[list[np.ndarray]] = None,
        indices: np.ndarray = None) -> tuple[DataLoader, DataLoader]:
    """
    Initialises training and validation data

    Parameters
    ----------
    spectra_path : string
        Path to synthetic data
    labels_path : string
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
    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(spectra_path, labels_path, log_params, transform=transform)
    val_amount = int(len(dataset) * val_frac)

    if indices is None or indices.size != len(dataset):
        indices = np.random.choice(len(dataset), len(dataset), replace=False)
        dataset.indices = indices

    train_dataset = Subset(dataset, indices[:-val_amount])
    val_dataset = Subset(dataset, indices[-val_amount:])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, **kwargs)

    print(f'Training data size: {len(train_dataset)}\tValidation data size: {len(val_dataset)}\n')

    return train_loader, val_loader
