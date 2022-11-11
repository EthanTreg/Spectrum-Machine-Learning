import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, random_split


class SpectrumDataset(Dataset):
    """
    A dataset object containing spectrum data for PyTorch training

    Attributes
    ----------
    spectra : tensor
        Spectra dataset
    params : tensor
        Parameters for each spectra if supervised
    names : ndarray
        Names of each spectrum

    Methods
    -------
    downscaler(downscales)
        Downscales input spectra
    """
    def __init__(self, data_file: str, labels_file: str, log_params: list):
        """
        Parameters
        ----------
        data_file : string
            Path to the file with the spectra dataset
        labels_file : string
            Path to the labels file, if none, then an unsupervised approach is used
        log_params : list
            Index of each free parameter in logarithmic space
        """
        self.spectra = torch.from_numpy(np.log10(np.maximum(1e-8, np.load(data_file)))).float()
        self.spectra = (self.spectra - torch.min(self.spectra)) / \
                       (torch.max(self.spectra) - torch.min(self.spectra))

        if self.spectra.size(1) % 2 != 0:
            self.spectra = torch.cat((
                self.spectra[:, :-2],
                torch.mean(self.spectra[:, -2:], dim=1, keepdim=True)
            ), dim=1)

        if '.npy' in labels_file:
            self.params = np.load(labels_file)
            self.names = np.empty(self.spectra.size(0))
        else:
            labels = np.loadtxt(labels_file, skiprows=6, dtype=str)
            self.params = labels[:, 9:].astype(float)
            self.names = labels[:, 6]

        # Scale parameters
        self.params[:, log_params] = np.log10(self.params[:, log_params])
        # self.params = (self.params - np.min(self.params, axis=0)) / \
        #               (np.max(self.params, axis=0) - np.min(self.params, axis=0))
        self.params = (self.params - np.mean(self.params, axis=0)) / np.std(self.params, axis=0)
        self.params = torch.from_numpy(self.params).float()

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str]:
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
        val_frac: float = 0.1) -> tuple[DataLoader, DataLoader]:
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

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Dataloaders for the training and validation datasets
    """
    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(spectra_path, labels_path, log_params)
    val_amount = int(len(dataset) * val_frac)

    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_amount, val_amount])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, **kwargs)

    print(f'Training data size: {len(train_dataset)}\tValidation data size: {len(val_dataset)}\n')

    return train_loader, val_loader
