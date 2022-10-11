# TODO: Does log x affect results?
import os
import torch
import numpy as np
from torch import nn
from torch import optim, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from xspec import Spectrum, Model, Fit, Xset, AllData

from data_preprocessing import spectrum_data


class SpectrumDataset(Dataset):
    """
    A dataset object containing spectrum data for PyTorch training

    Attributes
    ----------
    data_dir : str
            Directory of the spectra dataset
    data_amount : int
        Amount of data to retrieve
    annotations_file : str, default = None
        Path to the annotations file, if none, then an unsupervised approach is used
    transform, default = None
        PyTorch transformation to spectra dataset
    spectra_files : ndarray
        Spectra file names

    Methods
    -------
    spectra_names()
        Fetches the file names of all spectra up to the defined data amount
    """

    def __init__(
            self,
            data_dir: str,
            data_amount: int,
            annotations_file: str = None,
            transform=None):
        """
        Parameters
        ----------
        data_dir : str
            Directory of the spectra dataset
        data_amount : int
            Amount of data to retrieve
        annotations_file : str, default = None
            Path to the annotations file, if none, then an unsupervised approach is used
        transform, default = None
            PyTorch transformation to spectra dataset
        """
        self.data_dir = data_dir
        self.data_amount = data_amount
        self.annotations_file = annotations_file
        self.transform = transform

        self.spectra_files = self.spectra_names()

    def __len__(self):
        return self.spectra_files.size

    def __getitem__(self, idx: int) -> (
            tuple[Tensor | float, str, Tensor | float] | tuple[Tensor | float, str]):
        """
        Gets the spectrum and name of spectrum file for a given index
        If supervised learning return target parameters of spectrum

        Parameters
        ----------
        idx : int
            Index of the target spectrum

        Returns
        -------
        (tensor, string, optional list)
            Spectrum tensor, spectrum file name and target parameters if supervised
        """
        spectrum = torch.from_numpy(spectrum_data(self.data_dir, self.spectra_files[idx])[1])

        # Apply transformation to spectrum data
        if self.transform:
            spectrum = self.transform(spectrum)

        # If supervised learning
        if self.annotations_file:
            target_parameters = self.annotations_file[idx]
            return spectrum, self.spectra_files[idx], target_parameters

        return spectrum, self.spectra_files[idx]

    def spectra_names(self) -> np.ndarray:
        """
        Fetches the file names of all spectra up to the defined data amount

        Returns
        -------
        ndarray
            Array of spectra file names
        """
        # Fetch all files within directory
        all_files = os.listdir(self.data_dir)

        # Remove all files that aren't spectra and limit to data_amount
        return np.delete(all_files, np.char.find(all_files, '.jsgrp') == -1)[:self.data_amount]


class CNN(nn.Module):
    """
    Constructs a CNN network to predict model parameters from spectrum data

    Attributes
    ----------
    model : str
        Model(s) for PyXspec to use
    parameters : int
        Number of parameters in model
    conv1 : nn.Sequential
        First convolutional layer
    conv2 : nn.Sequential
        Second convolutional layer
    out : nn.Linear
        Fully connected layer to produce parameter predictions

    Methods
    -------
    forward(x)
        Forward pass of CNN
    """

    def __init__(self, model: str, parameters: int):
        """
        Parameters
        ----------
        model : str
            Model(s) for PyXspec to use
        parameters : int
            Number of parameters in model
        """
        super().__init__()
        self.model = model

        # Construct layers in CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.out = nn.Linear(in_features=int(32 * 82), out_features=parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN taking spectrum input and producing parameter predictions output

        Parameters
        ----------
        x : tensor
            Input spectrum

        Returns
        -------
        tensor
            Parameter predictions
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        # Convert parameters to 0-1 range for parameter limits
        output = torch.sigmoid(output)
        return output


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


# TODO: Is calculating loss manually faster
# TODO: Does this work with PyTorch?
def pyxspec_loss(names: list[str], model: str, parameters: torch.Tensor) -> torch.Tensor:
    """
    Custom loss function using PyXspec to calculate statistic
    for Poisson data and Gaussian background (PGStat)

    Parameters
    ----------
    names : list[string]
        Spectra names
    model : string
        Model name(s) for PyXspec to use
    parameters : tensor
        Output from CNN of parameter predictions between 0 & 1

    Returns
    -------
    tensor
        Average PGStat loss
    """
    loss = []

    # Loop through each spectrum in the batch
    for i, name in enumerate(names):
        # Load spectrum + background & response files
        Spectrum(name)

        # Create model of diskbb + pow
        xspec_model = Model(model)

        # Adjust CNN output from 0-1 range to parameter limits range
        for j in range(xspec_model.nParameters):
            param_min = xspec_model(j + 1).values[2]
            param_max = xspec_model(j + 1).values[5]
            parameters[i, j] = 0.9999 * parameters[i, j] * (param_max - param_min) + param_min

        # Update model parameters and get PGStat
        xspec_model.setPars(parameters[i].tolist())
        Fit.statMethod = 'pgstat'
        loss.append(Fit.statistic)
        AllData.clear()

        progress_bar(i, len(names))

    # Average loss of batch
    return torch.mean(torch.tensor(loss, requires_grad=True, dtype=float))


def train(
        device: torch.device,
        num_epochs: int,
        loader: DataLoader,
        cnn: CNN,
        optimizer: torch.optim.Optimizer,
        dirs: list[str],
        supervised: bool = False):
    """
    Trains the CNN on spectra data either supervised or unsupervised

    Parameters
    ----------
    device : device
        Device type for PyTorch to use
    num_epochs : int
        Number of epochs to train
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training
    optimizer : optimizer
        Optimisation method to use for training
    dirs : list[string]
        Directory of project & dataset files
    supervised : boolean
        Whether to use supervised or unsupervised learning
    """
    cnn.train()

    # Train for each epoch
    for epoch in range(num_epochs):
        for i, data in enumerate(loader):
            # Fetch data file for training
            if supervised:
                spectra, names, labels = data
                labels = labels.to(device)
            else:
                spectra, names = data

            # Pass data through CNN
            spectra = torch.unsqueeze(spectra.to(device=device, dtype=torch.float), dim=1)
            output = cnn(spectra)

            # Calculate loss
            if supervised:
                loss = nn.CrossEntropyLoss()(output, labels)
            else:
                os.chdir(dirs[1])
                loss = pyxspec_loss(names, cnn.model, output)
                os.chdir(dirs[0])

            # Optimise CNN
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % int(len(loader) / 10) == 0 or i + 1 == len(loader):
                print(f'Epoch [{epoch + 1}/{num_epochs}],'
                      f'Step [{i + 1}/{len(loader)}],'
                      f'Loss: {loss.item():.3e}')


# TODO: Test function


def main():
    """
    Main function for spectrum machine learning
    """
    # Initialize variables
    random_seed = 3
    num_epochs = 5
    Xset.chatter = 0
    Xset.logChatter = 0
    val_frac = 0.1
    torch.manual_seed(random_seed)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = '../../Documents/Nicer_Data/ethan'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(data_dir, 10000)
    val_amount = int(len(dataset) * val_frac)

    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_amount, val_amount])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
    # val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, **kwargs)

    # Initialise the CNN
    cnn = CNN('diskbb + pow', 4)
    cnn.to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)

    # Train CNN
    train(device, num_epochs, train_loader, cnn, optimizer, [root_dir, data_dir])


if __name__ == '__main__':
    main()
