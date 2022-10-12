# TODO: Does log x affect results?
import os
from time import time
import torch
import numpy as np
from torch import nn
from torch import optim, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from xspec import Spectrum, Model, Fit, Xset, AllData


class SpectrumDataset(Dataset):
    """
    A dataset object containing spectrum data for PyTorch training

    Attributes
    ----------
    labels_file : str, default = None
        Path to the labels file, if none, then an unsupervised approach is used
    spectra : tensor
        Spectra dataset
    labels : ndarray
        labels for each spectra if supervised
    """
    def __init__(self, data_dir: str, labels_file: str = None):
        """
        Parameters
        ----------
        data_dir : str
            Directory of the spectra dataset
        labels_file : str, default = None
            Path to the labels file, if none, then an unsupervised approach is used
        """
        self.labels_file = labels_file
        self.spectra = torch.from_numpy(np.load(data_dir))

        if labels_file:
            self.labels = torch.from_numpy(
                np.loadtxt(self.labels_file, skiprows=6, dtype=str)[:, 9:].astype(float)
            )

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor | float, Tensor | np.ndarray]:
        """
        Gets the spectrum data for a given index
        If supervised learning return target parameters of spectrum otherwise returns spectrum name

        Parameters
        ----------
        idx : int
            Index of the target spectrum

        Returns
        -------
        (tensor, tensor | ndarray)
            Spectrum tensor and target parameters if supervised otherwise spectrum name
        """
        spectrum = self.spectra[idx]

        # If supervised learning
        if self.labels_file:
            target_parameters = self.labels[idx]
            return spectrum, target_parameters

        # TODO: Fix unsupervised file loading
        return spectrum, self.spectra_files[idx]


class CNN(nn.Module):
    """
    Constructs a CNN network to predict model parameters from spectrum data

    Attributes
    ----------
    model : str
        Model(s) for PyXspec to use
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
    def __init__(self, model: str, spectra_size: int, parameters: int):
        """
        Parameters
        ----------
        model : str
            Model(s) for PyXspec to use
        spectra_size : int
            number of data points in spectra
        parameters : int
            Number of parameters in model
        """
        super().__init__()
        self.model = model
        test_tensor = torch.empty((1, 1, spectra_size))

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

        conv_output = self.conv2(self.conv1(test_tensor)).shape
        self.out = nn.Linear(in_features=conv_output[1] * conv_output[2], out_features=parameters)

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
# TODO: Load multiple spectra at once
def pyxspec_loss(
        device: torch.device,
        names: list[str],
        model: str,
        params: torch.Tensor) -> torch.Tensor:
    """
    Custom loss function using PyXspec to calculate statistic
    for Poisson data and Gaussian background (PGStat)

    params
    ----------
    device : device
        Device type for PyTorch to use
    names : list[string]
        Spectra names
    model : string
        Model name(s) for PyXspec to use
    params : tensor
        Output from CNN of parameter predictions between 0 & 1

    Returns
    -------
    tensor
        Average PGStat loss
    """
    loss = []

    # Loop through each spectrum in the batch
    for i, name in enumerate(names):
        spectrum_loss = 0

        # Load spectrum + background & response files
        Spectrum(name)

        # Create model of diskbb + pow
        xspec_model = Model(model)

        # Adjust CNN output from 0-1 range to parameter limits range
        for j in range(xspec_model.nparams):
            param_min = xspec_model(j + 1).values[2]
            param_max = xspec_model(j + 1).values[5]

            # Check parameter limits and if exceeded, use L1 loss and set parameter within limit
            if params[i, j] >= param_max:
                spectrum_loss += abs(params[i, j] - param_max)
                params[i, j] = param_max - 1e6 * abs(param_max)
            elif params[i, j] <= param_min:
                spectrum_loss += abs(params[i, j] - param_min)
                params[i, j] = param_min + 1e6 * abs(param_min)

        # Update model params and get PGStat
        xspec_model.setPars(params[i].tolist())
        Fit.statMethod = 'pgstat'
        spectrum_loss += Fit.statistic
        loss.append(spectrum_loss)
        AllData.clear()

        progress_bar(i, len(names))

    # Average loss of batch
    return torch.mean(torch.tensor(loss, requires_grad=True, dtype=float).to(device))


def train(
        device: torch.device,
        num_epochs: int,
        loader: DataLoader,
        cnn: CNN,
        optimizer: torch.optim.Optimizer,
        dirs: list[str] = None):
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
    dirs : list[string], default = None
        Directory of project & dataset files
    """
    print_num_epoch = 1
    t_initial = time()
    cnn.train()

    # Train for each epoch
    for epoch in range(num_epochs):
        for i, data in enumerate(loader):
            # Fetch data for training
            spectra, labels = data
            spectra = spectra.to(device)

            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)

            # Pass data through CNN
            spectra = torch.unsqueeze(spectra.to(device=device, dtype=torch.float), dim=1)
            output = cnn(spectra)

            # Calculate loss
            if isinstance(labels, torch.Tensor):
                loss = nn.CrossEntropyLoss()(output, labels)
            else:
                os.chdir(dirs[1])
                loss = pyxspec_loss(device, labels, cnn.model, output)
                os.chdir(dirs[0])

            # Optimise CNN
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % int(len(loader) / print_num_epoch) == 0 or i + 1 == len(loader):
                print(f'Epoch [{epoch + 1}/{num_epochs}]\t'
                      f'Step [{i + 1}/{len(loader)}]\t'
                      f'Loss: {loss.item() / spectra.shape[0]:.3e}\t'
                      f'Time: {time() - t_initial:.1f}')


def test(device: torch.device, loader: DataLoader, cnn: CNN, dirs: list[str] = None):
    """
    Tests the CNN on spectra data either supervised or unsupervised

    Parameters
    ----------
    device : device
        Device type for PyTorch to use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training
    dirs : list[string], default = None
        Directory of project & dataset files
    """
    loss = 0
    cnn.eval()

    with torch.no_grad():
        for data in loader:
            # Fetch data for testing
            spectra, labels = data
            spectra = spectra.to(device)

            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)

            # Pass data through CNN
            spectra = torch.unsqueeze(spectra.to(device=device, dtype=torch.float), dim=1)
            output = cnn(spectra)

            # Calculate loss
            # TODO: Quantitative loss
            if isinstance(labels, torch.Tensor):
                loss += nn.CrossEntropyLoss()(output, labels)
            else:
                os.chdir(dirs[1])
                loss += pyxspec_loss(device, labels, cnn.model, output)
                os.chdir(dirs[0])

    print(f'Average loss: {loss / len(loader):.2e}')


def main():
    """
    Main function for spectrum machine learning
    """
    # Initialize variables
    num_epochs = 20
    Xset.chatter = 0
    Xset.logChatter = 0
    val_frac = 0.1
    # torch.manual_seed(3)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = '../../Documents/Nicer_Data/ethan'
    spectra_path = './data/preprocessed_spectra.npy'
    labels_path = './data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(spectra_path, labels_path)
    val_amount = int(len(dataset) * val_frac)

    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_amount, val_amount])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, **kwargs)

    # Initialise the CNN
    # TODO: Change model
    cnn = CNN('diskbb + pow', dataset[0][0].size(dim=0), dataset[0][1].size(dim=0))
    cnn.to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    # Train & validate CNN
    test(device, val_loader, cnn)
    train(device, num_epochs, train_loader, cnn, optimizer, [root_dir, data_dir])
    test(device, val_loader, cnn)


if __name__ == '__main__':
    main()
