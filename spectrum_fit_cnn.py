# TODO: Does log x affect results?
import os
from time import time
from typing import Type
from multiprocessing import Process, Queue, Value

import xspec
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim, Tensor
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
    """
    def __init__(self, data_file: str, labels_file: str, log_params: list[int]):
        """
        Parameters
        ----------
        data_file : str
            Path to the file with the spectra dataset
        labels_file : str
            Path to the labels file, if none, then an unsupervised approach is used
        log_params : list[integer]
            Indices for parameters in logarithmic space
        """
        self.spectra = torch.from_numpy(np.load(data_file)).float()

        if '.npy' in labels_file:
            self.params = torch.from_numpy(np.load(labels_file)).float()
            self.params[:, log_params] = torch.log10(self.params[:, log_params])
            # self.names = np.sort(np.array(os.listdir(data_file[:-4])))
            self.names = np.empty(self.spectra.size(0))
        else:
            labels = np.loadtxt(labels_file, skiprows=6, dtype=str)
            self.params = torch.from_numpy(labels[:, 9:].astype(np.float32))
            self.names = labels[:, 6]

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


class PyXspecFitting:
    """
    Handles fitting parameters to spectrum using a model and calculates fit statistic

    Attributes
    ----------
    model : string
        Model to use for PyXspec
    fix_params : ndarray
        Parameter number & value of fixed parameters
    param_limits : tensor
        Parameter lower & upper limits

    Methods
    -------
    fit_statistic(params)
        Calculates fit statistic from spectrum and parameter predictions
    """
    def __init__(self, model: str, fix_params: np.ndarray):
        """
        Parameters
        ----------
        device : device
            Device type for PyTorch to use
        model : string
            Model to use for PyXspec
        fix_params : ndarray
            Parameter number & value of fixed parameters
        """
        super().__init__()
        self.fix_params = fix_params
        self.model = xspec.Model(model)
        self.param_limits = np.empty((0, 2))
        xspec.AllModels.setEnergies('0.002 500 1000 log')
        xspec.Fit.statMethod = 'pgstat'

        # Generate parameter limits
        for j in range(self.model.nParameters):
            if j + 1 not in fix_params[:, 0]:
                limits = np.array(self.model(j + 1).values)[[2, 5]]
                self.param_limits = np.vstack((self.param_limits, limits))

    def fit_statistic(self, params: np.ndarray) -> float:
        """
        Forward function of PyXspec loss
        Parameters
        ----------
        params : ndarray
            Parameter predictions from CNN
        Returns
        -------
        float
            Loss value
        """
        # Merge fixed & free parameters
        fix_params_index = self.fix_params[0] - np.arange(self.fix_params.shape[1])
        params = np.insert(params, fix_params_index, self.fix_params[1])

        # Update model parameters
        self.model.setPars(params.tolist())

        # Calculate fit statistic loss
        return xspec.Fit.statistic


class Encoder(nn.Module):
    """
    Constructs a CNN network to predict model parameters from spectrum data

    Attributes
    ----------
    model : XspecModel
        PyXspec model
    conv1 : nn.Sequential
        First convolutional layer
    conv2 : nn.Sequential
        Second convolutional layer
    downscale : nn.Sequential
        Fully connected layer to produce parameter predictions

    Methods
    -------
    forward(x)
        Forward pass of decoder
    """
    def __init__(self, spectra_size: int, model: PyXspecFitting):
        """
        Parameters
        ----------
        spectra_size : int
            number of data points in spectra
        model : XspecModel
            PyXspec model
        """
        super().__init__()
        self.model = model
        test_tensor = torch.empty((1, 1, spectra_size))

        # Construct layers in CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=7, padding='same'),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2)

        conv_output = self.pool(self.conv2(self.conv1(test_tensor))).shape
        self.downscale = nn.Sequential(
            nn.Linear(in_features=conv_output[1] * conv_output[2], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=model.param_limits.shape[0]),
        )

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
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.downscale(x)
        return x


class Decoder(nn.Module):
    """
    Constructs a CNN network to generate spectra from model parameters

    Attributes
    ----------
    spectra_size : int
        number of data points in spectra
    upscale : nn.Sequential
        Fully connected layers to produce spectra from parameters
    conv1 : nn.Sequential
        First convolutional layer
    conv2 : nn.Sequential
        Second convolutional layer

    Methods
    -------
    forward(x)
        Forward pass of encoder
    """
    def __init__(self, spectra_size: int, model: PyXspecFitting):
        """
        Parameters
        ----------
        spectra_size : int
            number of data points in spectra
        model : XspecModel
            PyXspec model
        """
        super().__init__()
        self.spectra_size = spectra_size

        # Construct layers in CNN
        self.upscale = nn.Sequential(
            nn.Linear(in_features=model.param_limits.shape[0], out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.spectra_size * 4),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=7, padding='same'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN taking parameter inputs and producing a spectrum output

        Parameters
        ----------
        x : tensor
            Input parameters

        Returns
        -------
        tensor
            Generated spectrum
        """
        x = self.upscale(x)
        x = x.view(x.size(0), -1, self.spectra_size)
        x = self.conv1(x)
        x = torch.squeeze(self.conv2(x), dim=1)
        return x


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
def fit_statistic(
        total: int,
        names: list[str],
        params: np.ndarray,
        model: PyXspecFitting,
        counter: Value,
        queue: Queue):
    """
    Custom loss function using PyXspec to calculate statistic for Poisson data
    and Gaussian background (PGStat) and L1 loss for parameters that exceed limits

    params
    ----------
    total : int
        Total number of spectra
    names : list[string]
        Spectra names
    params : ndarray
        Output from CNN of parameter predictions between 0 & 1
    model : Model
        PyXspec model
    counter : Value
        Number of spectra fitted
    queue : Queue
        Multiprocessing queue to add PGStat loss
    """
    loss = 0

    # Loop through each spectrum in the batch
    for i, name in enumerate(names):
        # Limit parameters
        param_min = model.param_limits[:, 0] + 1e-6 * (
                model.param_limits[:, 1] - model.param_limits[:, 0])
        param_max = model.param_limits[:, 1] - 1e-6 * (
                model.param_limits[:, 1] - model.param_limits[:, 0])

        params = np.clip(params, a_min=param_min, a_max=param_max)

        xspec.Spectrum(name)

        loss += model.fit_statistic(params[i])

        xspec.AllData.clear()

        with counter.get_lock():
            counter.value += 1

        progress_bar(counter.value, total)

    # Average loss of batch
    queue.put(loss / len(names))


def network_initialisation(
        val_frac: float,
        synth_path: str,
        labels_path: str,
        log_params: list[int],
        kwargs: dict,
        architecture: Type[Encoder | Decoder],
        model: PyXspecFitting,
        device: torch.device
) -> tuple[Encoder | Decoder, torch.optim.Optimizer, DataLoader, DataLoader]:
    """
    Initialises data and neural network for either encoder or decoder CNN

    Parameters
    ----------
    val_frac : float
        Fraction of validation data
    synth_path : string
        Path to synthetic data
    labels_path : string
        Path to labels
    log_params : list[integer]
        Indices of parameters in logarithmic space
    kwargs : dictionary
        Keyword arguments for dataloader
    architecture : Encoder | Decoder
        Which network architecture to use
    model : PyXspecFitting
        PyXspec model that the data is based on
    device : device
        Which device type PyTorch should use

    Returns
    -------
    tuple[Encoder | Decoder, Optimizer, DataLoader, DataLoader]
        Neural network, optimizer and dataloaders for the training and validation datasets
    """
    learning_rate = 5e-5

    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(synth_path, labels_path, log_params)
    val_amount = int(len(dataset) * val_frac)

    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_amount, val_amount])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, **kwargs)

    # Initialise the CNN
    cnn = architecture(dataset[0][0].size(dim=0), model)
    cnn.to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    return cnn, optimizer, train_loader, val_loader


def train(
        device: torch.device,
        loader: DataLoader,
        cnn: Encoder | Decoder,
        optimizer: torch.optim.Optimizer) -> float:
    """
    Trains the CNN on spectra data either supervised or unsupervised

    Parameters
    ----------
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training
    optimizer : optimizer
        Optimisation method to use for training

    Returns
    -------
    float
        Average loss value
    """
    epoch_loss = 0
    cnn.train()

    for spectra, params, _ in loader:
        spectra = spectra.to(device)
        params = params.to(device)

        # Generate predictions and loss
        if isinstance(cnn, Encoder):
            output = cnn(spectra)
            loss = nn.CrossEntropyLoss()(output, params)
        else:
            output = cnn(params)
            loss = nn.MSELoss()(output, spectra)

        # Optimise CNN
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def test(
        log_params: list[int],
        device: torch.device,
        loader: DataLoader,
        cnn: Decoder,
        dirs: list[str]) -> float:
    """
    Tests the CNN on spectra data either supervised or unsupervised

    Parameters
    ----------
    log_params : list[integer]
        Indices of parameters in logarithmic space
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training
    dirs : list[string]
        Directory of project & dataset files

    Returns
    -------
    float
        Average loss value
    """
    loss = 0
    cnn.eval()

    if isinstance(cnn, Encoder):
        counter = Value('i', 0)
        os.chdir(dirs[1])
        queue = Queue()
        loader_output = next(enumerate(loader))[1]
        batch = loader_output[0].size(0)
        outputs = torch.empty((0, loader_output[1].size(1)))

    with torch.no_grad():
        for spectra, params, _ in loader:
            spectra = spectra.to(device)

            # Generate predictions and loss
            if isinstance(cnn, Encoder):
                output = cnn(spectra)
                output[:, log_params] = 10 ** output[:, log_params]
                outputs = torch.vstack((outputs, output))
            else:
                params = params.to(device)
                output = cnn(params)
                loss += nn.MSELoss()(output, spectra).item()

    if isinstance(cnn, Encoder):
        processes = [Process(
            target=fit_statistic,
            args=(
                len(loader.dataset),
                data[2],
                outputs[batch * i:batch * (i + 1)],
                cnn.model,
                counter,
                queue
            )
        ) for i, data in enumerate(loader)]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        losses = [queue.get() for _ in processes]

        os.chdir(dirs[0])

        return sum(losses) / len(losses)

    return loss / len(loader)


def main():
    """
    Main function for spectrum machine learning
    """
    # Initialize variables
    num_epochs = 100
    val_frac = 0.1
    train_loss = []
    val_loss = []
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = '../../Documents/Nicer_Data/ethan'
    spectra_path = './data/preprocessed_spectra.npy'
    synth_path = './data/synth_spectra.npy'
    labels_path = './data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    synth_params_path = './data/synth_spectra_params.npy'
    fix_params = np.array([[4, 0], [5, 100]])
    log_params = [0, 2, 3, 4]

    # Xspec initialization
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0
    xspec.AllModels.lmod('simplcutx', dirPath='../../Documents/Xspec_Models/simplcutx')
    model = PyXspecFitting('tbabs(simplcutx(ezdiskbb))', fix_params)

    # Set device to GPU if available
    device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    decoder, d_optimizer, d_train_loader, d_val_loader = network_initialisation(
        val_frac,
        synth_path,
        synth_params_path,
        log_params,
        kwargs,
        Decoder,
        model,
        device
    )

    encoder, e_optimizer, e_train_loader, e_val_loader = network_initialisation(
        val_frac,
        spectra_path,
        labels_path,
        log_params,
        kwargs,
        Encoder,
        model,
        device
    )

    # Train for each epoch
    for epoch in range(num_epochs):
        t_initial = time()
        # Train & validate CNN
        # test(device, d_val_loader, decoder, [root_dir, data_dir])
        # losses = train(device, num_epochs, d_train_loader, decoder, d_optimizer)
        # test(device, d_val_loader, decoder, [root_dir, data_dir])
        val_loss.append(test(log_params, device, e_val_loader, encoder, [root_dir, data_dir]))
        train_loss.append(train(device, e_train_loader, encoder, e_optimizer))

        print(f'Epoch [{epoch + 1}/{num_epochs}]\t'
              f'Training loss: {train_loss[-1]:.3e}\t'
              f'Validation loss: {val_loss[-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    val_loss.append(test(log_params, device, e_val_loader, encoder, [root_dir, data_dir]))

    plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.yscale('log')
    plt.legend(fontsize=20)
    plt.show()


if __name__ == '__main__':
    main()
