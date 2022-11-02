import os
import json
from time import time
from typing import Type
from multiprocessing import Process, Queue, Value

import xspec
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import nn
from torch import optim, Tensor
from torch.utils.data import Dataset, DataLoader, random_split

# TODO: Simple data
# TODO: Longer training


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
        # self.spectra = torch.from_numpy(np.load(data_file)).float()
        self.spectra = torch.from_numpy(np.log10(np.maximum(1e-8, np.load(data_file)))).float()
        self.spectra = (self.spectra - torch.mean(self.spectra)) / torch.std(self.spectra)

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
        self.param_limits = np.empty((0, 2))

        xspec.Xset.chatter = 0
        xspec.Xset.logChatter = 0
        xspec.AllModels.lmod('simplcutx', dirPath='../../Documents/Xspec_Models/simplcutx')
        self.model = xspec.Model(model)
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


class EmptyLayer(nn.Module):
    """
    Empty layer to be used within the network module list for shortcut and reshaping layers
    """


class PixelShuffle1d(torch.nn.Module):
    """
    Used for upscaling by scale factor r for an input (*, C x r, L) to an output (*, C, L x r).

    Equivalent to torch.nn.PixelShuffle but for 1D.

    Attributes
    ----------
    upscale_factor : integer
        Upscaling factor

    Methods
    -------
    forward(x)
        Forward pass of PixelShuffle1D
    """
    def __init__(self, upscale_factor: int):
        """
        Parameters
        ----------
        upscale_factor : integer
            Upscaling factor
        """
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor):
        """
        Forward pass of pixel shuffle

        Parameters
        ----------
        x : Tensor, shape (*, C x r, L)
            Input tensor

        Returns
        -------
        Tensor, (*, C, L x r)
            Output tensor
        """
        output_channels = x.size(1) // self.upscale_factor
        output_size = self.upscale_factor * x.size(2)

        x = x.view([x.size(0), self.upscale_factor, output_channels, x.size(2)])
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), output_channels, output_size)
        return x


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
    layers : list[dictionary]
        Layers with layer parameters
    network : ModuleList
        Network construction

    Methods
    -------
    forward(x)
        Forward pass of decoder
    """
    def __init__(self, spectra_size: int, config_file: str, model: PyXspecFitting):
        """
        Parameters
        ----------
        spectra_size : int
            number of data points in spectra
        config_file : string
            Path to the config file
        model : XspecModel
            PyXspec model
        """
        super().__init__()
        # Construct layers in CNN
        self.layers, self.network = create_network(
            model.param_limits.shape[0],
            spectra_size,
            config_file,
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
        outputs = []

        for i, layer in enumerate(self.layers):
            if layer['type'] == 'reshape':
                x = x.view(x.size(0), *layer['output'])
            elif layer['type'] == 'shortcut':
                x = torch.cat((x, outputs[layer['layer']]), dim=1)
            else:
                x = self.network[i](x)

            outputs.append(x)

        return torch.squeeze(x, dim=1)


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


def create_network(
        input_dim: int,
        spectra_size: int,
        config_path: str) -> tuple[list[dict], nn.ModuleList]:
    """
    Creates a network from a config file

    Parameters
    ----------
    input_dim : integer
        Size of the input
    spectra_size : integer
        Size of the spectra
    config_path : string
        Path to the config file

    Returns
    -------
    tuple[list[dictionary], ModuleList]
        Layers in the network with parameters and network construction
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        file = json.load(file)

    dropout_prob = file['net']['dropout_prob']
    dims = [input_dim]
    module_list = nn.ModuleList()

    for i, layer in enumerate(file['layers']):
        module = nn.Sequential()

        if layer['type'] == 'linear':
            if isinstance(layer['output'], int):
                dims.append(layer['output'])
            else:
                dims.append(spectra_size)

            linear = nn.Linear(in_features=dims[-2], out_features=dims[-1])
            module.add_module(f'linear_{i}', linear)
            module.add_module(f'batch_norm_{i}', nn.BatchNorm1d(dims[-1]))
            module.add_module(f'SELU_{i}', nn.SELU())

        elif layer['type'] == 'reshape':
            dims.append(layer['output'][0])
            module.add_module(f'reshape_{i}', EmptyLayer())

        elif layer['type'] == 'convolutional':
            dims.append(layer['filters'])

            conv = nn.Conv1d(
                in_channels=dims[-2],
                out_channels=dims[-1],
                kernel_size=3,
                padding='same'
            )
            module.add_module(f'conv_{i}', conv)
            module.add_module(f'dropout_{i}', nn.Dropout1d(dropout_prob))
            module.add_module(f'batch_norm_{i}', nn.BatchNorm1d(dims[-1]))
            module.add_module(f'ELU_{i}', nn.ELU())

        elif layer['type'] == 'shortcut':
            dims.append(dims[-1] + dims[layer['layer']])
            module.add_module(f'shortcut_{i}', EmptyLayer())

        else:
            dims.append(layer['filters'])

            conv = nn.Conv1d(
                in_channels=dims[-2],
                out_channels=dims[-1],
                kernel_size=3,
                padding='same'
            )
            module.add_module(f'conv_{i}', conv)

            if layer['batch_norm']:
                module.add_module(f'batch_norm_{i}', nn.BatchNorm1d(dims[-1]))

            if layer['activation']:
                module.add_module(f'ELU_{i}', nn.ELU())

            module.add_module(f'pixel_shuffle_{i}', PixelShuffle1d(2))
            dims[-1] = int(dims[-1] / 2)

        module_list.append(module)

    return file['layers'], module_list


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


def weighted_mse(y: Tensor, y_hat: Tensor) -> Tensor:
    """
    Weighted MSE loss

    Parameters
    ----------
    y : Tensor
        Target data
    y_hat : Tensor
        Predicted data

    Returns
    -------
    Tensor
        Weighted MSE loss
    """
    weights = torch.abs(1 / y)
    return torch.mean(weights * (y_hat - y) ** 2) / torch.sum(weights)


def network_initialisation(
        val_frac: float,
        synth_path: str,
        labels_path: str,
        config_file: str,
        log_params: list,
        kwargs: dict,
        architecture: Type[Encoder | Decoder],
        model: PyXspecFitting,
        device: torch.device
) -> tuple[
    Encoder | Decoder,
    optim.Optimizer,
    optim.lr_scheduler.ReduceLROnPlateau,
    DataLoader,
    DataLoader
]:
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
    config_file : string
        Path to network config file
    log_params : list
        Index of each free parameter in logarithmic space
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
    tuple[Encoder | Decoder, Optimizer, lr_scheduler, DataLoader, DataLoader]
        Neural network, optimizer, scheduler and
        dataloaders for the training and validation datasets
    """
    learning_rate = 1e-5

    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(synth_path, labels_path, log_params)
    val_amount = int(len(dataset) * val_frac)

    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_amount, val_amount])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, **kwargs)

    # Initialise the CNN
    cnn = architecture(dataset[0][0].size(dim=0), config_file, model)
    cnn.to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True)

    print(f'{cnn.__class__.__name__}:\n'
          f'Training data size: {len(train_dataset)}\t'
          f'Validation data size: {len(val_dataset)}\n')

    return cnn, optimizer, scheduler, train_loader, val_loader


def plot_reconstructions(epoch: int, y_data: np.ndarray, y_recon: np.ndarray, axes: Axes):
    """
    Plots reconstructions for a given epoch

    Parameters
    ----------
    epoch : integer
        Epoch number
    y_data : ndarray
        Spectrum
    y_recon : ndarray
        Reconstructed Spectrum
    axes : Axes
        Plot axes
    """
    x_data = np.load('./data/spectra_x_axis.npy')

    if x_data.size % 2 != 0:
        x_data = np.append(x_data[:-2], np.mean(x_data[ -2:]))

    axes.set_title(f'Epoch: {epoch}', fontsize=16)
    axes.scatter(x_data, y_data, label='Spectrum')
    axes.scatter(x_data, y_recon, label='Reconstruction')
    axes.set_xlabel('Energy (keV)', fontsize=12)
    axes.set_ylabel(r'$log_{10}$ Counts ($s^{-1}$ $detector^{-1}$ $keV^{-1}$)', fontsize=12)
    axes.set_xscale('log')
    axes.legend(fontsize=16)


def train(
        device: torch.device,
        loader: DataLoader,
        cnn: Encoder | Decoder,
        optimizer: torch.optim.Optimizer) -> float:
    """
    Trains the encoder or decoder using cross entropy or mean squared error

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
            # loss = weighted_mse(spectra, output)

        # Optimise CNN
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate(
        device: torch.device,
        loader: DataLoader,
        cnn: Encoder | Decoder) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Validates the encoder or decoder using cross entropy or mean squared error

    Parameters
    ----------
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training

    Returns
    -------
    tuple[float, ndarray, ndarray]
        Average loss value, spectra & reconstructions
    """
    loss = 0
    cnn.eval()

    with torch.no_grad():
        for spectra, params, _ in loader:
            spectra = spectra.to(device)
            params = params.to(device)

            # Generate predictions and loss
            if isinstance(cnn, Encoder):
                output = cnn(spectra)
                loss += nn.CrossEntropyLoss()(output, params).item()
            else:
                output = cnn(params)
                loss += nn.MSELoss()(output, spectra).item()
                # loss += weighted_mse(spectra, output).item()

    return loss / len(loader), spectra.cpu().numpy(), output.cpu().numpy()


def test(
        log_params: list[int],
        device: torch.device,
        loader: DataLoader,
        cnn: Decoder,
        dirs: list[str]) -> float:
    """
    Tests the encoder or decoder using PyXspec or mean squared error

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

    # Initialize multiprocessing variables
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

            # Generate predictions and loss if network is a decoder
            if isinstance(cnn, Encoder):
                output = cnn(spectra)
                output[:, log_params] = 10 ** output[:, log_params]
                outputs = torch.vstack((outputs, output))
            else:
                params = params.to(device)
                output = cnn(params)
                loss += nn.MSELoss()(output, spectra).item()

    # Initialize processes for multiprocessing of encoder loss calculation
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

        # Start multiprocessing
        for process in processes:
            process.start()

        # End multiprocessing
        for process in processes:
            process.join()

        # Collect results
        losses = [queue.get() for _ in processes]

        os.chdir(dirs[0])

        return sum(losses) / len(losses)

    return loss / len(loader)


def main():
    """
    Main function for spectrum machine learning
    """
    # Variables
    num_epochs = 100
    val_frac = 0.1
    data_dir = '../../Documents/Nicer_Data/ethan'
    spectra_path = './data/preprocessed_spectra.npy'
    synth_path = './data/synth_spectra.npy'
    params_path = './data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    synth_params_path = './data/synth_spectra_params.npy'
    config_file = './decoder.json'
    fix_params = np.array([[4, 0], [5, 100]])
    log_params = [0, 2, 3, 4]

    # Constants
    train_loss = []
    val_loss = []
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Xspec initialization
    model = PyXspecFitting('tbabs(simplcutx(ezdiskbb))', fix_params)

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    decoder, d_optimizer, d_scheduler, d_train_loader, d_val_loader = network_initialisation(
        val_frac,
        synth_path,
        synth_params_path,
        config_file,
        log_params,
        kwargs,
        Decoder,
        model,
        device
    )

    # encoder, e_optimizer, e_scheduler, e_train_loader, e_val_loader = network_initialisation(
    #     val_frac,
    #     spectra_path,
    #     params_path,
    #     log_params,
    #     kwargs,
    #     Encoder,
    #     model,
    #     device
    # )

    _, axes = plt.subplots(4, 3, figsize=(24, 12), constrained_layout=True)
    axes = axes.flatten()
    plot_epochs = np.rint((np.arange(axes.size) + 1) * num_epochs / axes.size)

    # Train for each epoch
    for epoch in range(num_epochs):
        t_initial = time()

        val_loss.append(validate(device, d_val_loader, decoder)[0])
        d_scheduler.step(val_loss[-1])

        # Validate CNN
        # if epoch in plot_epochs:
        #     plot_reconstructions(
        #         epoch,
        #         spectrum,
        #         output,
        #         axes[np.argwhere(plot_epochs == epoch)[0, 0]]
        #     )

        # Train CNN
        train_loss.append(train(device, d_train_loader, decoder, d_optimizer))

        print(f'Epoch [{epoch + 1}/{num_epochs}]\t'
              f'Training loss: {train_loss[-1]:.3e}\t'
              f'Validation loss: {val_loss[-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    loss, spectra, outputs = validate(device, d_val_loader, decoder)
    val_loss.append(loss)
    print(f'\nFinal validation loss: {val_loss[-1]:.3e}')

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    for i in range(axes.size):
        plot_reconstructions(num_epochs, spectra[i], outputs[i], axes[i])

    plt.savefig('./plots/Reconstructions.png')

    plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.yscale('log')
    plt.legend(fontsize=20)
    plt.savefig('./plots/Epoch_Loss.png')


if __name__ == '__main__':
    main()
