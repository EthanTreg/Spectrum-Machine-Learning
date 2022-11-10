from __future__ import annotations

import json
from typing import Type, TYPE_CHECKING
from multiprocessing import Queue, Value

import xspec
import torch
import numpy as np
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader, random_split

if TYPE_CHECKING:
    from src.utils.networks import Encoder, Decoder


class Reshape(nn.Module):
    """
    Used for reshaping tensors within a neural network

    Attributes
    shape : list[integer]
        Desired shape of the output tensor, ignoring first dimension

    Methods
    -------
    forward(x)
        Forward pass of Reshape
    """

    def __init__(self, shape: list[int]):
        """
        Parameters
        ----------
        shape : list[integer]
            Desired shape of the output tensor, ignoring first dimension
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of reshaping tensors

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Output tensor
        """
        return x.view(x.size(0), *self.shape)


class GRUOutput(nn.Module):
    """
    GRU wrapper for compatibility with network

    Methods
    -------
    forward(x)
        Returns
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the GRU

        Parameters
        ----------
        x : Tensor, shape (N, 1, L)
            Input tensor

        Returns
        -------
        Tensor, shape (N, 1, L)
            Output tensor
        """
        return x[0]


class PixelShuffle1d(nn.Module):
    """
    Used for upscaling by scale factor r for an input (*, C x r, L) to an output (*, C, L x r)

    Equivalent to torch.nn.PixelShuffle but for 1D

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

    def forward(self, x: Tensor) -> Tensor:
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
        self.params = (self.params - np.min(self.params, axis=0)) / \
                      (np.max(self.params, axis=0) - np.min(self.params, axis=0))
        # self.params = (self.params - np.mean(self.params, axis=0)) / np.std(self.params, axis=0)
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
    fit_loss(total, names, params, counter, queue)
        Evaluates the loss using PyXspec and each predicted parameter
    """
    def __init__(self, model: str, fix_params: np.ndarray):
        """
        Parameters
        ----------
        model : string
            Model to use for PyXspec
        fix_params : ndarray
            Parameter number & value of fixed parameters
        """
        super().__init__()
        self.fix_params = fix_params
        self.param_limits = np.empty((0, 2))

        # PyXspec initialization
        xspec.Xset.chatter = 0
        xspec.Xset.logChatter = 0
        xspec.AllModels.lmod('simplcutx', dirPath='../../../Documents/Xspec_Models/simplcutx')
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

    def fit_loss(
            self,
            total: int,
            names: list[str],
            params: np.ndarray,
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
        counter : Value
            Number of spectra fitted
        queue : Queue
            Multiprocessing queue to add PGStat loss
        """
        loss = 0

        # Loop through each spectrum in the batch
        for i, name in enumerate(names):
            # Limit parameters
            param_min = self.param_limits[:, 0] + 1e-6 * (
                    self.param_limits[:, 1] - self.param_limits[:, 0])
            param_max = self.param_limits[:, 1] - 1e-6 * (
                    self.param_limits[:, 1] - self.param_limits[:, 0])

            params = np.clip(params, a_min=param_min, a_max=param_max)

            # Load spectrum
            xspec.Spectrum(name)

            # Calculate fit statistic
            loss += self.fit_statistic(params[i])

            xspec.AllData.clear()

            # Increase progress
            with counter.get_lock():
                counter.value += 1

            progress_bar(counter.value, total)

        # Average loss of batch
        queue.put(loss / len(names))


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
        input_size: int,
        spectra_size: int,
        config_path: str) -> tuple[list[dict], nn.ModuleList]:
    """
    Creates a network from a config file

    Parameters
    ----------
    input_size : integer
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
    # Load network configuration file
    with open(config_path, 'r', encoding='utf-8') as file:
        file = json.load(file)

    # Initialize variables
    data_size = input_size
    dims = [input_size]
    dropout_prob = file['net']['dropout_prob']
    module_list = nn.ModuleList()

    # Create layers
    for i, layer in enumerate(file['layers']):
        module = nn.Sequential()

        # Linear layer
        if layer['type'] == 'linear':
            dims.append(int(spectra_size * layer['factor']))

            linear = nn.Linear(in_features=dims[-2], out_features=dims[-1])
            module.add_module(f'linear_{i}', linear)
            # module.add_module(f'batch_norm_{i}', nn.BatchNorm1d(dims[-1]))
            module.add_module(f'SELU_{i}', nn.SELU())

            # Data size equals number of nodes
            data_size = dims[-1]

        # Convolutional layer
        elif layer['type'] == 'convolutional':
            dims.append(layer['filters'])

            conv = nn.Conv1d(
                in_channels=dims[-2],
                out_channels=dims[-1],
                kernel_size=3,
                padding='same',
                padding_mode='replicate',
            )

            module.add_module(f'conv_{i}', conv)
            module.add_module(f'dropout_{i}', nn.Dropout1d(dropout_prob))
            # module.add_module(f'batch_norm_{i}', nn.BatchNorm1d(dims[-1]))
            module.add_module(f'ELU_{i}', nn.ELU())

        # GRU layer
        elif layer['type'] == 'GRU':
            gru = nn.GRU(
                input_size=data_size,
                hidden_size=data_size,
                num_layers=2,
                batch_first=True
            )

            module.add_module(f'GRU_{i}', gru)
            module.add_module(f'GRU_output_{i}', GRUOutput())
            module.add_module(f'ELU_{i}', nn.ELU())

        # Linear upscaling
        elif layer['type'] == 'linear_upscale':
            linear = nn.Linear(in_features=data_size, out_features=data_size * 2)

            module.add_module(f'reshape_{i}', Reshape([-1]))
            module.add_module(f'linear_{i}', linear)
            module.add_module(f'SELU_{i}', nn.SELU())
            module.add_module(f'reshape_{i}', Reshape([1, -1]))

            # Data size doubles
            data_size *= 2
            dims.append(1)

        # Pixel shuffle convolutional upscaling
        elif layer['type'] == 'conv_upscale':
            dims.append(layer['filters'])

            conv = nn.Conv1d(
                in_channels=dims[-2],
                out_channels=dims[-1],
                kernel_size=3,
                padding='same',
            )
            module.add_module(f'conv_{i}', conv)

            # Optional batch norm layers
            if layer['batch_norm']:
                module.add_module(f'batch_norm_{i}', nn.BatchNorm1d(dims[-1]))

            # Optional activation layers
            if layer['activation']:
                module.add_module(f'ELU_{i}', nn.ELU())

            # Upscaling done using pixel shuffling
            module.add_module(f'pixel_shuffle_{i}', PixelShuffle1d(2))
            dims[-1] = int(dims[-1] / 2)

            # Data size doubles
            data_size *= 2

        # Transpose convolutional upscaling
        elif layer['type'] == 'conv_transpose':
            dims.append(layer['filters'])

            conv = nn.ConvTranspose1d(
                in_channels=dims[-2],
                out_channels=dims[-1],
                kernel_size=2,
                stride=2,
            )

            module.add_module(f'conv_transpose_{i}', conv)
            module.add_module(f'dropout_{i}', nn.Dropout1d(dropout_prob))
            module.add_module(f'ELU_{i}', nn.ELU())

            # Data size doubles
            data_size *= 2

        # Upscaling by upsampling
        elif layer['type'] == 'upsample':
            module.add_module(f'upsample_{i}', nn.Upsample(scale_factor=2, mode='linear'))
            # Data size doubles
            data_size *= 2

        # Depth downscale for compatibility using convolution with kernel size of 1
        elif layer['type'] == 'conv_downscale':
            dims.append(1)

            conv = nn.Conv1d(
                in_channels=dims[-2],
                out_channels=dims[-1],
                kernel_size=1,
                padding='same'
            )

            module.add_module(f'conv_downscale_{i}', conv)
            module.add_module(f'ELU_{i}', nn.ELU())

        # Reshape data layer
        elif layer['type'] == 'reshape':
            dims.append(layer['output'][0])
            module.add_module(f'reshape_{i}', Reshape(layer['output']))

            # Data size equals the previous size divided by the first shape dimension
            data_size = int(dims[-2] / dims[-1])

        # Shortcut layer or skip connection using concatenation
        elif layer['type'] == 'concatenate':
            dims.append(dims[-1] + dims[layer['layer']])

        # Shortcut layer or skip connection using addition
        elif layer['type'] == 'shortcut':
            dims.append(dims[-1])

        # Unknown layer
        else:
            print(f"\nERROR: Layer {layer['type']} not supported\n")

        module_list.append(module)

    return file['layers'], module_list


def load_network(
        load_num: int,
        states_dir: str,
        cnn: Encoder | Decoder,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau
) -> tuple[
    int,
    Encoder | Decoder,
    optim.Optimizer,
    optim.lr_scheduler.ReduceLROnPlateau,
    list,
    list
]:
    """
    Loads the network from a previously saved state

    Can account for changes in the network

    Parameters
    ----------
    load_num : integer
        File number of the saved state
    states_dir : string
        Directory to the save files
    cnn : Decoder | Encoder
        The network to append saved state to
    optimizer : Optimizer
        The optimizer to append saved state to
    scheduler : ReduceLROnPlateau
        The scheduler to append saved state to

    Returns
    -------
    tuple[int, Encoder | Decoder, Optimizer, ReduceLROnPlateau, list, list]
        The initial epoch, the updated network, optimizer
        and scheduler, and the training and validation losses
    """
    d_state = torch.load(f'{states_dir}{type(cnn).__name__}_{load_num}.pth')

    # Updates some saved states with the new network for compatibility
    d_state['optimizer']['param_groups'][0]['params'] = \
        optimizer.state_dict()['param_groups'][0]['params']
    d_state['scheduler']['best'] = scheduler.state_dict()['best']
    old_keys = d_state['state_dict'].copy().keys()

    # Remove layers not present in the new network
    for i, key in enumerate(old_keys):
        if key not in cnn.state_dict().keys():
            del d_state['state_dict'][key]
            del d_state['optimizer']['state'][i]

    # Apply the saved states to the new network
    initial_epoch = d_state['epoch']
    cnn.load_state_dict(cnn.state_dict() | d_state['state_dict'])
    optimizer.load_state_dict(d_state['optimizer'])
    scheduler.load_state_dict(d_state['scheduler'])
    train_loss = d_state['train_loss']
    val_loss = d_state['val_loss']

    return initial_epoch, cnn, optimizer, scheduler, train_loss, val_loss


def data_initialisation(
        val_frac: float,
        spectra_path: str,
        labels_path: str,
        log_params: list,
        kwargs: dict) -> tuple[DataLoader, DataLoader]:
    """
    Initialises training and validation data

    Parameters
    ----------
    val_frac : float
        Fraction of validation data
    spectra_path : string
        Path to synthetic data
    labels_path : string
        Path to labels
    log_params : list
        Index of each free parameter in logarithmic space
    kwargs : dictionary
        Keyword arguments for dataloader

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

    return train_loader, val_loader


def network_initialisation(
        spectra_size: int,
        learning_rate: float,
        model_args: tuple,
        architecture: Type[Encoder | Decoder],
        device: torch.device
) -> tuple[
    Encoder | Decoder,
    optim.Optimizer,
    optim.lr_scheduler.ReduceLROnPlateau
]:
    """
    Initialises neural network for either encoder or decoder CNN

    Parameters
    ----------
    spectra_size : integer
        Size of the spectra
    learning_rate : float
        Learning rate of the optimizer
    model_args : tuple
        Arguments for the network
    architecture : Encoder | Decoder
        Which network architecture to use
    device : device
        Which device type PyTorch should use

    Returns
    -------
    tuple[Encoder | Decoder, Optimizer, lr_scheduler]
        Neural network, optimizer, scheduler and
    """
    # Initialise the CNN
    cnn = architecture(spectra_size, *model_args)
    cnn.to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True)

    return cnn, optimizer, scheduler
