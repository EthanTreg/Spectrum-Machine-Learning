from __future__ import annotations

import json
from typing import Type, TYPE_CHECKING

import torch
from torch import nn, optim, Tensor

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
            module.add_module(f'batch_norm_{i}', nn.BatchNorm1d(dims[-1]))
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
] | None:
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
    try:
        d_state = torch.load(f'{states_dir}{type(cnn).__name__}_{load_num}.pth')
    except FileNotFoundError:
        print(f'ERROR: {states_dir}{type(cnn).__name__}_{load_num}.pth does not exist')
        return None

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


def network_initialisation(
        spectra_size: int,
        learning_rate: float,
        model_args: tuple,
        architecture: Type[Encoder | Decoder],
        device: torch.device = 'cpu',
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
    device : device, default = cpu
        Which device type PyTorch should use

    Returns
    -------
    tuple[Encoder | Decoder, Optimizer, lr_scheduler]
        Neural network, optimizer, scheduler and
    """
    # Initialise the CNN
    cnn = architecture(spectra_size, *model_args).to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True)

    return cnn, optimizer, scheduler
