"""
Implements several layer types to be loaded into a network
"""
import torch
from torch import nn, Tensor

from fspnet.utils.utils import get_device


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
        return x.contiguous().view(x.size(0), *self.shape)


class RecurrentOutput(nn.Module):
    """
    GRU wrapper for compatibility with network & can handle output of bidirectional GRUs

    Attributes
    ----------
    bidirectional : string, default = None
        If GRU is bidirectional, and if so, what method to use,
        can be either mean, sum or concatenation,
        if None, GRU is mono-directional and concatenate will be used

    Methods
    -------
    forward(x)
        Returns
    """
    def __init__(self, recurrent_layer: nn.Module, bidirectional: str = None):
        """
        Parameters
        ----------
        bidirectional : string, default = None
            If GRU is bidirectional, and if so, what method to use,
            can be either mean, sum or concatenate,
            if None, GRU is mono-directional and concatenation will be used
        """
        super().__init__()
        self.options = [None, 'sum', 'mean', 'concatenate']
        self.bidirectional = bidirectional
        self.recurrent_layer = recurrent_layer

        if self.bidirectional not in self.options:
            raise ValueError(
                f'{self.bidirectional} is not a valid bidirectional method, options: {self.options}'
            )

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
        x = self.recurrent_layer(torch.transpose(x, 1, 2))[0]

        if self.bidirectional and self.bidirectional != self.options[3]:
            x = x.view(*x.size()[:2], 2, -1)

            if self.bidirectional == self.options[1]:
                x = torch.sum(x, dim=-2)
            elif self.bidirectional == self.options[2]:
                x = torch.mean(x, dim=-2)

        return torch.transpose(x, 1, 2)


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


class Sample(nn.Module):
    """
    Samples random values from a Gaussian distribution for a variational autoencoder

    Attributes
    ----------
    sample_layer : Module
        Layer to sample values from a Gaussian distribution

    Methods
    -------
    forward(x)
        Forward pass of the sampling layer
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        device = get_device()[1]

        self.mean_layer = nn.Linear(in_features=in_features, out_features=out_features)
        self.std_layer = nn.Linear(in_features=in_features, out_features=out_features)
        self.sample_layer = torch.distributions.Normal(
            torch.tensor(0.).to(device),
            torch.tensor(1.).to(device),
        ).sample

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the sampling layer for a variational autoencoder

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        tuple[Tensor, Tensor]
            Output tensor and KL Divergence loss
        """
        mean = self.mean_layer(x)
        std = torch.exp(self.std_layer(x))
        x = mean + std * self.sample_layer(mean.shape)
        kl_loss = 0.5 * torch.mean(mean ** 2 + std ** 2 - 2 * torch.log(std) - 1)

        return x, kl_loss


def _optional_layer(
        default: bool,
        arg: str,
        kwargs: dict,
        layer: dict,
        layer_func: nn.Module):
    """
    Implements an optional layer for a parent layer to use

    Parameters
    ----------
    default : boolean
        If the layer should be used by default
    arg : string
        Argument for the user to call this layer
    kwargs : dictionary
        kwargs dictionary used by the parent
    layer : dictionary
        layer dictionary used by the parent
    layer_func : Module
        Optional layer to add to the network
    """
    if (arg in layer and layer[arg]) or (arg not in layer and default):
        kwargs['module'].add_module(f"{type(layer_func).__name__}_{kwargs['i']}", layer_func)


def linear(kwargs: dict, layer: dict) -> dict:
    """
    Linear layer constructor

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
        output_size : integer, optional
            Size of the network's output, required only if layer contains factor and not features;
        dropout_prob : float, optional
            Probability of dropout if dropout from layer is True;
    layer : dictionary
        factor : float, optional
            Output features is equal to the factor of the network's output,
            will be used if provided, else features will be used;
        features : integer, optional
            Number of output features for the layer,
            if output_size from kwargs and factor is provided, features will not be used;
        dropout : boolean, default = False
            If dropout should be used;
        activation : boolean, default = True
            If SELU activation should be used;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    # Number of features can be defined by either a factor of the output size or explicitly
    try:
        kwargs['dims'].append(int(kwargs['output_size'] * layer['factor']))
    except KeyError:
        kwargs['dims'].append(layer['features'])

    linear_layer = nn.Linear(in_features=kwargs['dims'][-2], out_features=kwargs['dims'][-1])
    kwargs['module'].add_module(f"linear_{kwargs['i']}", linear_layer)

    # Optional layers
    _optional_layer(False, 'dropout', kwargs, layer, nn.Dropout1d(kwargs['dropout_prob']))
    _optional_layer(False, 'batch_norm', kwargs, layer, nn.BatchNorm1d(kwargs['dims'][-1]))
    _optional_layer(True, 'activation', kwargs, layer, nn.SELU())

    # Data size equals number of nodes
    kwargs['data_size'].append(kwargs['dims'][-1])

    return kwargs


def convolutional(kwargs: dict, layer: dict) -> dict:
    """
    Convolutional layer constructor

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
        dropout_prob : float, optional
            Probability of dropout, not required if dropout from layer is False;
    layer : dictionary
        filters : integer
            Number of convolutional filters;
        dropout : boolean, default = True
            If dropout should be used;
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;
        kernel : integer, default = 3
            Size of the kernel;
        stride : integer, default = 1
            Stride of the kernel;
        padding : integer | string, default = 'same'
            Input padding, can an integer or 'same' where 'same' preserves the input shape;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kernel_size = 3
    stride = 1
    padding = 'same'
    kwargs['dims'].append(layer['filters'])

    # Optional parameters
    if 'kernel' in layer:
        kernel_size = layer['kernel']

    if 'stride' in layer:
        stride = layer['stride']

    if 'padding' in layer:
        padding = layer['padding']

    conv = nn.Conv1d(
        in_channels=kwargs['dims'][-2],
        out_channels=kwargs['dims'][-1],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode='replicate',
    )
    kwargs['module'].add_module(f"conv_{kwargs['i']}", conv)

    # Optional layers
    _optional_layer(True, 'dropout', kwargs, layer, nn.Dropout1d(kwargs['dropout_prob']))
    _optional_layer(False, 'batch_norm', kwargs, layer, nn.BatchNorm1d(kwargs['dims'][-1]))
    _optional_layer(True, 'activation', kwargs, layer, nn.ELU())

    if padding != 'same':
        kwargs['data_size'].append(int(
            (kwargs['data_size'][-1] + 2 * padding - kernel_size) / stride + 1
        ))
    else:
        kwargs['data_size'].append(kwargs['data_size'][-1])

    return kwargs


def recurrent(kwargs: dict, layer: dict) -> dict:
    """
    Gated recurrent unit (GRU) layer constructor

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
        dropout_prob : float, optional
            Probability of dropout, only required if layers from layer > 1;
    layer : dictionary
        dropout : boolean, default = True
            If dropout should be used;
        activation : boolean, default = True
            If ELU activation should be used;
        layers : integer, default = 2
            Number of stacked GRU layers;
        factor : float, default = 1
            Output size of the layer depending on the network's output size;
        bidirectional : string, default = None
            If a bidirectional GRU should be used and method for combining the two directions,
            can be sum, mean or concatenation;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    try:
        layers = layer['layers']
    except KeyError:
        layers = 2

    try:
        kwargs['dims'].append(layer['filters'])
    except KeyError:
        kwargs['dims'].append(1)

    try:
        bidirectional = layer['bidirectional']

        if bidirectional == 'None':
            bidirectional = None
    except KeyError:
        bidirectional = None

    if layers > 1 and (('dropout' in layer and layer['dropout']) or 'dropout' not in layer):
        dropout_prob = kwargs['dropout_prob']
    else:
        dropout_prob = 0

    recurrent_kwargs = {
        'input_size': kwargs['dims'][-2],
        'hidden_size': kwargs['dims'][-1],
        'num_layers': layers,
        'batch_first': True,
        'dropout': dropout_prob,
        'bidirectional': bidirectional is not None,
    }

    if bidirectional == 'concatenate':
        kwargs['dims'][-1] *= 2

    try:
        if layer['method'] == 'rnn':
            recurrent_layer = nn.RNN(**recurrent_kwargs, nonlinearity='relu')
        elif layer['method'] == 'lstm':
            recurrent_layer = nn.LSTM(**recurrent_kwargs)
        else:
            recurrent_layer = nn.GRU(**recurrent_kwargs)
    except KeyError:
        recurrent_layer = nn.GRU(**recurrent_kwargs)

    kwargs['module'].add_module(
        f"GRU_output_{kwargs['i']}",
        RecurrentOutput(recurrent_layer, bidirectional=bidirectional)
    )

    _optional_layer(True, 'activation', kwargs, layer, nn.ELU())
    _optional_layer(False, 'batch_norm', kwargs, layer, nn.BatchNorm1d(kwargs['dims'][-1]))

    kwargs['data_size'].append(kwargs['data_size'][-1])

    return kwargs


def linear_upscale(kwargs: dict, _: dict) -> dict:
    """
    Constructs a 2x upscaler using a linear layer,
    combines reshape for use within convolutional layers

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    linear_layer = nn.Linear(
        in_features=kwargs['data_size'][-1],
        out_features=kwargs['data_size'][-1] * 2,
    )

    kwargs['module'].add_module(f"reshape_{kwargs['i']}", Reshape([-1]))
    kwargs['module'].add_module(f"linear_{kwargs['i']}", linear_layer)
    kwargs['module'].add_module(f"SELU_{kwargs['i']}", nn.SELU())
    kwargs['module'].add_module(f"reshape_{kwargs['i']}", Reshape([1, -1]))

    # Data size doubles
    kwargs['data_size'].append(kwargs['data_size'][-1] * 2)
    kwargs['dims'].append(1)

    return kwargs


def conv_upscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a 2x upscaler using a convolutional layer and pixel shuffling

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer], optional
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    layer : dictionary
        filters : integer
            Number of convolutional filters, must be a multiple of 2;
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;
        kernel : integer, default = 3
            Size of the kernel;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    layer['dropout'] = False
    layer['stride'] = 1
    layer['padding'] = 'same'
    kwargs = convolutional(kwargs, layer)

    # Upscaling done using pixel shuffling
    kwargs['module'].add_module(f"pixel_shuffle_{kwargs['i']}", PixelShuffle1d(2))
    kwargs['dims'][-1] = int(kwargs['dims'][-1] / 2)

    # Data size doubles
    kwargs['data_size'].append(kwargs['data_size'][-1] * 2)

    return kwargs


def conv_transpose(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a 2x upscaler using a transpose convolutional layer with fractional stride

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
        dropout_prob : float, optional
            Probability of dropout, not required if dropout from layer is False;
    layer : dictionary
        filters : integer
            Number of convolutional filters;
        dropout : boolean, default = True
            If dropout should be used;
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(layer['filters'])

    conv = nn.ConvTranspose1d(
        in_channels=kwargs['dims'][-2],
        out_channels=kwargs['dims'][-1],
        kernel_size=2,
        stride=2,
    )

    kwargs['module'].add_module(f"conv_transpose_{kwargs['i']}", conv)

    # Optional layers
    _optional_layer(True, 'dropout', kwargs, layer, nn.Dropout1d(kwargs['dropout_prob']))
    _optional_layer(False, 'batch_norm', kwargs, layer, nn.BatchNorm1d(kwargs['dims'][-1]))
    _optional_layer(True, 'activation', kwargs, layer, nn.ELU())

    # Data size doubles
    kwargs['data_size'].append(kwargs['data_size'][-1] * 2)

    return kwargs


def upsample(kwargs: dict, _: dict) -> dict:
    """
    Constructs a 2x upscaler using linear upsampling

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1])
    kwargs['module'].add_module(
        f"upsample_{kwargs['i']}",
        nn.Upsample(scale_factor=2, mode='linear')
    )
    # Data size doubles
    kwargs['data_size'].append(kwargs['data_size'][-1] * 2)

    return kwargs


def conv_depth_downscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs depth downscaler using convolution with kernel size of 1

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    layer : dictionary
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    layer['dropout'] = False
    layer['filters'] = 1
    layer['kernel'] = 1
    layer['padding'] = 'same'

    convolutional(kwargs, layer)

    return kwargs


def conv_downscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a convolutional layer with stride 2 for 2x downscaling

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
        dropout_prob : float, optional
            Probability of dropout, not required if dropout from layer is False;
    layer : dictionary
        filters : integer
            Number of convolutional filters;
        dropout : boolean, default = True
            If dropout should be used;
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    layer['kernel'] = 3
    layer['stride'] = 2
    layer['padding'] = 1

    convolutional(kwargs, layer)

    return kwargs


def pool(kwargs: dict, _: dict) -> dict:
    """
    Constructs a max pooling layer for 2x downscaling

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(int(kwargs['dims'][-1]))

    kwargs['module'].add_module(f"pool_{kwargs['i']}", nn.MaxPool1d(kernel_size=2))

    # Data size halves
    kwargs['data_size'].append(int(kwargs['data_size'][-1] / 2))

    return kwargs


def reshape(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a reshaping layer to change the data dimensions

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    layer : dictionary
        output : integer | tuple[integer, integer]
            Output dimensions of input tensor, ignoring the first dimension (batch size)

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    # If reshape reduces the number of dimensions
    if len(layer['output']) == 1:
        kwargs['dims'].append(kwargs['data_size'][-1] * kwargs['dims'][-1])

        # Data size equals the previous size multiplied by the previous dimension
        kwargs['data_size'].append(kwargs['dims'][-1])
    else:
        kwargs['dims'].append(layer['output'][0])

        # Data size equals the previous size divided by the first shape dimension
        kwargs['data_size'].append(int(kwargs['data_size'][-1] / kwargs['dims'][-1]))

    kwargs['module'].add_module(
        f"reshape_{kwargs['i']}",
        Reshape(layer['output'])
    )

    return kwargs


def sample(kwargs: dict, layer: dict) -> dict:
    """
    Generates mean and standard deviation and randomly samples from a Gaussian distribution
    for a variational autoencoder

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
        output_size : integer, optional
            Size of the network's output, required only if layer contains factor and not features;
    layer : dictionary
        factor : float, optional
            Output features is equal to the factor of the network's output,
            will be used if provided, else features will be used;
        features : integer, optional
            Number of output features for the layer,
            if output_size from kwargs and factor is provided, features will not be used;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    # Number of features can be defined by either a factor of the output size or explicitly
    try:
        kwargs['dims'].append(int(kwargs['output_size'] * layer['factor']))
    except KeyError:
        kwargs['dims'].append(layer['features'])

    kwargs['data_size'].append(kwargs['dims'][-1])
    kwargs['module'].add_module(
        f"sample_{kwargs['i']}",
        Sample(kwargs['dims'][-2], kwargs['dims'][-1]),
    )
    return kwargs


def extract(kwargs: dict, layer: dict) -> dict:
    """
    Extracts a number of values from the tensor, returning two tensors

    Parameters
    ----------
    kwargs : dictionary
        data_size : integer
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    layer : dictionary
        number : integer
            Number of values to extract from the previous layer

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1] - layer['number'])
    kwargs['data_size'].append(kwargs['dims'][-1])
    return kwargs


def clone(kwargs: dict, _: dict) -> dict:
    """
    Constructs a layer to clone a number of values from the previous layer

    Parameters
    ----------
    kwargs : dictionary
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1])
    kwargs['data_size'].append(kwargs['data_size'][-1])
    return kwargs


def concatenate(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a concatenation layer to combine the outputs from two layers

    Parameters
    ----------
    kwargs : dictionary
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    layer : dictionary
        layer : integer
            Layer index to concatenate the previous layer output with

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1] + kwargs['dims'][layer['layer']])
    kwargs['data_size'].append(kwargs['data_size'][-1])
    return kwargs


def shortcut(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a shortcut layer to add the outputs from two layers

    Parameters
    ----------
    kwargs : dictionary
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    layer : dictionary
        layer : integer
            Layer index to add the previous layer output with;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    if kwargs['dims'][-1] == 1:
        kwargs['dims'].append(kwargs['dims'][layer['layer']])
    else:
        kwargs['dims'].append(kwargs['dims'][-1])

    kwargs['data_size'].append(kwargs['data_size'][-1])
    return kwargs


def skip(kwargs: dict, layer: dict) -> dict:
    """
    Bypasses previous layers by retrieving the output from the defined layer

    Parameters
    ----------
    kwargs : dictionary
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    layer : dictionary
        layer : integer
            Layer index to retrieve the output;
    """
    kwargs['dims'].append(kwargs['dims'][layer['layer']])
    kwargs['data_size'].append(kwargs['data_size'][layer['layer']])
    return kwargs
