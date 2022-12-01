import torch
from torch import nn, Tensor


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
    GRU wrapper for compatibility with network & can handle output of bidirectional GRUs

    Attributes
    ----------
    bidirectional : boolean
        If GRU is bidirectional
    method : string, default = None
        If GRU is bidirectional, how to handle output.
        Can be either sum, mean or None. If None, concatenation is used

    Methods
    -------
    forward(x)
        Returns
    """
    def __init__(self, bidirectional: bool, method: str = None):
        """
        Parameters
        ----------
        bidirectional : bool
            If GRU is bidirectional
        method : string, default = None
            If GRU is bidirectional, how to handle output.
            Can be either sum, mean or None. If None, concatenation is used
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.method = method

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
        output = x[0]

        if self.bidirectional:
            output = output.view(*output.size()[:2], -1, 2)

            if self.method == 'sum':
                output = torch.sum(output, dim=-1)
            elif self.method == 'mean':
                output = torch.mean(output, dim=-1)

        return output


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


def linear(kwargs: dict, layer: dict) -> dict:
    """
    Linear layer constructor

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), data size (data_size),
        dimension list (dims) & sequential module (module).
        Must contain output_size if layer uses factor rather than features
    layer : dictionary
        Must contain either factor of output size or features

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
    kwargs['module'].add_module(f"SELU_{kwargs['i']}", nn.SELU())

    # Data size equals number of nodes
    kwargs['data_size'] = kwargs['dims'][-1]

    return kwargs


def convolutional(kwargs: dict, layer: dict) -> dict:
    """
    Convolutional layer constructor

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), dropout probability (dropout_prob),
        dimension list (dims), sequential module (module)
    layer : dictionary
        Must contain number of filters (filters).
        Can contain kernel size (kernel), else 3 is used
        and/or batch normalisation (batch_norm) else it isn't used

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(layer['filters'])

    # Kernel size is optional with a default of 3
    try:
        kernel_size = layer['kernel']
    except KeyError:
        kernel_size = 3

    conv = nn.Conv1d(
        in_channels=kwargs['dims'][-2],
        out_channels=kwargs['dims'][-1],
        kernel_size=kernel_size,
        padding='same',
        padding_mode='replicate',
    )

    kwargs['module'].add_module(f"conv_{kwargs['i']}", conv)
    kwargs['module'].add_module(
        f"dropout_{kwargs['i']}",
        nn.Dropout1d(kwargs['dropout_prob'])
    )

    # Optional batch normalization layer
    try:
        if layer['batch_norm']:
            kwargs['module'].add_module(
                f"batch_norm_{kwargs['i']}",
                nn.BatchNorm1d(kwargs['dims'][-1])
            )
    except KeyError:
        pass

    kwargs['module'].add_module(f"ELU_{kwargs['i']}", nn.ELU())

    return kwargs


def gru(kwargs: dict, _) -> dict:
    """
    Gated recurrent unit (GRU) layer constructor

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), data size (data_size),
        sequential module (module)
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    gru_layer = nn.GRU(
        input_size=kwargs['data_size'],
        hidden_size=kwargs['data_size'],
        num_layers=2,
        batch_first=True,
    )

    kwargs['module'].add_module(f"GRU_{kwargs['i']}", gru_layer)
    kwargs['module'].add_module(
        f"GRU_output_{kwargs['i']}",
        GRUOutput(False)
    )
    kwargs['module'].add_module(f"ELU_{kwargs['i']}", nn.ELU())

    return kwargs


def linear_upscale(kwargs: dict, _: dict) -> dict:
    """
    Constructs a 2x upscaler using a linear layer

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), data size (data_size),
        dimension list (dims), sequential module (module)
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    linear_layer = nn.Linear(in_features=kwargs['data_size'], out_features=kwargs['data_size'] * 2)

    kwargs['module'].add_module(f"reshape_{kwargs['i']}", Reshape([-1]))
    kwargs['module'].add_module(f"linear_{kwargs['i']}", linear_layer)
    kwargs['module'].add_module(f"SELU_{kwargs['i']}", nn.SELU())
    kwargs['module'].add_module(f"reshape_{kwargs['i']}", Reshape([1, -1]))

    # Data size doubles
    kwargs['data_size'] *= 2
    kwargs['dims'].append(1)

    return kwargs


def conv_upscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a 2x upscaler using a convolutional layer and pixel shuffling

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), data size (data_size),
        dimension list (dims), sequential module (module)
    layer : dictionary
        Must contain number of filters (filters).
        Can contain batch normalisation (batch_norm) else it isn't used
        and/or activation layer (activation) else it isn't used

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(layer['filters'])

    conv = nn.Conv1d(
        in_channels=kwargs['dims'][-2],
        out_channels=kwargs['dims'][-1],
        kernel_size=3,
        padding='same',
    )
    kwargs['module'].add_module(f"conv_{kwargs['i']}", conv)

    # Optional batch normalization layer
    try:
        if layer['batch_norm']:
            kwargs['module'].add_module(
                f"batch_norm_{kwargs['i']}",
                nn.BatchNorm1d(kwargs['dims'][-1])
            )
    except KeyError:
        pass

    # Optional activation layer
    try:
        if layer['activation']:
            kwargs['module'].add_module(f"ELU_{kwargs['i']}", nn.ELU())
    except KeyError:
        pass

    # Upscaling done using pixel shuffling
    kwargs['module'].add_module(f"pixel_shuffle_{kwargs['i']}", PixelShuffle1d(2))
    kwargs['dims'][-1] = int(kwargs['dims'][-1] / 2)

    # Data size doubles
    kwargs['data_size'] *= 2

    return kwargs


def conv_transpose(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a 2x upscaler using a transpose convolutional layer with fractional stride

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), data size (data_size), dropout probability (dropout_prob),
        dimension list (dims), sequential module (module)
    layer : dictionary
        Must contain number of filters (filters)

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
    kwargs['module'].add_module(
        f"dropout_{kwargs['i']}",
        nn.Dropout1d(kwargs['dropout_prob'])
    )
    kwargs['module'].add_module(f"ELU_{kwargs['i']}", nn.ELU())

    # Data size doubles
    kwargs['data_size'] *= 2

    return kwargs


def upsample(kwargs: dict, _: dict) -> dict:
    """
    Constructs a 2x upscaler using linear upsampling

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), data size (data_size), sequential module (module)
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['module'].add_module(
        f"upsample_{kwargs['i']}",
        nn.Upsample(scale_factor=2, mode='linear')
    )
    # Data size doubles
    kwargs['data_size'] *= 2

    return kwargs


def conv_depth_downscale(kwargs: dict, _: dict) -> dict:
    """
    Constructs depth downscaler using convolution with kernel size of 1

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), dimension list (dims), sequential module (module)
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(1)

    conv = nn.Conv1d(
        in_channels=kwargs['dims'][-2],
        out_channels=kwargs['dims'][-1],
        kernel_size=1,
        padding='same'
    )

    kwargs['module'].add_module(f"conv_downscale_{kwargs['i']}", conv)
    kwargs['module'].add_module(f"ELU_{kwargs['i']}", nn.ELU())

    return kwargs


def conv_downscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a convolutional layer with stride 2 for 2x downscaling

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), dropout probability (dropout_prob),
        dimension list (dims), sequential module (module)
    layer : dictionary
        Must contain number of filters (filters).
        Can contain batch normalisation (batch_norm) else it isn't used

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(layer['filters'])

    conv = nn.Conv1d(
        in_channels=kwargs['dims'][-2],
        out_channels=kwargs['dims'][-1],
        kernel_size=3,
        stride=2,
        padding=1,
        padding_mode='replicate',
    )

    kwargs['module'].add_module(f"conv_{kwargs['i']}", conv)
    kwargs['module'].add_module(
        f"dropout_{kwargs['i']}",
        nn.Dropout1d(kwargs['dropout_prob'])
    )

    # Optional batch normalization layer
    try:
        if layer['batch_norm']:
            kwargs['module'].add_module(
                f"batch_norm_{kwargs['i']}",
                nn.BatchNorm1d(kwargs['dims'][-1])
            )
    except KeyError:
        pass

    kwargs['module'].add_module(f"ELU_{kwargs['i']}", nn.ELU())

    # Data size halves
    kwargs['data_size'] = int(kwargs['data_size'] / 2)

    return kwargs


def pool(kwargs: dict, _: dict) -> dict:
    """
    Constructs a max pooling layer for 2x downscaling

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), data size (data_size),
        dimension list (dims), sequential module (module)
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
    kwargs['data_size'] = int(kwargs['data_size'] / 2)

    return kwargs


def reshape(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a reshaping layer to change the data dimensions

    Parameters
    ----------
    kwargs : dictionary
        Must contain layer number (i), data size (data_size),
        dimension list (dims), sequential module (module)
    layer : dictionary
        Must contain output dimensions (output)

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    # If reshape reduces the number of dimensions
    if len(layer['output']) == 1:
        kwargs['dims'].append(kwargs['data_size'] * kwargs['dims'][-1])

        # Data size equals the previous size multiplied by the previous dimension
        kwargs['data_size'] = kwargs['dims'][-1]
    else:
        kwargs['dims'].append(layer['output'][0])

        # Data size equals the previous size divided by the first shape dimension
        kwargs['data_size'] = int(kwargs['dims'][-2] / kwargs['dims'][-1])

    kwargs['module'].add_module(
        f"reshape_{kwargs['i']}",
        Reshape(layer['output'])
    )

    return kwargs


def concatenate(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a concatenation layer to combine the outputs from two layers

    Parameters
    ----------
    kwargs : dictionary
        Must contain dimension list (dims)
    layer : dictionary
        Must contain layer to concatenate the output with (layer)

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1] + kwargs['dims'][layer['layer']])
    return kwargs


def shortcut(kwargs: dict, _: dict) -> dict:
    """
    Constructs a shortcut layer to add the outputs from two layers

    Parameters
    ----------
    kwargs : dictionary
        Must contain dimension list (dims)
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1])
    return kwargs
