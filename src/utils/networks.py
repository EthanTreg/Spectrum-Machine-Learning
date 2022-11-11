import torch
from torch import nn, Tensor

from src.utils.utils import PyXspecFitting
from src.utils.network_utils import create_network


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

    def forward(self, x: Tensor) -> Tensor:
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
    phase : float
        Blend constant between progressive learning
    layers : list[dictionary]
        Layers with layer parameters
    network : ModuleList
        Network construction

    Methods
    -------
    forward(x)
        Forward pass of decoder
    """
    def __init__(self, spectra_size: int, num_params: int, config_file: str):
        """
        Parameters
        ----------
        spectra_size : integer
            number of data points in spectra
        num_params : integer
            Number of input parameters
        config_file : string
            Path to the config file
        """
        super().__init__()
        self.phase = 0

        # Construct layers in CNN
        self.layers, self.network = create_network(
            num_params,
            spectra_size,
            config_file,
        )

    def forward(self, x: Tensor) -> Tensor:
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
            # Concatenation layers
            if layer['type'] == 'concatenate':
                x = torch.cat((x, outputs[layer['layer']]), dim=1)
            # Shortcut layers
            if layer['type'] == 'shortcut':
                x = self.phase * x + (1 - self.phase) * outputs[layer['layer']]
            # All other layers
            else:
                x = self.network[i](x)

            outputs.append(x)

        return torch.squeeze(x, dim=1)
