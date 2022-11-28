import torch
from torch import nn, optim, Tensor

from src.utils.network_utils import create_network


class Network(nn.Module):
    """
    Constructs a CNN network from a configuration file

    Attributes
    ----------
    encoder : boolean
        If network is an encoder
    layers : list[dictionary]
        Layers with layer parameters
    network : ModuleList
        Network construction
    optimizer : Optimizer
        Network optimizer
    scheduler : ReduceLROnPlateau
        Optimizer scheduler

    Methods
    -------
    forward(x)
        Forward pass of CNN
    """
    def __init__(
            self,
            spectra_size: int,
            params_size: int,
            learning_rate: float,
            name: str,
            config_dir: str):
        """
        Parameters
        ----------
        spectra_size : int
            Size of the input tensor
        params_size : int
            Size of the output tensor
        learning_rate : float
            Optimizer initial learning rate
        name : str
            Name of the network
        config_dir : string
            Path to the network config directory
        """
        super().__init__()
        self.name = name

        # If network is an encoder
        if 'Encoder' in name:
            self.encoder = True
            input_size = spectra_size
            output_size = params_size
        else:
            self.encoder = False
            input_size = params_size
            output_size = spectra_size

        # Construct layers in CNN
        self.layers, self.network = create_network(
            input_size,
            output_size,
            f'{config_dir}{name}.json',
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            verbose=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the CNN

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Output tensor from the CNN
        """
        outputs = []

        for i, layer in enumerate(self.layers):
            # Concatenation layers
            if layer['type'] == 'concatenate':
                x = torch.cat((x, outputs[layer['layer']]), dim=1)
            # Shortcut layers
            if layer['type'] == 'shortcut':
                x = x + outputs[layer['layer']]
            # All other layers
            else:
                x = self.network[i](x)

            outputs.append(x)

        return x
