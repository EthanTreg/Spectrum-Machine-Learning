from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from torch import nn

from src.utils import layer_utils

if TYPE_CHECKING:
    from src.utils.networks import Network


def create_network(
        input_size: int,
        output_size: int,
        config_path: str) -> tuple[list[dict], nn.ModuleList]:
    """
    Creates a network from a config file

    Parameters
    ----------
    input_size : integer
        Size of the input
    output_size : integer
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
    kwargs = {
        'data_size': input_size,
        'output_size': output_size,
        'dims': [input_size],
        'dropout_prob': file['net']['dropout_prob'],
    }
    module_list = nn.ModuleList()

    # Create layers
    for i, layer in enumerate(file['layers']):
        kwargs['i'] = i
        kwargs['module'] = nn.Sequential()

        kwargs = getattr(layer_utils, layer['type'])(kwargs, layer)

        module_list.append(kwargs['module'])

    return file['layers'], module_list


def load_network(
        load_num: int,
        states_dir: str,
        cnn: Network) -> tuple[int, Network, tuple[list, list]] | None:
    """
    Loads the network from a previously saved state

    Can account for changes in the network

    Parameters
    ----------
    load_num : integer
        File number of the saved state
    states_dir : string
        Directory to the save files
    cnn : Network
        The network to append saved state to

    Returns
    -------
    tuple[int, Encoder | Decoder, Optimizer, ReduceLROnPlateau, tuple[list, list]]
        The initial epoch, the updated network, optimizer
        and scheduler, and the training and validation losses
    """
    try:
        d_state = torch.load(f'{states_dir}{cnn.name}_{load_num}.pth')
    except FileNotFoundError:
        print(f'ERROR: {states_dir}{cnn.name}_{load_num}.pth does not exist')
        return None

    # Updates some saved states with the new network for compatibility
    d_state['optimizer']['param_groups'][0]['params'] = \
        cnn.optimizer.state_dict()['param_groups'][0]['params']
    d_state['scheduler']['best'] = cnn.scheduler.state_dict()['best']
    old_keys = d_state['state_dict'].copy().keys()

    # Remove layers not present in the new network
    for i, key in enumerate(old_keys):
        if key not in cnn.state_dict().keys():
            del d_state['state_dict'][key]
            del d_state['optimizer']['state'][i]

    # Apply the saved states to the new network
    initial_epoch = d_state['epoch']
    cnn.load_state_dict(cnn.state_dict() | d_state['state_dict'])
    cnn.optimizer.load_state_dict(d_state['optimizer'])
    cnn.scheduler.load_state_dict(d_state['scheduler'])
    train_loss = d_state['train_loss']
    val_loss = d_state['val_loss']

    return initial_epoch, cnn, (train_loss, val_loss)
