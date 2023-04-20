"""
Calculates the saliency of decoders or autoencoders using backpropagation
"""
import re
import pickle

import torch
import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader

from fspnet.utils.network import Network
from fspnet.utils.utils import open_config


def autoencoder_saliency(
        loader: DataLoader,
        device: torch.device,
        encoder: Network,
        decoder: Network) -> tuple[ndarray, ndarray, ndarray]:
    """
    Calculates the importance of each part of the input spectrum on the output spectrum
    by calculating the saliency using backpropagation of the autoencoder

    Parameters
    ----------
    loader : DataLoader
        Autoencoder validation data loader
    device : device
        Which device type PyTorch should use
    encoder : Network
        Encoder half of the network
    decoder : Network
        Decoder half of the network

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Original spectra, output, and saliency
    """
    # Constants
    spectra = next(iter(loader))[0][:8].to(device)

    # Initialization
    torch.backends.cudnn.enabled = False
    encoder.eval()
    decoder.eval()
    spectra.requires_grad_()

    # Calculate saliency through backpropagation
    output = decoder(encoder(spectra))
    loss = torch.nn.MSELoss()(output, spectra)
    loss.backward()
    saliency = spectra.grad.data.abs().cpu()
    torch.backends.cudnn.enabled = True

    return spectra.detach().cpu().numpy(), output.detach().cpu().numpy(), saliency.numpy()


def decoder_saliency(loader: DataLoader, device: torch.device, decoder: Network):
    """
    Calculates the importance of each parameter on the output spectrum
    by calculating the saliency using backpropagation of the decoder

    Parameters
    ----------
    loader : DataLoader
        Decoder validation data loader
    device : device
        Which device type PyTorch should use
    decoder : Network
        Decoder half of the network
    """
    # Constants
    saliency = []

    # Initialization
    torch.backends.cudnn.enabled = False
    decoder.eval()

    for spectra, params, *_ in loader:
        spectra = spectra.to(device)
        params = params.to(device)
        params.requires_grad_()

        # Calculate saliency through backpropagation
        output = decoder(params)
        torch.zeros([1, output.size(-1)])
        loss = torch.nn.MSELoss()(output, spectra)
        loss.backward()
        saliency.append(params.grad.data.abs())

    torch.backends.cudnn.enabled = True

    # Measure impact of input parameters on decoder output
    saliency = torch.cat(saliency).cpu()
    saliency_params = torch.mean(saliency, dim=0)
    params_impact = saliency_params / torch.min(saliency_params)
    params_std = torch.std(saliency, dim=0) / torch.min(saliency_params)

    print(
        f'\nParameter impact on decoder:\n{params_impact.tolist()}'
        f'\nSaliency spread:\n{params_std.tolist()}\n'
    )


def param_comparison(config: dict | str = '../config.yaml') -> tuple[ndarray, ndarray]:
    """
    Gets and transforms data to compare parameter values

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or file path to the configuration file

    Returns
    -------
    tuple[ndarray, ndarray]
        Target parameters and parameter predictions
    """
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', config)

    # Load config parameters
    data_path = config['data']['encoder-data-path']
    prediction_path = config['output']['parameter-predictions-path']
    log_params = config['model']['log-parameters']

    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    target = np.array(data['params'])
    predictions = np.loadtxt(prediction_path, delimiter=',', dtype=str)

    if 'names' in data:
        spectra_names = np.array(data['names'])
    else:
        spectra_names = np.arange(target.shape[0], dtype=float).astype(str)

    # Sort target spectra by name
    sort_idx = np.argsort(spectra_names)
    spectra_names = spectra_names[sort_idx]
    target = target[sort_idx]

    # Filter target parameters using spectra that was predicted and log parameters
    target_idx = np.searchsorted(spectra_names, predictions[:, 0])
    target = target[target_idx]
    predictions = predictions[:, 1:].astype(float)

    if log_params:
        target[:, log_params] = np.log10(target[:, log_params])
        predictions[:, log_params] = np.log10(predictions[:, log_params])

    return target, predictions


def linear_weights(network: Network) -> ndarray:
    """
    Returns the mapping of the weights from the lowest dimension
    to a high dimension for the 3 smallest linear layers


    If the low dimension is 5 and the high dimension is 240,
    with two hidden dimensions of 60 and 120, then the output will be 5x240
    where each row is the strength of the input on each output
    calculated by the multiplication of every weight that comes from the input
    and connects to the output

    Parameters
    ----------
    network : Network
        Network to learn the mapping for

    Returns
    -------
    ndarray
        Mapping of low dimension to high dimension
    """
    weights = []
    param_weights = []

    # Get all linear weights in the network
    for name, param in network.named_parameters():
        if re.search(r'.*linear_\d.weight', name):
            weights.append(param)

    weights_idx = np.argsort([weight.numel() for weight in weights])
    weights = [weights[idx].detach().cpu().numpy() for idx in weights_idx]

    for weights_1l in np.swapaxes(weights[0], 1, 0):
        weights_i = []
        weights_1l = np.repeat(weights_1l[np.newaxis], weights[1].shape[0], axis=0)

        for weights_3i in weights[2]:
            weights_3i = np.repeat(weights_3i[:, np.newaxis], weights[1].shape[1], axis=1)
            weights_i.append(np.sum(weights_1l * weights_3i * weights[1]))

        param_weights.append(weights_i)

    return np.array(param_weights)
