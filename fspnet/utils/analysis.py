"""
Calculates the saliency of decoders or autoencoders using backpropagation
"""
import re

import torch
import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader

from fspnet.utils.network import Network
from fspnet.utils.data import load_params
from fspnet.utils.utils import get_device


def autoencoder_saliency(
        loader: DataLoader,
        encoder: Network,
        decoder: Network) -> tuple[ndarray, ndarray, ndarray]:
    """
    Calculates the importance of each part of the input spectrum on the output spectrum
    by calculating the saliency using backpropagation of the autoencoder

    Parameters
    ----------
    loader : DataLoader
        Autoencoder validation data loader
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
    spectra = next(iter(loader))[0][:8].to(get_device()[1])

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


def decoder_saliency(loader: DataLoader, decoder: Network):
    """
    Calculates the importance of each parameter on the output spectrum
    by calculating the saliency using backpropagation of the decoder

    Parameters
    ----------
    loader : DataLoader
        Decoder validation data loader
    decoder : Network
        Decoder half of the network
    """
    # Constants
    saliency = []
    device = get_device()[1]

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
        f"\nParameter impact on decoder:\n{[f'{x:.2f}' for x in params_impact]}"
        f"\nSaliency spread:\n{[f'{x:.2f}' for x in params_std]}\n"
    )


def param_comparison(data_paths: tuple[str, str]) -> tuple[ndarray, ndarray]:
    """
    Gets and transforms data to compare parameter values

    Parameters
    ----------
    data_paths : tuple[string, string]
        Paths to the parameters to compare

    Returns
    -------
    tuple[ndarray, ndarray]
        Target parameters and parameter predictions
    """
    names = []
    params = []

    # Load config parameters
    for data_path in data_paths:
        returns = load_params(data_path, load_kwargs={'dtype': str})
        names.append(returns[0])
        params.append(returns[1])

    # Sort for longest dataset first
    sort_idx = np.argsort([param.shape[0] for param in params])[::-1]
    params = [params[i] for i in sort_idx]
    names = [names[i] for i in sort_idx]

    # Sort target spectra by name
    name_sort_idx = np.argsort(names[0])
    names[0] = names[0][name_sort_idx]
    params[0] = params[0][name_sort_idx]

    # Filter target parameters using spectra that was predicted and log parameters
    target_idx = np.searchsorted(names[0], names[1])
    params[0] = params[0][target_idx]
    shuffle_idx = np.random.permutation(params[0].shape[0])
    params[0] = params[0][shuffle_idx]
    params[1] = params[1][shuffle_idx]

    params = [params[i] for i in sort_idx]

    return np.swapaxes(params[0], 0, 1), np.swapaxes(params[1], 0, 1)


def linear_weights(network: Network) -> ndarray:
    """
    Returns the mapping of all linear weights from the input to the output


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
        if re.search(r'.*linear_\d+.weight', name):
            weights.append(param.detach().cpu().numpy())

    # Calculate weight mapping for each input parameter
    for param_weight in np.swapaxes(weights[0], 0, 1):
        # Loop through each layer except the first and multiple weight mapping between layers
        for i, layer_weights in enumerate(weights[1:]):
            # If second layer, multiply by input weight, otherwise multiple all weight mappings
            if i == 0:
                weight = np.sum(param_weight * layer_weights, axis=1)
            else:
                weight = np.sum(weight * layer_weights, axis=1)

        param_weights.append(weight)

    return np.array(param_weights)
