"""
Calculates the saliency of decoders or autoencoders using backpropagation
"""
import torch
from torch.utils.data import DataLoader

from fspnet.utils.network import Network


def autoencoder_saliency(
        loader: DataLoader,
        device: torch.device,
        encoder: Network,
        decoder: Network) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    tuple[Tensor, Tensor, Tensor]
        Original spectra, predicted spectra, and saliency
    """
    # Constants
    spectra = next(iter(loader))[0][:8].to(device)

    # Initialization
    encoder.train()
    spectra.requires_grad_()

    # Calculate saliency through backpropagation
    output = decoder(encoder(spectra))
    loss = torch.nn.MSELoss()(output, spectra)
    loss.backward()
    saliency = spectra.grad.data.abs().cpu()

    spectra = spectra.cpu().detach()
    output = output.cpu().detach()
    saliency = saliency.cpu()

    return spectra, output, saliency


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
    d_spectra, d_parameters, *_ = next(iter(loader))

    # Initialization
    decoder.train()
    d_spectra = d_spectra.to(device)
    d_parameters = d_parameters.to(device)
    d_parameters.requires_grad_()

    # Calculate saliency through backpropagation
    d_output = decoder(d_parameters)
    d_loss = torch.nn.MSELoss()(d_output, d_spectra)
    d_loss.backward()
    d_saliency = d_parameters.grad.data.abs().cpu()

    # Measure impact of input parameters on decoder output
    parameter_saliency = torch.mean(d_saliency, dim=0)
    parameter_impact = parameter_saliency / torch.min(parameter_saliency)
    parameter_std = torch.std(d_saliency, dim=0) / torch.min(parameter_saliency)

    print(
        f'\nParameter impact on decoder:\n{parameter_impact.tolist()}'
        f'\nParameter spread:\n{parameter_std.tolist()}\n'
    )
