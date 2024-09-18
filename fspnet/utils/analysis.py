"""
Calculates the saliency of decoders or autoencoders using backpropagation
"""
import re

import torch
import numpy as np
from numpy import ndarray
from netloader.utils.utils import get_device
from torch.utils.data import DataLoader
from torch.nn import Module

from fspnet.utils.multiprocessing import check_cpus, mpi_multiprocessing


def autoencoder_saliency(
        loader: DataLoader,
        net: Module) -> tuple[ndarray, ndarray, ndarray]:
    """
    Calculates the importance of each part of the input spectrum on the output spectrum
    by calculating the saliency using backpropagation of the autoencoder

    Parameters
    ----------
    loader : DataLoader
        Autoencoder validation data loader
    net : Module
        Autoencoder

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Original spectra, output, and saliency
    """
    # Constants
    spectra = next(iter(loader))[-1][:8].to(get_device()[1])

    # Initialization
    torch.backends.cudnn.enabled = False
    net.eval()
    spectra.requires_grad_()

    # Calculate saliency through backpropagation
    output = net(spectra)
    loss = torch.nn.MSELoss()(output, spectra)
    loss.backward()
    saliency = spectra.grad.data.abs().cpu()
    torch.backends.cudnn.enabled = True

    return spectra.detach().cpu().numpy(), output.detach().cpu().numpy(), saliency.numpy()


def decoder_saliency(loader: DataLoader, decoder: Module):
    """
    Calculates the importance of each parameter on the output spectrum
    by calculating the saliency using backpropagation of the decoder

    Parameters
    ----------
    loader : DataLoader
        Decoder validation data loader
    decoder : Module
        Decoder half of the network
    """
    # Constants
    saliency = []
    device = get_device()[1]

    # Initialization
    torch.backends.cudnn.enabled = False
    decoder.eval()

    for _, params, spectra, *_ in loader:
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


def linear_weights(net: Module) -> ndarray:
    """
    Returns the mapping of all linear weights from the input to the output


    If the low dimension is 5 and the high dimension is 240,
    with two hidden dimensions of 60 and 120, then the output will be 5x240
    where each row is the strength of the input on each output
    calculated by the multiplication of every weight that comes from the input
    and connects to the output

    Parameters
    ----------
    net : Module
        Network to learn the mapping for

    Returns
    -------
    ndarray
        Mapping of low dimension to high dimension
    """
    weights = []
    param_weights = []

    # Get all linear weights in the network
    for name, param in net.named_parameters():
        if re.search(r'.*linear.weight', name):
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


def pyxspec_test(
        worker_dir: str,
        names: ndarray,
        params: ndarray,
        cpus: int = 1,
        job_name: str | None = None,
        python_path: str = 'python3') -> None:
    """
    Calculates the PGStat loss using PyXspec
    Done using multiprocessing if > 2 cores available

    Parameters
    ----------
    worker_dir : str
        Directory to where to save worker data
    names : ndarray
        Files names of the FITS spectra corresponding to the parameters
    params : ndarray
        Parameter predictions
    cpus : int, default = 1
        Number of threads to use, 0 will use all available
    job_name : str, default = None
        If not None, file name to save the output to
    python_path : str, default = python3
        Path to the python executable if using virtual environments
    """
    i: int
    data: list[ndarray] = []
    worker_names: list[ndarray]
    worker_params: list[ndarray]
    job: ndarray
    data_: ndarray
    names_batch: ndarray
    params_batch: ndarray

    # Divide work between workers
    cpus = check_cpus(cpus)
    worker_names = np.array_split(names, cpus)
    worker_params = np.array_split(params, cpus)

    # Save data to file for each worker
    for i, (names_batch, params_batch) in enumerate(zip(worker_names, worker_params)):
        job = np.hstack((np.expand_dims(names_batch, axis=1), params_batch))
        np.savetxt(f'{worker_dir}worker_{i}_job.csv', job, delimiter=',', fmt='%s')

    # Run workers to calculate PGStat
    mpi_multiprocessing(
        cpus,
        len(names),
        f'fspnet.utils.pyxspec_worker {worker_dir}',
        python_path=python_path,
    )

    # Retrieve worker outputs
    for i in range(cpus):
        data.append(np.loadtxt(f'{worker_dir}worker_{i}_job.csv', delimiter=',', dtype=str))

    data_ = np.concatenate(data)

    # If job_name is provided, save all worker data to file
    if job_name:
        np.savetxt(f'{worker_dir}{job_name}.csv', data_, delimiter=',', fmt='%s')

    # Median loss
    print(f'Reduced PGStat Loss: {np.median(data_[:, -1].astype(float)):.3e}')
