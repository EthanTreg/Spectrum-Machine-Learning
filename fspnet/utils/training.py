"""
Trains the network and evaluates the performance
"""
from time import time

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from fspnet.utils.network import Network
from fspnet.utils.multiprocessing import check_cpus, mpi_multiprocessing


def _xspec_loss(
        worker_dir: str,
        names: list[str],
        params: np.ndarray,
        cpus: int = 1,
        job_name: str = None) -> float:
    """
    Calculates the PGStat loss using PyXspec
    Done using multiprocessing if >2 cores available

    Parameters
    ----------
    worker_dir : string
        Directory to where to save worker data
    names : list[string]
        Spectra names
    params : ndarray
        Parameter predictions from CNN
    cpus : integer, default = 1
        Number of threads to use, 0 will use all available
    job_name : string, default = None
        If not None, file name to save the output to

    Returns
    -------
    float
        Average loss
    """
    # Initialize variables
    data = []

    cpus = check_cpus(cpus)

    # Divide work between workers
    worker_names = np.array_split(names, cpus)
    worker_params = np.array_split(params, cpus)

    # Save data to file for each worker
    for i, (names_batch, params_batch) in enumerate(zip(worker_names, worker_params)):
        job = np.hstack((np.expand_dims(names_batch, axis=1), params_batch))
        np.savetxt(f'{worker_dir}worker_{i}_job.csv', job, delimiter=',', fmt='%s')

    # Run workers to calculate PGStat
    mpi_multiprocessing(cpus, len(names), f'fspnet.utils.pyxspec_worker {worker_dir}')

    # Retrieve worker outputs
    for i in range(cpus):
        data.append(np.loadtxt(f'{worker_dir}worker_{i}_job.csv', delimiter=',', dtype=str))

    data = np.concatenate(data)

    # If job_name is provided, save all worker data to file
    if job_name:
        np.savetxt(f'{worker_dir}{job_name}.csv', data, delimiter=',', fmt='%s')

    # Median loss
    return float(np.median(data[:, -1].astype(float)))


def pyxspec_test(
        worker_dir: str,
        loader: DataLoader,
        cpus: int = 1,
        job_name: str = None,
        defaults: torch.Tensor = None,
        device: torch.device = None,
        encoder: Network = None):
    """
    If encoder is provided, creates predictions for each spectrum, otherwise, uses true parameters
    then calculates loss

    Parameters
    ----------
    worker_dir : string
        Directory to where to save worker data
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cpus : integer, default = 1
        Number of threads to use, 0 will use all available
    job_name : string, default = None
        If not None, file name to save the output to
    defaults : Tensor, default = None
        If default parameters should be used
    device : device, default = None
        Which device type PyTorch should use, not needed if encoder is None
    encoder : Network, default = None
        Encoder to predict parameters, if None, true parameters will be used
    """
    # Initialize variables
    initial_time = time()
    names = []
    params = []
    log_params = loader.dataset.dataset.log_params
    param_transform = loader.dataset.dataset.transform[1]

    if encoder:
        encoder.eval()

    # Initialize processes for multiprocessing of encoder loss calculation
    with torch.no_grad():
        for data in loader:
            names.extend(data[-1])

            # Use defaults if provided, else if encoder is provided, generate predictions,
            # else get true parameters
            if defaults is not None:
                param_batch = defaults.repeat(data[0].size(0), 1)
            elif encoder:
                param_batch = encoder(data[0].to(device)).cpu()
            else:
                param_batch = data[1]

            # Transform parameters if not defaults
            if defaults is None:
                param_batch = param_batch * param_transform[1] + param_transform[0]
                param_batch[:, log_params] = 10 ** param_batch[:, log_params]

            params.append(param_batch)

    params = torch.cat(params)
    print(f'Parameter retrieval time: {time() - initial_time:.3e} s')

    loss = _xspec_loss(worker_dir, names, params.numpy(), cpus=cpus, job_name=job_name)
    print(f'Reduced PGStat Loss: {loss:.3e}')


def train_val(
        device: torch.device,
        loader: DataLoader,
        cnn: Network,
        train: bool = True,
        surrogate: Network = None) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Trains the encoder or decoder using cross entropy or mean squared error

    Parameters
    ----------
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : Network
        CNN to use for training/validation
    train : bool, default = True
        If network should be trained or validated
    surrogate : Network, defualt = None
        Surrogate network for encoder training

    Returns
    -------
    tuple[float, ndarray, ndarray]
        Average loss value, spectra & reconstructions
    """
    epoch_loss = 0

    if train:
        cnn.train()

        if surrogate:
            surrogate.train()
    else:
        cnn.eval()

        if surrogate:
            surrogate.eval()

    with torch.set_grad_enabled(train):
        for spectra, params, *_ in loader:
            spectra = spectra.to(device)

            # If surrogate is not none, train encoder with surrogate
            if surrogate:
                output = surrogate(cnn(spectra))
                target = spectra
            else:
                params = params.to(device)

                # Train encoder with supervision or decoder
                if cnn.encoder:
                    output = cnn(spectra)
                    target = params
                else:
                    output = cnn(params)
                    target = spectra

            loss = nn.MSELoss()(output, target)

            if train:
                # Optimise CNN
                cnn.optimizer.zero_grad()
                loss.backward()
                cnn.optimizer.step()

            epoch_loss += loss.item()

    return epoch_loss / len(loader), spectra.cpu().numpy(), output.detach().cpu().numpy()


def training(
        epochs: tuple[int, int],
        loaders: tuple[DataLoader, DataLoader],
        cnn: Network,
        device: torch.device,
        save_num: int = 0,
        states_dir: str = None,
        losses: tuple[list, list] = None,
        surrogate: Network = None) -> tuple[tuple[list, list], np.ndarray, np.ndarray]:
    """
    Trains & validates the network for each epoch

    Parameters
    ----------
    epochs : tuple[integer, integer]
        Initial epoch & number of epochs to train
    loaders : tuple[DataLoader, DataLoader]
        Train and validation dataloaders
    cnn : Network
        CNN to use for training
    device : device
        Which device type PyTorch should use
    save_num : integer, default = 0
        The file number to save the new state, if 0, nothing will be saved
    states_dir : string, default = None
        Path to the folder where the network state will be saved, not needed if save_num = 0
    losses : tuple[list, list], default = ([], [])
        Train and validation losses for each epoch, can be empty
    surrogate : Network, default = None
        Surrogate network to use for training

    Returns
    -------
    tuple[tuple[list, list], ndarray, ndarray]
        Train & validation losses, spectra & reconstructions
    """
    if not losses:
        losses = ([], [])

    # Train for each epoch
    for epoch in range(*epochs):
        t_initial = time()
        epoch += 1

        # Train CNN
        losses[0].append(train_val(device, loaders[0], cnn, surrogate=surrogate)[0])

        # Validate CNN
        losses[1].append(train_val(device, loaders[1], cnn, train=False, surrogate=surrogate)[0])
        cnn.scheduler.step(losses[1][-1])

        # Save training progress
        if save_num:
            state = {
                'epoch': epoch,
                'transform': loaders[0].dataset.dataset.transform,
                'train_loss': losses[0],
                'val_loss': losses[1],
                'indices': loaders[0].dataset.dataset.indices,
                'state_dict': cnn.state_dict(),
                'optimizer': cnn.optimizer.state_dict(),
                'scheduler': cnn.scheduler.state_dict(),
            }

            torch.save(state, f'{states_dir}{cnn.name}_{save_num}.pth')

        print(f'Epoch [{epoch}/{epochs[1]}]\t'
              f'Training loss: {losses[0][-1]:.3e}\t'
              f'Validation loss: {losses[1][-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    # Final validation
    loss, spectra, outputs = train_val(device, loaders[1], cnn, train=False, surrogate=surrogate)
    losses[1].append(loss)
    print(f'\nFinal validation loss: {losses[1][-1]:.3e}')

    return losses, spectra, outputs
