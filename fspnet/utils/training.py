"""
Trains the network and evaluates the performance
"""
from time import time

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from fspnet.utils.network import Network
from fspnet.utils.utils import get_device
from fspnet.utils.data import load_params
from fspnet.utils.multiprocessing import check_cpus, mpi_multiprocessing


def pyxspec_test(
        worker_dir: str,
        params: str | tuple[np.ndarray, np.ndarray],
        cpus: int = 1,
        job_name: str = None):
    """
    Calculates the PGStat loss using PyXspec
    Done using multiprocessing if > 2 cores available

    Parameters
    ----------
    worker_dir : string
        Directory to where to save worker data
    params : string | tuple[ndarray, ndarray]
        Path to the parameter predictions or tuple of spectra file names and parameter predictions
    cpus : integer, default = 1
        Number of threads to use, 0 will use all available
    job_name : string, default = None
        If not None, file name to save the output to
    """
    # Initialize variables
    data = []
    cpus = check_cpus(cpus)

    # Load parameters if file path provided
    if isinstance(params, str):
        names, params = load_params(params, load_kwargs={'dtype': str})
    else:
        names, params = params

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
    print(f'Reduced PGStat Loss: {np.median(data[:, -1].astype(float)):.3e}')


def train_val(
        loader: DataLoader,
        network: Network,
        train: bool = True,
        surrogate: Network = None) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Trains the encoder or decoder using cross entropy or mean squared error

    Parameters
    ----------
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    network : Network
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
    device = get_device()[1]

    if train:
        network.train()

        if surrogate:
            surrogate.train()
    else:
        network.eval()

        if surrogate:
            surrogate.eval()

    with torch.set_grad_enabled(train):
        for spectra, params, *_ in loader:
            loss = torch.tensor(0.).to(device)
            spectra = spectra.to(device)
            params = params.to(device)

            # If surrogate is not none, train encoder with surrogate
            if surrogate:
                output = network(spectra)
                loss += network.latent_mse_weight * nn.MSELoss()(output, params)
                output = surrogate(output)
                target = spectra
            else:
                # Train encoder with supervision or decoder
                if 'encoder' in network.name.lower():
                    output = network(spectra)
                    target = params
                else:
                    output = network(params)
                    target = spectra

            loss += nn.MSELoss()(output, target) + network.kl_loss.to(device)

            if train:
                # Optimise CNN
                network.optimizer.zero_grad()
                loss.backward()
                network.optimizer.step()

            epoch_loss += loss.item()

    return epoch_loss / len(loader), spectra.cpu().numpy(), output.detach().cpu().numpy()


def training(
        epochs: tuple[int, int],
        loaders: tuple[DataLoader, DataLoader],
        network: Network,
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
    network : Network
        CNN to use for training
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
        losses[0].append(train_val(loaders[0], network, surrogate=surrogate)[0])

        # Validate CNN
        losses[1].append(train_val(loaders[1], network, train=False, surrogate=surrogate)[0])
        network.scheduler.step(losses[1][-1])

        # Save training progress
        if save_num:
            state = {
                'epoch': epoch,
                'transform': loaders[0].dataset.dataset.transform,
                'train_loss': losses[0],
                'val_loss': losses[1],
                'indices': loaders[0].dataset.dataset.indices,
                'state_dict': network.state_dict(),
                'optimizer': network.optimizer.state_dict(),
                'scheduler': network.scheduler.state_dict(),
            }

            torch.save(state, f'{states_dir}{network.name}_{save_num}.pth')

        print(f'Epoch [{epoch}/{epochs[1]}]\t'
              f'Training loss: {losses[0][-1]:.3e}\t'
              f'Validation loss: {losses[1][-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    # Final validation
    loss, spectra, outputs = train_val(loaders[1], network, train=False, surrogate=surrogate)
    losses[1].append(loss)
    print(f'\nFinal validation loss: {losses[1][-1]:.3e}')

    return losses, spectra, outputs
