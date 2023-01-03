from time import time
from multiprocessing import Process, Queue, Value

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from src.utils.networks import Network
from src.utils.utils import PyXspecFitting


def xspec_loss(
        log_params: list[int],
        loader: DataLoader,
        model: PyXspecFitting,
        params: np.ndarray = None) -> float:
    """
    Calculates the PGStat loss using PyXspec

    Parameters
    ----------
    log_params : list[integer]
        Indices of parameters in logarithmic space
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    model : PyXspecFitting
        PyXspec model for fit evaluation
    params : ndarray, default = None
        Parameters to measure performance using Xspec,
        if None, parameters from dataloader will be used

    Returns
    -------
    float
        Average loss value
    """
    # Initialize variables
    processes = []
    batch = len(next(iter(loader))[2])
    param_transform = loader.dataset.dataset.transform[1]
    counter = Value('i', 0)
    queue = Queue()

    # Initialize processes for multiprocessing of encoder loss calculation
    for i, data in enumerate(loader):
        if params is None:
            param_batch = data[1]
        else:
            param_batch = params[batch * i:batch * (i + 1)]

        param_batch = param_batch * param_transform[1] + param_transform[0]
        param_batch[:, log_params] = 10 ** param_batch[:, log_params]

        processes.append(Process(target=model.fit_loss, args=(
            len(loader.dataset),
            data[2],
            param_batch,
            counter,
            queue,
        )))

    # Start multiprocessing
    for process in processes:
        process.start()

    # End multiprocessing
    for process in processes:
        process.join()

    # Collect results
    losses = [queue.get() for _ in processes]

    return sum(losses) / len(losses)


def encoder_test(
        log_params: list[int],
        device: torch.device,
        loader: DataLoader,
        cnn: Network,
        model: PyXspecFitting) -> float:
    """
    Tests the encoder using PyXspec

    Parameters
    ----------
    log_params : list[integer]
        Indices of parameters in logarithmic space
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : Network
        Model to use for training
    model : PyXspecFitting
        PyXspec model for fit evaluation

    Returns
    -------
    float
        Average loss value
    """
    cnn.eval()

    # Initialize multiprocessing variables
    spectra_count = 0
    loader_output = next(iter(loader))
    outputs = torch.empty((0, loader_output[1].size(1))).to(device)
    initial_time = time()

    with torch.no_grad():
        for spectra, _, _ in loader:
            spectra = spectra.to(device)
            spectra_count += spectra.size(0)

            # Generate parameter predictions
            output = cnn(spectra)
            outputs = torch.vstack((outputs, output))

    # Transform outputs from normalized to real values
    outputs = outputs.cpu().numpy()
    final_time = time() - initial_time
    print(f'\nParameter prediction time: {final_time:.3f} s'
          f'\tSpectra per second: {spectra_count / final_time:.2e}')

    loss = xspec_loss(log_params, loader, model, params=outputs)

    return loss


def validate(
        device: torch.device,
        loader: DataLoader,
        cnn: Network,
        surrogate: Network = None) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Validates the encoder or decoder using cross entropy or mean squared error

    Parameters
    ----------
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training
    surrogate : Network, default = None
        Surrogate network for encoder training

    Returns
    -------
    tuple[float, ndarray, ndarray]
        Average loss value, spectra & reconstructions
    """
    loss = 0
    cnn.eval()

    if surrogate:
        surrogate.eval()

    with torch.no_grad():
        for spectra, params, _ in loader:
            spectra = spectra.to(device)

            # If surrogate is not none, train encoder with surrogate
            if surrogate:
                predictions = cnn(spectra)
                output = surrogate(predictions)
                loss += nn.MSELoss()(output, spectra).item()
            else:
                params = params.to(device)

                # Train encoder with supervision or decoder
                if cnn.encoder:
                    output = cnn(spectra)
                    loss += nn.MSELoss()(output, params).item()
                    # loss += nn.CrossEntropyLoss()(output, params).item()
                else:
                    output = cnn(params)
                    loss += nn.MSELoss()(output, spectra).item()

    return loss / len(loader), spectra.cpu().numpy(), output.cpu().numpy()


def train(
        device: torch.device,
        loader: DataLoader,
        cnn: Network,
        surrogate: Network = None) -> float:
    """
    Trains the encoder or decoder using cross entropy or mean squared error

    Parameters
    ----------
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training
    surrogate : Network, defualt = None
        Surrogate network for encoder training

    Returns
    -------
    float
        Average loss value
    """
    epoch_loss = 0
    cnn.train()

    if surrogate:
        surrogate.train()

    for spectra, params, _ in loader:
        spectra = spectra.to(device)

        # If surrogate is not none, train encoder with surrogate
        if surrogate:
            target = spectra
            output = surrogate(cnn(spectra))
        else:
            params = params.to(device)

            # Train encoder with supervision or decoder
            if cnn.encoder:
                target = params
                output = cnn(spectra)
            else:
                target = spectra
                output = cnn(params)

        loss = nn.MSELoss()(output, target)

        # Optimise CNN
        cnn.optimizer.zero_grad()
        loss.backward()
        cnn.optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def train_val(
        num_epochs: int,
        losses: tuple[list, list],
        loaders: tuple[DataLoader, DataLoader],
        cnn: Network,
        device: torch.device,
        initial_epoch: int = 0,
        save_num: int = 0,
        states_dir: str = None,
        surrogate: Network = None) -> tuple[tuple[list, list], np.ndarray, np.ndarray]:
    """
    Trains & validates the network for each epoch

    Parameters
    ----------
    num_epochs : integer
        Number of epochs to train
    losses : tuple[list, list]
        Train and validation losses for each epoch
    loaders : tuple[DataLoader, DataLoader]
        Train and validation dataloaders
    cnn : Network
        CNN to use for training
    device : device
        Which device type PyTorch should use
    initial_epoch : integer, default = 0
        The epoch to start from
    save_num : integer, default = 0
        The file number to save the new state, if 0, nothing will be saved
    states_dir : string, default = None
        Path to the folder where the network state will be saved, not needed if save_num = 0
    surrogate : Network, default = None
        Surrogate network to use for training

    Returns
    -------
    tuple[tuple[list, list], ndarray, ndarray]
        Train & validation losses, spectra & reconstructions
    """

    # Train for each epoch
    for epoch in range(num_epochs - initial_epoch):
        t_initial = time()
        epoch += initial_epoch + 1

        # Train CNN
        losses[0].append(train(device, loaders[0], cnn, surrogate=surrogate))

        # Validate CNN
        losses[1].append(validate(device, loaders[1], cnn, surrogate=surrogate)[0])
        cnn.scheduler.step(losses[1][-1])

        # Save training progress
        if save_num:
            state = {
                'epoch': epoch,
                'train_loss': losses[0],
                'val_loss': losses[1],
                'indices': loaders[0].dataset.dataset.indices,
                'state_dict': cnn.state_dict(),
                'optimizer': cnn.optimizer.state_dict(),
                'scheduler': cnn.scheduler.state_dict(),
            }

            torch.save(state, f'{states_dir}{cnn.name}_{save_num}.pth')

        print(f'Epoch [{epoch}/{num_epochs}]\t'
              f'Training loss: {losses[0][-1]:.3e}\t'
              f'Validation loss: {losses[1][-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    # Final validation
    loss, spectra, outputs = validate(device, loaders[1], cnn, surrogate=surrogate)
    losses[1].append(loss)
    print(f'\nFinal validation loss: {losses[1][-1]:.3e}')

    return losses, spectra, outputs
