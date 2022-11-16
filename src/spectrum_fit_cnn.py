import os
from time import time
from multiprocessing import Process, Queue, Value

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from src.utils.networks import Encoder, Decoder
from src.utils.data_utils import data_initialisation
from src.utils.network_utils import network_initialisation, load_network
from src.utils.utils import PyXspecFitting, plot_loss, plot_reconstructions


def train(
        device: torch.device,
        loader: DataLoader,
        cnn: Encoder | Decoder,
        optimizer: torch.optim.Optimizer) -> float:
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
    optimizer : optimizer
        Optimisation method to use for training

    Returns
    -------
    float
        Average loss value
    """
    epoch_loss = 0
    cnn.train()

    for spectra, params, _ in loader:
        spectra = spectra.to(device)
        params = params.to(device)

        # Generate predictions and loss
        if isinstance(cnn, Encoder):
            output = cnn(spectra)
            loss = nn.CrossEntropyLoss()(output, params)
        else:
            output = cnn(params)
            loss = nn.MSELoss()(output, spectra)
            # loss = weighted_mse(spectra, output)

        # Optimise CNN
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate(
        device: torch.device,
        loader: DataLoader,
        cnn: Encoder | Decoder) -> tuple[float, np.ndarray, np.ndarray]:
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

    Returns
    -------
    tuple[float, ndarray, ndarray]
        Average loss value, spectra & reconstructions
    """
    loss = 0
    cnn.eval()

    with torch.no_grad():
        for spectra, params, _ in loader:
            spectra = spectra.to(device)
            params = params.to(device)

            # Generate predictions and loss
            if isinstance(cnn, Encoder):
                output = cnn(spectra)
                loss += nn.CrossEntropyLoss()(output, params).item()
            else:
                output = cnn(params)
                loss += nn.MSELoss()(output, spectra).item()
                # loss += weighted_mse(spectra, output).item()

    return loss / len(loader), spectra.cpu().numpy(), output.cpu().numpy()


def test(
        log_params: list[int],
        dirs: list[str],
        device: torch.device,
        loader: DataLoader,
        cnn: Decoder,
        model: PyXspecFitting) -> float:
    """
    Tests the encoder or decoder using PyXspec or mean squared error

    Parameters
    ----------
    log_params : list[integer]
        Indices of parameters in logarithmic space
    dirs : list[string]
        Directory of project & dataset files
    device : device
        Which device type PyTorch should use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training
    model : PyXspecFitting
        PyXspec model for fit evaluation

    Returns
    -------
    float
        Average loss value
    """
    loss = 0
    cnn.eval()

    # Initialize multiprocessing variables
    if isinstance(cnn, Encoder):
        counter = Value('i', 0)
        os.chdir(dirs[1])
        queue = Queue()
        loader_output = next(enumerate(loader))[1]
        batch = loader_output[0].size(0)
        outputs = torch.empty((0, loader_output[1].size(1)))

    with torch.no_grad():
        for spectra, params, _ in loader:
            spectra = spectra.to(device)

            # Generate parameter predictions if encoder, else generate spectra and calculate loss
            if isinstance(cnn, Encoder):
                output = cnn(spectra)
                output[:, log_params] = 10 ** output[:, log_params]
                outputs = torch.vstack((outputs, output))
            else:
                params = params.to(device)
                output = cnn(params)
                loss += nn.MSELoss()(output, spectra).item()

    # Initialize processes for multiprocessing of encoder loss calculation
    if isinstance(cnn, Encoder):
        processes = [Process(
            target=model.fit_loss,
            args=(
                len(loader.dataset),
                data[2],
                outputs[batch * i:batch * (i + 1)],
                cnn.model,
                counter,
                queue
            )
        ) for i, data in enumerate(loader)]

        # Start multiprocessing
        for process in processes:
            process.start()

        # End multiprocessing
        for process in processes:
            process.join()

        # Collect results
        losses = [queue.get() for _ in processes]

        os.chdir(dirs[0])

        return sum(losses) / len(losses)

    return loss / len(loader)


def initialization(
        num_epochs: int,
        learning_rate: float,
        config_path: str,
        load_num: int = 0,
        save_num: int = 0) -> tuple[list, list, np.ndarray, np.ndarray]:
    """
    Trains & validates network, used for progressive learning

    Parameters
    ----------
    num_epochs : integer
        Number of epochs to train
    learning_rate : float
        Learning rate for the optimizer
    config_path : string
        Path to the network config file
    load_num : integer, default = 0
        The file number for the previous state, if 0, nothing will be loaded
    save_num : integer, default = 0
        The file number to save the new state, if 0, nothing will be saved

    Returns
    -------
    tuple[list, list, ndarray, ndarray]
        Train losses, validation losses, spectra & spectra predictions
    """
    # Variables
    synth_path = '../data/synth_spectra.npy'
    synth_params_path = '../data/synth_spectra_params.npy'
    log_params = [0, 2, 3, 4]

    # Constants
    initial_epoch = 0
    train_loss = []
    val_loss = []
    states_dir = '../model_states/'

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    # Initialize datasets
    d_train_loader, d_val_loader = data_initialisation(
        synth_path,
        synth_params_path,
        log_params,
        kwargs,
    )

    # Initialize decoder
    decoder, d_optimizer, d_scheduler = network_initialisation(
        d_train_loader.dataset[0][0].size(0),
        learning_rate,
        (d_train_loader.dataset[0][1].size(0), config_path),
        Decoder,
    )
    decoder = decoder.to(device)

    # Load states from previous training
    if load_num:
        initial_epoch, decoder, d_optimizer, d_scheduler, train_loss, val_loss = load_network(
            load_num,
            states_dir,
            decoder,
            d_optimizer,
            d_scheduler
        )

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Train for each epoch
    for epoch in range(num_epochs - initial_epoch):
        t_initial = time()
        epoch += initial_epoch

        # Validate CNN
        val_loss.append(validate(device, d_val_loader, decoder)[0])
        d_scheduler.step(val_loss[-1])

        # Train CNN
        train_loss.append(train(device, d_train_loader, decoder, d_optimizer))

        # Save training progress
        if save_num:
            d_state = {
                'epoch': epoch,
                'state_dict': decoder.state_dict(),
                'optimizer': d_optimizer.state_dict(),
                'scheduler': d_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }

            torch.save(d_state, f'{states_dir}{type(decoder).__name__}_{save_num}.pth')

        print(f'Epoch [{epoch + 1}/{num_epochs}]\t'
              f'Training loss: {train_loss[-1]:.3e}\t'
              f'Validation loss: {val_loss[-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    # Final validation
    loss, spectra, outputs = validate(device, d_val_loader, decoder)
    val_loss.append(loss)
    print(f'\nFinal validation loss: {val_loss[-1]:.3e}')

    return train_loss, val_loss, spectra, outputs


def main():
    """
    Main function for spectrum machine learning
    """
    # Variables
    load_num = 0
    save_num = 1
    num_epochs = 200
    learning_rate = 2e-4
    config_path = '../network_configs/decoder.json'
    # spectra_path = '../data/preprocessed_spectra.npy'
    # params_path = '../data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    # data_dir = '../../../Documents/Nicer_Data/ethan/'

    # Constants
    plots_dir = '../plots/'
    # root_dir = os.path.dirname(os.path.abspath(__file__))

    train_loss, val_loss, spectra, outputs = initialization(
        num_epochs,
        learning_rate,
        config_path,
        load_num,
        save_num,
    )

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Initialize reconstructions plots
    _, axes = plt.subplots(4, 4, figsize=(24, 12), constrained_layout=True)
    axes = axes.flatten()

    # Plot reconstructions
    for i in range(axes.size):
        plot_reconstructions(num_epochs, spectra[i], outputs[i], axes[i])

    plt.savefig(plots_dir + 'Reconstructions.png')

    # Plot loss over epochs
    plot_loss(plots_dir, train_loss, val_loss)


if __name__ == '__main__':
    main()
