import os
from time import time
from multiprocessing import Process, Queue, Value

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import nn
from torch.utils.data import DataLoader

from src.utils.networks import Encoder, Decoder
from src.utils.utils import PyXspecFitting, data_initialisation, \
    network_initialisation, load_network


def plot_reconstructions(epoch: int, y_data: np.ndarray, y_recon: np.ndarray, axes: Axes):
    """
    Plots reconstructions for a given epoch

    Parameters
    ----------
    epoch : integer
        Epoch number
    y_data : ndarray
        Spectrum
    y_recon : ndarray
        Reconstructed Spectrum
    axes : Axes
        Plot axes
    """
    x_data = np.load('../data/spectra_x_axis.npy')

    # Make sure x data size is even
    if x_data.size % 2 != 0:
        x_data = np.append(x_data[:-2], np.mean(x_data[-2:]))

    # Make sure x data size is of the same size as y data
    if x_data.size != y_data.size and x_data.size % y_data.size == 0:
        x_data = x_data.reshape(int(x_data.size / y_data.size), - 1)
        x_data = np.mean(x_data, axis=0)

    axes.set_title(f'Epoch: {epoch}', fontsize=16)
    axes.scatter(x_data, y_data, label='Spectrum')
    axes.scatter(x_data, y_recon, label='Reconstruction')
    axes.set_xlabel('Energy (keV)', fontsize=12)
    axes.set_ylabel(r'$log_{10}$ Counts ($s^{-1}$ $detector^{-1}$ $keV^{-1}$)', fontsize=12)
    axes.legend(fontsize=16)


def plot_loss(plots_dir: str, train_loss: list, val_loss: list):
    """
    Plots training and validation loss as a function of epochs

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    train_loss : list
        Training losses
    val_loss : list
        Validation losses
    """
    plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.yscale('log')
    plt.text(
        0.8, 0.75,
        f'Final loss: {val_loss[-1]:.3e}',
        fontsize=16,
        transform=plt.gca().transAxes
    )
    plt.legend(fontsize=20)
    plt.savefig(plots_dir + 'Epoch_Loss.png')


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


def learning(
        load: bool,
        save: bool,
        load_num: int,
        save_num: int,
        num_epochs: int,
        learning_rate: float,
        config_path: str) -> tuple[list, list, np.ndarray, np.ndarray]:
    """
    Trains & validates network, used for progressive learning

    Parameters
    ----------
    load : boolean
        If a previous state should be loaded
    save : boolean
        If new states should be saved
    load_num : integer
        The file number for the previous state
    save_num : integer
        The file number to save the new state
    num_epochs : integer
        Number of epochs to train
    learning_rate : float
        Learning rate for the optimizer
    config_path : string
        Path to the network config file

    Returns
    -------
    tuple[list, list, ndarray, ndarray]
        Train losses, validation losses, spectra & spectra predictions
    """
    # Variables
    val_frac = 0.1
    synth_path = '../data/synth_spectra.npy'
    synth_params_path = '../data/synth_spectra_params.npy'
    log_params = [0, 2, 3, 4]

    # Constants
    initial_epoch = 0
    states_dir = '../model_states/'
    train_loss = []
    val_loss = []
    fix_params = np.array([[4, 0], [5, 100]])
    phases = np.clip(np.linspace(0, 2, num_epochs), a_min=0, a_max=1)

    # Xspec initialization
    model = PyXspecFitting('tbabs(simplcutx(ezdiskbb))', fix_params)

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    # Initialize datasets
    d_train_loader, d_val_loader = data_initialisation(
        val_frac,
        synth_path,
        synth_params_path,
        log_params,
        kwargs
    )

    # Initialize decoder
    decoder, d_optimizer, d_scheduler = network_initialisation(
        d_train_loader.dataset[0][0].size(0),
        learning_rate,
        (model.param_limits.shape[0], config_path),
        Decoder,
        device
    )

    print(f'{decoder.__class__.__name__}:\n'
          f'Training data size: {len(d_train_loader.dataset)}\t'
          f'Validation data size: {len(d_val_loader.dataset)}\n')

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Load states from previous training
    if load:
        initial_epoch, decoder, d_optimizer, d_scheduler, train_loss, val_loss = load_network(
            load_num,
            states_dir,
            decoder,
            d_optimizer,
            d_scheduler
        )

    # Train for each epoch
    for epoch in range(num_epochs - initial_epoch):
        t_initial = time()
        epoch += initial_epoch
        decoder.phase = phases[epoch]

        # Validate CNN
        val_loss.append(validate(device, d_val_loader, decoder)[0])
        d_scheduler.step(val_loss[-1])

        # Train CNN
        train_loss.append(train(device, d_train_loader, decoder, d_optimizer))

        # Save training progress
        if save:
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
    load = False
    save = True
    config_path = '../decoder.json'
    load_num = 6
    save_num = 6
    num_epochs = 200
    learning_rate = 1e-5
    # spectra_path = '../data/preprocessed_spectra.npy'
    # params_path = '../data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    # data_dir = '../../../Documents/Nicer_Data/ethan/'

    # Constants
    plots_dir = '../plots/'
    # root_dir = os.path.dirname(os.path.abspath(__file__))

    train_loss, val_loss, spectra, outputs = learning(
        load,
        save,
        load_num,
        save_num,
        num_epochs,
        learning_rate,
        config_path,
    )

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Initialize reconstructions plots
    _, axes = plt.subplots(4, 3, figsize=(24, 12), constrained_layout=True)
    axes = axes.flatten()

    # Plot reconstructions
    for i in range(axes.size):
        plot_reconstructions(epochs[-1], spectra[i], outputs[i], axes[i])

    plt.savefig(plots_dir + 'Reconstructions.png')

    # Plot loss over epochs
    plot_loss(plots_dir, train_loss, val_loss)


if __name__ == '__main__':
    main()
