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
from src.utils.utils import PyXspecFitting, network_initialisation


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

    if x_data.size % 2 != 0:
        x_data = np.append(x_data[:-2], np.mean(x_data[ -2:]))

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

            # Generate predictions and loss if network is a decoder
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


def main():
    """
    Main function for spectrum machine learning
    """
    # Variables
    save = True
    load = True
    num_epochs = 200
    val_frac = 0.1
    learning_rate = 1e-5
    config_path = '../decoder.json'
    synth_path = '../data/synth_spectra.npy'
    # spectra_path = '../data/preprocessed_spectra.npy'
    synth_params_path = '../data/synth_spectra_params.npy'
    # params_path = '../data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    # data_dir = '../../../Documents/Nicer_Data/ethan/'
    fix_params = np.array([[4, 0], [5, 100]])
    log_params = [0, 2, 3, 4]

    # Constants
    initial_epoch = 0
    states_dir = '../model_states/'
    plots_dir = '../plots/'
    train_loss = []
    val_loss = []
    # root_dir = os.path.dirname(os.path.abspath(__file__))

    # Xspec initialization
    model = PyXspecFitting('tbabs(simplcutx(ezdiskbb))', fix_params)

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    # Initialize decoder
    decoder, d_optimizer, d_scheduler, d_train_loader, d_val_loader = network_initialisation(
        learning_rate,
        val_frac,
        synth_path,
        synth_params_path,
        log_params,
        (model.param_limits.shape[0], config_path),
        kwargs,
        Decoder,
        device
    )

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Load states from previous training
    if load:
        d_state = torch.load(f'{states_dir}{type(decoder).__name__}.pth')
        initial_epoch = d_state['epoch']
        decoder.load_state_dict(d_state['state_dict'])
        d_optimizer.load_state_dict(d_state['optimizer'])
        d_scheduler.load_state_dict(d_state['scheduler'])
        train_loss = d_state['train_loss']
        val_loss = d_state['val_loss']

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
        if save:
            d_state = {
                'epoch': epoch,
                'state_dict': decoder.state_dict(),
                'optimizer': d_optimizer.state_dict(),
                'scheduler': d_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }

            torch.save(d_state, f'{states_dir}{type(decoder).__name__}.pth')

        print(f'Epoch [{epoch + 1}/{num_epochs}]\t'
              f'Training loss: {train_loss[-1]:.3e}\t'
              f'Validation loss: {val_loss[-1]:.3e}\t'
              f'Time: {time() - t_initial:.1f}')

    # Final validation
    loss, spectra, outputs = validate(device, d_val_loader, decoder)
    val_loss.append(loss)
    print(f'\nFinal validation loss: {val_loss[-1]:.3e}')

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Initialize reconstructions plots
    _, axes = plt.subplots(4, 3, figsize=(24, 12), constrained_layout=True)
    axes = axes.flatten()

    # Plot reconstructions
    for i in range(axes.size):
        plot_reconstructions(num_epochs, spectra[i], outputs[i], axes[i])

    plt.savefig(plots_dir + 'Reconstructions.png')

    # Plot loss over epochs
    plot_loss(plots_dir, train_loss, val_loss)


if __name__ == '__main__':
    main()
