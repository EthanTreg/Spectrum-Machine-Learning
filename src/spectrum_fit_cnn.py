import os
from time import time
from multiprocessing import Process, Queue, Value

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from src.utils.networks import Network
from src.utils.data_utils import data_initialisation
from src.utils.network_utils import load_network
from src.utils.utils import PyXspecFitting, plot_loss, plot_reconstructions


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

    # for param in surrogate.parameters():
    #     param.requires_grad = False

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


def encoder_test(
        log_params: list[int],
        dirs: list[str],
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
    dirs : list[string]
        Directory of project & dataset files
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
    counter = Value('i', 0)
    os.chdir(dirs[1])
    queue = Queue()
    loader_output = next(enumerate(loader))[1]
    batch = loader_output[0].size(0)
    outputs = torch.empty((0, loader_output[1].size(1)))

    with torch.no_grad():
        for spectra, _, _ in loader:
            spectra = spectra.to(device)

            # Generate parameter predictions if encoder, else generate spectra and calculate loss
            output = cnn(spectra)
            output[:, log_params] = 10 ** output[:, log_params]
            outputs = torch.vstack((outputs, output))

    # Initialize processes for multiprocessing of encoder loss calculation
    processes = [Process(
        target=model.fit_loss,
        args=(
            len(loader.dataset),
            data[2],
            outputs[batch * i:batch * (i + 1)],
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

        # Validate CNN
        losses[1].append(validate(device, loaders[1], cnn, surrogate=surrogate)[0])
        cnn.scheduler.step(losses[1][-1])

        # Train CNN
        losses[0].append(train(device, loaders[0], cnn, surrogate=surrogate))

        # Save training progress
        if save_num:
            state = {
                'epoch': epoch,
                'state_dict': cnn.state_dict(),
                'optimizer': cnn.optimizer.state_dict(),
                'scheduler': cnn.scheduler.state_dict(),
                'train_loss': losses[0],
                'val_loss': losses[1],
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


def initialization(
        name: str,
        config_dir: str,
        spectra_path: str,
        params_path: str,
        states_dir: str,
        log_params: list,
        load_num: int = 0,
        learning_rate: float = 2e-4) -> tuple[
    int,
    tuple[list, list],
    tuple[DataLoader, DataLoader],
    Network,
    torch.device,
]:
    """
    Trains & validates network, used for progressive learning

    Parameters
    ----------
    name : string
        Name of the network
    config_dir : string
        Path to the network config directory
    spectra_path : string
        Path to the spectra data
    params_path : string
        Path to the parameters
    states_dir : string,
        Path to the folder where the network state will be saved
    log_params : list
        Indices of parameters in logarithmic space
    load_num : integer, default = 0
        The file number for the previous state, if 0, nothing will be loaded
    learning_rate : float, default = 2e-4
        Learning rate for the optimizer

    Returns
    -------
    tuple[int, tuple[list, list], tuple[DataLoader, DataLoader], Network]
        Initial epoch; train & validation losses; train & validation dataloaders; & network
    """
    # Constants
    initial_epoch = 0
    losses = ([], [])

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    # Initialize datasets
    loaders = data_initialisation(
        spectra_path,
        params_path,
        log_params,
        kwargs,
    )

    # Initialize network
    network = Network(
        loaders[0].dataset[0][0].size(0),
        loaders[0].dataset[0][1].size(0),
        learning_rate,
        name,
        config_dir,
    ).to(device)

    # Load states from previous training
    if load_num:
        initial_epoch, network, losses = load_network(
            load_num,
            states_dir,
            network,
        )

    return initial_epoch, losses, loaders, network, device


def plot_initialization(
    prefix: str,
    plots_dir: str,
    losses: tuple[list, list],
    spectra: np.ndarray,
    outputs: np.ndarray):
    """
    Initializes & plots reconstruction & loss plots

    Parameters
    ----------
    prefix : string
        Name prefix for plots
    plots_dir : string
        Directory to save plots
    losses : tuple[list, list]
        Training & validation losses
    spectra : ndarray
        Original spectra
    outputs : ndarray
        Reconstructions
    """
    # Initialize reconstructions plots
    _, axes = plt.subplots(4, 4, figsize=(24, 12), constrained_layout=True)
    axes = axes.flatten()

    # Plot reconstructions
    for i in range(axes.size):
        plot_reconstructions(spectra[i], outputs[i], axes[i])

    plt.savefig(f'{plots_dir}{prefix} Reconstructions.png')

    # Plot loss over epochs
    plot_loss(losses[0], losses[1])
    plt.savefig(f'{plots_dir}{prefix} Loss.png')


def main():
    """
    Main function for spectrum machine learning
    """
    # Variables
    e_load_num = 0
    e_save_num = 3
    d_load_num = 0
    d_save_num = 6
    num_epochs = 200
    learning_rate = 2e-4
    config_dir = '../network_configs/'
    synth_path = '../data/synth_spectra.npy'
    synth_params_path = '../data/synth_spectra_params.npy'
    spectra_path = '../data/preprocessed_spectra.npy'
    params_path = '../data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    # data_dir = '../../../Documents/Nicer_Data/ethan/'
    log_params = [0, 2, 3, 4]

    # Constants
    states_dir = '../model_states/'
    plots_dir = '../plots/'
    # root_dir = os.path.dirname(os.path.abspath(__file__))

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Initialize data & encoder
    e_initial_epoch, e_losses, e_loaders, encoder, _ = initialization(
        'Encoder V2',
        config_dir,
        spectra_path,
        params_path,
        # synth_path,
        # synth_params_path,
        states_dir,
        log_params,
        load_num=e_load_num,
        learning_rate=learning_rate,
    )

    # Initialize data & decoder
    d_initial_epoch, d_losses, d_loaders, decoder, device = initialization(
        'Decoder',
        config_dir,
        synth_path,
        synth_params_path,
        states_dir,
        log_params,
        load_num=d_load_num,
        learning_rate=learning_rate,
    )

    # Train decoder
    losses, spectra, outputs = train_val(
        num_epochs,
        d_losses,
        d_loaders,
        decoder,
        device,
        d_initial_epoch,
        d_save_num,
        states_dir,
    )

    plot_initialization(
        'Decoder',
        plots_dir,
        losses,
        spectra,
        outputs,
    )

    # Train encoder
    losses, spectra, outputs = train_val(
        num_epochs,
        e_losses,
        e_loaders,
        encoder,
        device,
        e_initial_epoch,
        e_save_num,
        states_dir,
        surrogate=decoder,
    )

    # losses, spectra, outputs = train_val(
    #     num_epochs,
    #     losses,
    #     e_loaders,
    #     encoder,
    #     device,
    #     num_epochs,
    #     0,
    #     surrogate=decoder,
    # )

    plot_initialization(
        'Encoder-Decoder',
        plots_dir,
        losses,
        spectra,
        outputs,
    )


if __name__ == '__main__':
    main()
