import os

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.utils.networks import Network
from src.utils.network_utils import load_network
from src.utils.data_utils import data_initialisation
from src.utils.train_utils import train_val, encoder_test, xspec_loss
from src.utils.utils import PyXspecFitting, plot_loss, plot_reconstructions


def plot_initialization(
    prefix: str,
    plots_dir: str,
    losses: tuple[list, list],
    spectra: np.ndarray,
    outputs: np.ndarray
):
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
    text_color = '#d9d9d9'
    matplotlib.rcParams.update({
        'text.color': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'axes.labelcolor': text_color,
        'axes.edgecolor': text_color,
        'axes.facecolor': (0, 0, 1, 0),
    })

    # Initialize reconstructions plots
    _, axes = plt.subplots(4, 4, figsize=(24, 12), sharex='col', gridspec_kw={'hspace': 0})
    axes = axes.flatten()

    # Plot reconstructions
    for i in range(axes.size):
        plot_reconstructions(spectra[i], outputs[i], axes[i])

    plt.figtext(0.5, 0.02, 'Energy (keV)', ha='center', va='center', fontsize=16)
    plt.figtext(
        0.02,
        0.5,
        'Scaled Log Counts',
        ha='center',
        va='center',
        rotation='vertical',
        fontsize=16,
    )

    legend = plt.figlegend(
        *axes[0].get_legend_handles_labels(),
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.95),
        fontsize=16,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    plt.savefig(f'{plots_dir}{prefix} Reconstructions.png', transparent=True)

    # Plot loss over epochs
    plot_loss(losses[0], losses[1])
    plt.savefig(f'{plots_dir}{prefix} Loss.png', transparent=True)


def initialization(
        name: str,
        config_dir: str,
        spectra_path: str,
        params_path: str,
        states_dir: str,
        log_params: list,
        load_num: int = 0,
        learning_rate: float = 2e-4,
        transform: list[list[np.ndarray]] = None,
) -> tuple[
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
    transform : list[list[ndarray]], default = None
        Min and max spectral range and mean & standard deviation of parameters

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
        transform=transform,
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


def main():
    """
    Main function for spectrum machine learning
    """
    # Variables
    e_load_num = 0
    e_save_num = 1
    d_load_num = 1
    d_save_num = 0
    num_epochs = 200
    learning_rate = 2e-4
    config_dir = '../network_configs/'
    synth_path = '../data/synth_spectra.npy'
    synth_params_path = '../data/synth_spectra_params.npy'
    spectra_path = '../data/preprocessed_spectra.npy'
    params_path = '../data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    data_dir = '../../../Documents/Nicer_Data/spectra/'
    log_params = [0, 2, 3, 4]
    fix_params = np.array([[4, 0], [5, 100]])

    # Constants
    states_dir = '../model_states/'
    plots_dir = '../plots/'
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize PyXspec model
    model = PyXspecFitting('tbabs(simplcutx(ezdiskbb))', [root_dir, data_dir], fix_params)

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Initialize data & decoder
    d_initial_epoch, d_losses, d_loaders, decoder, device = initialization(
        'Decoder V2',
        config_dir,
        synth_path,
        synth_params_path,
        states_dir,
        log_params,
        load_num=d_load_num,
        learning_rate=learning_rate,
    )

    # Initialize data & encoder
    e_initial_epoch, e_losses, e_loaders, encoder, _ = initialization(
        'Encoder V3',
        config_dir,
        spectra_path,
        params_path,
        # synth_path,
        # synth_params_path,
        states_dir,
        log_params,
        load_num=e_load_num,
        learning_rate=learning_rate,
        transform=d_loaders[0].dataset.dataset.transform
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

    plot_initialization('Decoder', plots_dir, losses, spectra, outputs)

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

    plot_initialization('Encoder-Decoder', plots_dir, losses, spectra, outputs)

    loss = encoder_test(
        log_params,
        device,
        e_loaders[1],
        encoder,
        model,
    )
    print(f'\nFinal PGStat Loss: {loss:.3e}')

    loss = xspec_loss(log_params, e_loaders[1], model)
    print(f'\nXspec Fitting Loss: {loss:.3e}')


if __name__ == '__main__':
    main()
