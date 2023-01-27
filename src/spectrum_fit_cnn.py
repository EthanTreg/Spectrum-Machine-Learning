import os

import torch
import numpy as np
import matplotlib
from torch.utils.data import DataLoader

from src.utils.networks import Network
from src.utils.network_utils import load_network
from src.utils.data_utils import data_initialisation
from src.utils.plot_utils import plot_initialization, plot_saliency
from src.utils.train_utils import training, encoder_test


def calculate_saliency(
        plots_dir: str,
        loaders: tuple[DataLoader, DataLoader],
        device: torch.device,
        encoder: Network,
        decoder: Network):
    """
    Generates saliency values for autoencoder & decoder
    Prints stats on decoder parameter significance

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    loaders : tuple[DataLoader, DataLoader]
        Autoencoder & decoder validation dataloaders
    device : device
        Which device type PyTorch should use
    encoder : Network
        Encoder half of the network
    decoder : Network
        Decoder half of the network
    """
    # Constants
    d_spectra, d_parameters, *_ = next(iter(loaders[0]))
    e_spectra = next(iter(loaders[1]))[0][:8].to(device)

    # Network initialization
    decoder.train()
    encoder.train()
    d_spectra = d_spectra.to(device)
    d_parameters = d_parameters.to(device)
    d_parameters.requires_grad_()
    e_spectra.requires_grad_()

    # Generate predictions
    d_output = decoder(d_parameters)
    e_output = decoder(encoder(e_spectra))

    # Perform backpropagation
    d_loss = torch.nn.MSELoss()(d_output, d_spectra)
    e_loss = torch.nn.MSELoss()(e_output, e_spectra)
    d_loss.backward()
    e_loss.backward()

    # Calculate saliency
    d_saliency = d_parameters.grad.data.abs().cpu()
    e_saliency = e_spectra.grad.data.abs().cpu()

    # Measure impact of input parameters on decoder output
    parameter_saliency = torch.mean(d_saliency, dim=0)
    parameter_impact = parameter_saliency / torch.min(parameter_saliency)
    parameter_std = torch.std(d_saliency, dim=0) / torch.min(parameter_saliency)

    print(
        f'\nParameter impact on decoder:\n{parameter_impact.tolist()}'
        f'\nParameter spread:\n{parameter_std.tolist()}\n'
    )

    e_spectra = e_spectra.cpu().detach()
    e_output = e_output.cpu().detach()
    e_saliency = e_saliency.cpu()

    plot_saliency(plots_dir, e_spectra, e_output, e_saliency)


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
    tuple[integer, tuple[list, list], tuple[DataLoader, DataLoader], Network, device]
        Initial epoch; train & validation losses; train & validation dataloaders; network; & device
    """
    # Constants
    initial_epoch = 0
    losses = ([], [])

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    if load_num:
        try:
            indices = torch.load(f'{states_dir}{name}_{load_num}.pth')['indices']
        except FileNotFoundError:
            print(f'ERROR: {states_dir}{name}_{load_num}.pth does not exist')
            load_num = 0
            indices = None
    else:
        indices = None

    # Initialize datasets
    loaders = data_initialisation(
        spectra_path,
        params_path,
        log_params,
        kwargs,
        transform=transform,
        indices=indices,
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
    e_load_num = 8
    d_load_num = 11
    e_save_num = 8
    d_save_num = 11
    num_epochs = 100
    learning_rate = 2e-4
    config_dir = '../network_configs/'
    synth_path = '../data/synth_spectra.npy'
    synth_params_path = '../data/synth_spectra_params.npy'
    spectra_path = '../data/preprocessed_spectra.npy'
    params_path = '../data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    log_params = [0, 2, 3, 4]
    worker_data = {
        'optimize': False,
        'dirs': [
            os.path.dirname(os.path.abspath(__file__)),
            '../../../Documents/Nicer_Data/spectra/',
        ],
        'fix_params': np.array([[4, 0], [5, 100]]),
        'model': 'tbabs(simplcutx(ezdiskbb))',
    }

    # Constants
    states_dir = '../model_states/'
    plots_dir = '../plots/'
    worker_dir = '../data/worker/'

    # Initialize Matplotlib display parameters
    # text_color = '#d9d9d9'
    text_color = '#222222'
    matplotlib.rcParams.update({
        'text.color': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'axes.labelcolor': text_color,
        'axes.edgecolor': text_color,
        'axes.facecolor': (0, 0, 1, 0),
    })

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create worker directory
    if not os.path.exists(worker_dir):
        os.makedirs(worker_dir)

    # Save worker variables
    np.save(f'{worker_dir}worker_data.npy', worker_data)

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
    losses, spectra, outputs = training(
        (d_initial_epoch, num_epochs),
        d_losses,
        d_loaders,
        decoder,
        device,
        save_num=d_save_num,
        states_dir=states_dir,
    )

    plot_initialization('Decoder', plots_dir, losses, spectra, outputs)

    # Train encoder
    losses, spectra, outputs = training(
        (e_initial_epoch, num_epochs),
        e_losses,
        e_loaders,
        encoder,
        device,
        save_num=e_save_num,
        states_dir=states_dir,
        surrogate=decoder,
    )

    plot_initialization('Encoder-Decoder', plots_dir, losses, spectra, outputs)

    calculate_saliency(plots_dir, (d_loaders[1], e_loaders[1]), device, encoder, decoder)

    # Encoder validation performance
    print('\nTesting Encoder...')
    loss = encoder_test(
        worker_dir,
        log_params,
        e_loaders[1],
        job_name='Encoder_output',
        device=device,
        encoder=encoder,
    )
    print(f'PGStat Loss: {loss:.3e}')

    # Xspec performance
    print('\nTesting Xspec...')
    loss = encoder_test(worker_dir, log_params, e_loaders[1], job_name='Xspec_output')
    print(f'PGStat Loss: {loss:.3e}')

    # Default performance
    print('\nTesting Defaults...')
    loss = encoder_test(
        worker_dir,
        log_params,
        e_loaders[1],
        defaults=torch.tensor([1, 2.5, 0.02, 1, 1])
    )
    print(f'PGStat Loss: {loss:.3e}')

    # Allow Xspec optimization
    worker_data['optimize'] = True
    np.save(f'{worker_dir}worker_data.npy', worker_data)

    # Encoder + Xspec performance
    print('\nTesting Encoder + Fitting...')
    loss = encoder_test(
        worker_dir,
        log_params,
        e_loaders[1],
        job_name='Encoder_Xspec_output',
        device=device,
        encoder=encoder,
    )
    print(f'PGStat Loss: {loss:.3e}')

    print('\nTesting Defaults + Fitting...')
    loss = encoder_test(
        worker_dir,
        log_params,
        e_loaders[1],
        defaults=torch.tensor([1, 2.5, 0.02, 1, 1])
    )
    print(f'PGStat Loss: {loss:.3e}')


if __name__ == '__main__':
    main()
