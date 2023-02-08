"""
Main spectral machine learning module
"""
import os
import argparse
import logging as log

import yaml
import torch
import numpy as np
import matplotlib
from torch.utils.data import DataLoader

from fspnet.utils.data_utils import data_initialisation
from fspnet.utils.train_utils import training, pyxspec_test
from fspnet.utils.network_utils import load_network, Network
from fspnet.utils.saliency_utils import autoencoder_saliency, decoder_saliency
from fspnet.utils.plot_utils import plot_initialization, plot_saliency, plot_param_distribution


def initialization(
        name: str,
        config_path: str = '../config.yaml',
        transform: list[list[np.ndarray]] = None) -> tuple[
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
    config_path : string, default = '../config.yaml'
        File path to the configuration file
    transform : list[list[ndarray]], default = None
        Min and max spectral range and mean & standard deviation of parameters

    Returns
    -------
    tuple[integer, tuple[list, list], tuple[DataLoader, DataLoader], Network, device]
        Initial epoch; train & validation losses; train & validation dataloaders; network; & device
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = list(yaml.safe_load_all(file))[0]

    # Constants
    initial_epoch = 0
    losses = ([], [])

    if 'encoder' in name.lower():
        network_type = 'encoder'
    elif 'decoder' in name.lower():
        network_type = 'decoder'
    else:
        raise NameError(f'Unknown network type: {name}\n'
                        f'Make sure encoder or decoder is included in the name')

    # Load config parameters
    load_num = config['training'][f'{network_type}-load']
    learning_rate = config['training']['learning-rate']
    networks_dir = config['training']['network-configs-directory']
    spectra_path = config['data'][f'{network_type}-data-path']
    params_path = config['data'][f'{network_type}-parameters-path']
    states_dir = config['output']['network-states-directory']
    log_params = config['model']['log-parameters']

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Set device to GPU if available
    device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    if load_num:
        try:
            indices = torch.load(f'{states_dir}{name}_{load_num}.pth')['indices']
        except FileNotFoundError:
            log.warning(f'{states_dir}{name}_{load_num}.pth does not exist\n'
                  f'No state will be loaded')
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
        networks_dir,
    ).to(device)

    # Load states from previous training
    if load_num:
        initial_epoch, network, losses = load_network(
            load_num,
            states_dir,
            network,
        )

    return initial_epoch, losses, loaders, network, device


def predict_parameters(config_path: str = '../config.yaml'):
    """
    Predicts parameters using the encoder & saves the results to a file

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        File path to the configuration file
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = list(yaml.safe_load_all(file))[0]

    (_, loader), encoder, device = initialization(
        config['training']['encoder-name'],
        config_path,
    )[2:]

    output_path = config['output']['parameter-predictions-path']
    names = []
    params = []
    log_params = loader.dataset.dataset.log_params
    param_transform = loader.dataset.dataset.transform[1]

    encoder.eval()

    # Initialize processes for multiprocessing of encoder loss calculation
    with torch.no_grad():
        for data in loader:
            names.extend(data[-1])
            param_batch = encoder(data[0].to(device)).cpu()

            # Transform parameters
            param_batch = param_batch * param_transform[1] + param_transform[0]
            param_batch[:, log_params] = 10 ** param_batch[:, log_params]

            params.append(param_batch)

    output = np.hstack((np.expand_dims(names, axis=1), torch.cat(params).numpy()))
    np.savetxt(output_path, output, delimiter=',', fmt='%s')


def main(config_path: str = '../config.yaml'):
    """
    Main function for spectrum machine learning

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        File path to the configuration file
    """
    # If run by command line, optional argument can be used
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default=config_path)
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, 'r', encoding='utf-8') as file:
        config = list(yaml.safe_load_all(file))[0]

    # Training variables
    num_epochs = config['training']['epochs']

    # Model parameters
    default_params = config['model']['default-parameters']
    worker_data = {
        'optimize': False,
        'dirs': [
            os.path.dirname(os.path.abspath(__file__)),
            config['data']['spectra-directory'],
        ],
        'fix_params': np.array(config['model']['fixed-parameters']),
        'model': config['model']['model-name'],
        'custom_model': config['model']['custom-model-name'],
        'model_dir': config['model']['model-directory'],
    }

    # Output directories
    states_dir = config['output']['network-states-directory']
    plots_dir = config['output']['plots-directory']
    worker_dir = config['output']['worker-directory']

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

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create worker directory
    if not os.path.exists(worker_dir):
        os.makedirs(worker_dir)

    # Save worker variables
    np.save(f'{worker_dir}worker_data.npy', worker_data)

    # Initialize data & decoder
    d_initial_epoch, d_losses, d_loaders, decoder, _ = initialization(
        config['training']['decoder-name'],
        config_path,
    )

    # Initialize data & encoder
    e_initial_epoch, e_losses, e_loaders, encoder, device = initialization(
        config['training']['encoder-name'],
        config_path,
        transform=d_loaders[0].dataset.dataset.transform,
    )

    # Train decoder
    decoder_return = training(
        (d_initial_epoch, num_epochs),
        d_losses,
        d_loaders,
        decoder,
        device,
        save_num=config['training']['decoder-save'],
        states_dir=states_dir,
    )

    plot_initialization('Decoder', plots_dir, *decoder_return)

    # Train encoder
    encoder_return = training(
        (e_initial_epoch, num_epochs),
        e_losses,
        e_loaders,
        encoder,
        device,
        save_num=config['training']['encoder-save'],
        states_dir=states_dir,
        surrogate=decoder,
    )

    plot_initialization('Encoder-Decoder', plots_dir, *encoder_return)

    # Generate parameter predictions
    predict_parameters(config_path=config_path)

    # Calculate saliencies
    decoder_saliency(d_loaders[1], device, decoder)
    saliency_output = autoencoder_saliency(e_loaders[1], device, encoder, decoder)
    plot_saliency(plots_dir, *saliency_output)

    # Encoder validation performance
    print('\nTesting Encoder...')
    loss, params = pyxspec_test(
        worker_dir,
        e_loaders[1],
        job_name='Encoder_output',
        device=device,
        encoder=encoder,
    )
    print(f'PGStat Loss: {loss:.3e}')

    plot_param_distribution(plots_dir, config['model']['parameter-names'], params, e_loaders[1])

    # Xspec performance
    print('\nTesting Xspec...')
    loss = pyxspec_test(worker_dir, e_loaders[1], job_name='Xspec_output')[0]
    print(f'PGStat Loss: {loss:.3e}')

    # Default performance
    print('\nTesting Defaults...')
    loss = pyxspec_test(
        worker_dir,
        e_loaders[1],
        defaults=torch.tensor(default_params)
    )[0]
    print(f'PGStat Loss: {loss:.3e}')

    # Allow Xspec optimization
    worker_data['optimize'] = True
    np.save(f'{worker_dir}worker_data.npy', worker_data)

    # Encoder + Xspec performance
    print('\nTesting Encoder + Fitting...')
    loss = pyxspec_test(
        worker_dir,
        e_loaders[1],
        job_name='Encoder_Xspec_output',
        device=device,
        encoder=encoder,
    )[0]
    print(f'PGStat Loss: {loss:.3e}')

    # Default + Xspec performance
    print('\nTesting Defaults + Fitting...')
    loss = pyxspec_test(
        worker_dir,
        e_loaders[1],
        defaults=torch.tensor(default_params)
    )[0]
    print(f'PGStat Loss: {loss:.3e}')


if __name__ == '__main__':
    main()
