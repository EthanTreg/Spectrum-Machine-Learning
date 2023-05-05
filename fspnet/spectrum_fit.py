"""
Main spectral machine learning module
"""
import os
import pickle
import logging as log

import torch
import matplotlib
import numpy as np
from torch.utils.data import DataLoader

from fspnet.utils.data import data_initialisation
from fspnet.utils.utils import open_config, get_device
from fspnet.utils.training import training, pyxspec_test
from fspnet.utils.network import load_network, Network
from fspnet.utils.analysis import (
    autoencoder_saliency,
    decoder_saliency,
    linear_weights,
)
from fspnet.utils.plots import (
    plot_saliency,
    plot_param_pairs,
    plot_param_distribution,
    plot_param_comparison,
    plot_linear_weights,
    plot_training,
)


def predict_parameters(
        config: dict | str = '../config.yaml',
        encoder: Network = None) -> np.ndarray:
    """
    Predicts parameters using the encoder & saves the results to a file

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or file path to the configuration file
    encoder : Network, default = None
        Encoder network for generating parameters,
        otherwise will load network from the configuration file

    Returns
    -------
    ndarray
        Spectra names and parameter predictions
    """
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', config)

    returns = initialization(
        config['training']['encoder-name'],
        config,
    )[2:]
    _, loader = returns[0]

    if not encoder:
        encoder = returns[1]

    output_path = config['output']['parameter-predictions-path']
    names = []
    params = []
    log_params = loader.dataset.dataset.log_params
    param_transform = loader.dataset.dataset.transform[1]

    encoder.eval()

    # Initialize processes for multiprocessing of encoder loss calculation
    with torch.no_grad():
        for spectra, *_, names_batch in loader:
            names.extend(names_batch)
            param_batch = encoder(spectra.to(get_device()[1])).cpu()

            # Transform parameters
            param_batch = param_batch * param_transform[1] + param_transform[0]

            if log_params:
                param_batch[:, log_params] = 10 ** param_batch[:, log_params]

            params.append(param_batch)

    output = np.hstack((np.expand_dims(names, axis=1), torch.cat(params).numpy()))
    np.savetxt(output_path, output, delimiter=',', fmt='%s')

    return output


def initialization(
        name: str,
        config: dict | str = '../config.yaml',
        transform: tuple[tuple[float, float], tuple[np.ndarray, np.ndarray]] = None) -> tuple[
    int,
    tuple[list, list],
    tuple[DataLoader, DataLoader],
    Network,
]:
    """
    Trains & validates network, used for progressive learning

    Parameters
    ----------
    name : string
        Name of the network
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or file path to the configuration file
    transform : tuple[tuple[float, float], tuple[ndarray, ndarray]], default = None
        Min and max spectral range and mean & standard deviation of parameters

    Returns
    -------
    tuple[integer, tuple[list, list], tuple[DataLoader, DataLoader], Network]
        Initial epoch; train & validation losses; train & validation dataloaders; & network
    """
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', config)

    # Constants
    initial_epoch = 0
    losses = ([], [])
    device = get_device()[1]

    if 'encoder' in name.lower():
        network_type = 'encoder'
    elif 'decoder' in name.lower():
        network_type = 'decoder'
    else:
        raise NameError(f'Unknown network type: {name}\n'
                        f'Make sure encoder or decoder is included in the name')

    # Load config parameters
    load_num = config['training'][f'{network_type}-load']
    num_params = config['model']['parameters-number']
    learning_rate = config['training']['learning-rate']
    networks_dir = config['training']['network-configs-directory']
    spectra_path = config['data'][f'{network_type}-data-path']
    states_dir = config['output']['network-states-directory']
    log_params = config['model']['log-parameters']

    if load_num:
        try:
            state = torch.load(f'{states_dir}{name}_{load_num}.pth', map_location=device)
            indices = state['indices']
            transform = state['transform']
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
        log_params,
        transform=transform,
        indices=indices,
    )

    # Initialize network
    network = Network(
        loaders[0].dataset[0][0].size(0),
        num_params,
        learning_rate,
        name,
        networks_dir,
    ).to(device)

    # Load states from previous training
    if load_num:
        initial_epoch, network, losses = load_network(load_num, states_dir, network)

    return initial_epoch, losses, loaders, network


def main(config_path: str = '../config.yaml'):
    """
    Main function for spectrum machine learning

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        File path to the configuration file
    """
    _, config = open_config('spectrum-fit', config_path)

    # Training variables
    tests = config['training']['tests']
    num_epochs = config['training']['epochs']
    cpus = config['training']['cpus']

    # Data paths
    e_data_path = config['data']['encoder-data-path']
    d_data_path = config['data']['decoder-data-path']

    # Model parameters
    default_params = config['model']['default-parameters']
    param_names = config['model']['parameter-names']
    worker_data = {
        'optimize': False,
        'dirs': [
            os.path.dirname(os.path.abspath(__file__)),
            config['data']['spectra-directory'],
        ],
        'fix_params': config['model']['fixed-parameters'],
        'model': config['model']['model-name'],
        'custom_model': config['model']['custom-model-name'],
        'model_dir': config['model']['model-directory'],
    }

    # Output paths
    predictions_path = config['output']['parameter-predictions-path']
    states_dir = config['output']['network-states-directory']
    plots_dir = config['output']['plots-directory']
    worker_dir = config['output']['worker-directory']

    # Initialize Matplotlib display parameters
    matplotlib.use('qt5agg')
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

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

    # Save worker variables
    with open(f'{worker_dir}worker_data.pickle', 'wb') as file:
        pickle.dump(worker_data, file)

    # Initialize data & decoder
    d_initial_epoch, d_losses, d_loaders, decoder = initialization(
        config['training']['decoder-name'],
        config=config,
    )

    # Initialize data & encoder
    e_initial_epoch, e_losses, e_loaders, encoder = initialization(
        config['training']['encoder-name'],
        config=config,
        transform=d_loaders[0].dataset.dataset.transform,
    )

    # Train decoder
    decoder_return = training(
        (d_initial_epoch, num_epochs),
        d_loaders,
        decoder,
        save_num=config['training']['decoder-save'],
        states_dir=states_dir,
        losses=d_losses,
    )

    plot_training('Decoder', plots_dir, *decoder_return)

    # Train encoder
    encoder_return = training(
        (e_initial_epoch, num_epochs),
        e_loaders,
        encoder,
        save_num=config['training']['encoder-save'],
        states_dir=states_dir,
        losses=e_losses,
        surrogate=decoder,
    )

    plot_training('Autoencoder', plots_dir, *encoder_return)

    # Plot linear weight mappings
    plot_linear_weights(config, linear_weights(decoder))

    # Generate parameter predictions
    predict_parameters(config=config, encoder=encoder)
    plot_param_comparison(
        plots_dir,
        param_names,
        config,
    )
    plot_param_distribution(
        'Decoder_Param_Distribution',
        [d_data_path, predictions_path],
        ['Target', 'Prediction'],
        config,
    )
    plot_param_distribution(
        'Encoder_Param_Distribution',
        [e_data_path, predictions_path],
        ['Target', 'Prediction'],
        config,
    )
    plot_param_pairs([e_data_path, predictions_path], ['Targets', 'Predictions'], config)

    # Calculate saliencies
    decoder_saliency(d_loaders[1], decoder)
    saliency_output = autoencoder_saliency(e_loaders[1], encoder, decoder)
    plot_saliency(plots_dir, *saliency_output)

    if not tests:
        return

    # Encoder validation performance
    print('\nTesting Encoder...')
    pyxspec_test(
        worker_dir,
        e_loaders[1],
        cpus=cpus,
        job_name='Encoder_output',
        encoder=encoder,
    )

    # Xspec performance
    print('\nTesting Xspec...')
    pyxspec_test(worker_dir, e_loaders[1], cpus=cpus, job_name='Xspec_output')

    # Default performance
    print('\nTesting Defaults...')
    pyxspec_test(
        worker_dir,
        e_loaders[1],
        cpus=cpus,
        defaults=torch.tensor(default_params)
    )

    # Allow Xspec optimization
    worker_data['optimize'] = True
    with open(f'{worker_dir}worker_data.pickle', 'wb') as file:
        pickle.dump(worker_data, file)

    # Encoder + Xspec performance
    print('\nTesting Encoder + Fitting...')
    pyxspec_test(
        worker_dir,
        e_loaders[1],
        cpus=cpus,
        job_name='Encoder_Xspec_output',
        encoder=encoder,
    )

    # Default + Xspec performance
    print('\nTesting Defaults + Fitting...')
    pyxspec_test(
        worker_dir,
        e_loaders[1],
        cpus=cpus,
        defaults=torch.tensor(default_params)
    )


if __name__ == '__main__':
    main()
