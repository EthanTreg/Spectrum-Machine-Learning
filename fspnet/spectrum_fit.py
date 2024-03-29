"""
Main spectral machine learning module
"""
import os
import pickle
import logging as log
from time import time

import torch
import numpy as np
import matplotlib as mpl
from torch.utils.data import DataLoader

from fspnet.utils import plots
from fspnet.utils.data import data_initialisation
from fspnet.utils.utils import open_config, get_device
from fspnet.utils.network import load_network, Network
from fspnet.utils.training import training, pyxspec_test
from fspnet.utils.analysis import autoencoder_saliency, decoder_saliency


def pyxspec_tests(config: dict, dataset: torch.utils.data.Dataset):
    """
    Tests the PGStats of the different fitting methods using PyXspec

    Parameters
    ----------
    config : dictionary
        Configuration dictionary
    dataset : Dataset
        Validation dataset to test the PGStats for
    """
    # Initialize variables
    cpus = config['training']['cpus']
    default_params = config['model']['default-parameters']
    log_params = config['model']['log-parameters']
    predictions_path = config['output']['parameter-predictions-path']
    worker_dir = config['output']['worker-directory']
    worker_data = {
        'optimize': False,
        'dirs': [
            os.path.dirname(os.path.abspath(__file__)),
            config['data']['spectra-directory'],
        ],
        'iterations': config['model']['iterations'],
        'step': config['model']['step'],
        'fix_params': config['model']['fixed-parameters'],
        'model': config['model']['model-name'],
        'custom_model': config['model']['custom-model-name'],
        'model_dir': config['model']['model-directory'],
    }
    indices = dataset.indices
    names = dataset.dataset.names[indices]
    xspec_params = dataset.dataset.params[indices]
    default_params = np.repeat([default_params], names.size, axis=0)

    # Untransform Xspec parameters
    xspec_params = xspec_params * dataset.dataset.transform[1][1] + dataset.dataset.transform[1][0]

    if log_params:
        xspec_params[:, log_params] = 10 ** xspec_params[:, log_params]

    # Save worker variables
    with open(f'{worker_dir}worker_data.pickle', 'wb') as file:
        pickle.dump(worker_data, file)

    # Encoder validation performance
    print('\nTesting Encoder...')
    pyxspec_test(worker_dir, predictions_path, cpus=cpus, job_name='Encoder_output')

    # Xspec performance
    print('\nTesting Xspec...')
    pyxspec_test(worker_dir, (names, xspec_params), cpus=cpus, job_name='Xspec_output')

    # Default performance
    print('\nTesting Defaults...')
    pyxspec_test(worker_dir, (names, default_params), cpus=cpus)

    # Allow Xspec optimization
    worker_data['optimize'] = True
    with open(f'{worker_dir}worker_data.pickle', 'wb') as file:
        pickle.dump(worker_data, file)

    # Encoder + Xspec performance
    print('\nTesting Encoder + Fitting...')
    pyxspec_test(worker_dir, predictions_path, cpus=cpus, job_name='Encoder_Xspec_output')

    # Default + Xspec performance
    print('\nTesting Defaults + Fitting...')
    pyxspec_test(worker_dir, (names, default_params), cpus=cpus, job_name='Default_Xspec_output')


def predict_parameters(config: dict | str = '../config.yaml') -> np.ndarray:
    """
    Predicts parameters using the encoder & saves the results to a file

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or file path to the configuration file

    Returns
    -------
    ndarray
        Spectra names and parameter predictions
    """
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', config)

    if config['training']['encoder-save']:
        config['training']['encoder-load'] = config['training']['encoder-save']

    (_, loader), encoder = initialization(
        config['training']['encoder-name'],
        config,
    )[2:]

    initial_time = time()
    output_path = config['output']['parameter-predictions-path']
    names = []
    params = []
    log_params = loader.dataset.dataset.log_params
    param_transform = loader.dataset.dataset.transform[1]

    encoder.eval()

    # Initialize processes for multiprocessing of encoder loss calculation
    with torch.no_grad():
        for names_batch, spectra, *_ in loader:
            names.extend(names_batch)
            param_batch = encoder(spectra.to(get_device()[1]))
            params.append(param_batch)

    params = torch.cat(params).cpu().numpy() * param_transform[1] + param_transform[0]

    if log_params:
        params[:, log_params] = 10 ** params[:, log_params]

    output = np.hstack((np.expand_dims(names, axis=1), params))
    print(f'Parameter prediction time: {time() - initial_time:.3e} s')
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
    batch_size = config['training']['batch-size']
    learning_rate = config['training']['learning-rate']
    networks_dir = config['training']['network-configs-directory']
    spectra_path = config['data'][f'{network_type}-data-path']
    num_params = config['model']['parameters-number']
    log_params = config['model']['log-parameters']
    states_dir = config['output']['network-states-directory']

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
        batch_size=batch_size,
        transform=transform,
        indices=indices,
    )

    # Initialize network
    network = Network(
        loaders[0].dataset[0][1].size(0),
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

    # Data paths
    e_data_path = config['data']['encoder-data-path']
    d_data_path = config['data']['decoder-data-path']

    # Output paths
    predictions_path = config['output']['parameter-predictions-path']
    states_dir = config['output']['network-states-directory']
    plots_dir = config['output']['plots-directory']
    worker_dir = config['output']['worker-directory']

    # Initialize Matplotlib backend
    mpl.use('qt5agg')

    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create worker directory
    if not os.path.exists(worker_dir):
        os.makedirs(worker_dir)

    # Create folder to save network progress
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)

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
    plots.plot_training('Decoder', plots_dir, *decoder_return)

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
    plots.plot_training('Autoencoder', plots_dir, *encoder_return)

    # Plot linear weight mappings
    plots.plot_linear_weights(config, decoder)
    plots.plot_encoder_pgstats(f'{plots_dir}{worker_dir}Encoder_Xspec_output.csv', config)
    plots.plot_pgstat_iterations(
        [f'{worker_dir}Encoder_Xspec_output_60.csv',
         f'{worker_dir}Default_Xspec_output_60.csv'],
        ['Encoder', 'Defaults'],
        config,
    )

    # Generate parameter predictions
    predict_parameters(config=config)
    plots.plot_param_comparison(config)
    plots.plot_param_distribution(
        'Decoder_Param_Distribution',
        [d_data_path, predictions_path],
        config,
        y_axis=False,
        labels=['Target', 'Prediction'],
    )
    plots.plot_param_distribution(
        'Encoder_Param_Distribution',
        [e_data_path, predictions_path],
        config,
        y_axis=False,
        labels=['Target', 'Prediction'],
    )
    plots.plot_param_pairs(
        (e_data_path, predictions_path),
        config,
        labels=('Targets', 'Predictions'),
    )

    # Calculate saliencies
    decoder_saliency(d_loaders[1], decoder)
    saliency_output = autoencoder_saliency(e_loaders[1], encoder, decoder)
    plots.plot_saliency(plots_dir, *saliency_output)

    if tests:
        pyxspec_tests(config, e_loaders[1].dataset)


if __name__ == '__main__':
    main()
