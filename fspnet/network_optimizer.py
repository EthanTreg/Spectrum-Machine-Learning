"""
Optimizes the hyperparameters of the network using Optuna
"""
import os
import json
import logging as log

import torch
import joblib
import optuna
from torch.utils.data import DataLoader

from fspnet.utils.network import Network
from fspnet.utils.training import train_val
from fspnet.utils.data import data_initialisation
from fspnet.utils.utils import progress_bar, open_config


def _build_network(config_path: str, filters: list[int], conv_layers: list[int]):
    """
    Builds a temporary network based off an existing one for testing different hyperparameters

    Only varies the number of filters and number of convolutional layers per upscale section

    Parameters
    ----------
    config_path : string
        Path to existing network to base the temporary one off
    filters : list[integer]
        List of the number of filters for each convolutional layer in an upscale section
    conv_layers : list[integer]
        Number of convolutional layers per upscale section
    """
    # Constants
    conv_layer = 0

    # Open original network file
    with open(config_path, 'r', encoding='utf-8') as file:
        file = json.load(file)

    # Copy network architecture
    layers = file['layers']
    new_layers = layers.copy()

    # Change convolutional layers
    for i, layer in enumerate(layers):
        if layer['type'] == 'convolutional':
            # Change filters
            index = len(layers) - i
            new_layers[-index]['filters'] = filters[conv_layer]

            # Change number of layers
            for _ in range(conv_layers[conv_layer] - 1):
                new_layers.insert(-index, new_layers[-index])

            conv_layer += 1

            if conv_layer == len(conv_layers):
                break

    new_file = {'net': file['net'], 'layers': new_layers}

    # Save new architecture to temp file
    with open(config_path.replace('.json', '_temp.json'), 'w', encoding='utf-8') as file:
        json.dump(new_file, file, ensure_ascii=False, indent=4)


def _objective(
        trial: optuna.trial.Trial,
        loaders: tuple,
        config: dict,
        device: torch.device) -> float:
    """
    Objective function that trials parameters to optimize hyperparameters

    Parameters
    ----------
    trial : Trial
        Optuna trial object
    loaders : tuple
        Train dataloader and validation dataloader
    config : dictionary
        Configuration dictionary
    device : device
        Which device type PyTorch should use

    Returns
    -------
    float
        Final validation loss of the trial network
    """
    # Variables
    epochs = config['optimization']['epochs']
    network_name = config['optimization']['name']
    config_dir = config['optimization']['network-configs-directory']

    # Trial parameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    conv_layers_1 = trial.suggest_int('conv_layers_1', 1, 5)
    conv_layers_2 = trial.suggest_int('conv_layers_2', 1, 5)
    filters_1 = trial.suggest_int('filters_1', 1, 128)
    filters_2 = trial.suggest_int('filters_2', 1, 128)

    # Build temporary network for testing
    _build_network(
        f'{config_dir}{network_name}.json',
        [filters_1, filters_2],
        [conv_layers_1, conv_layers_2]
    )

    # Initialize decoder
    decoder = Network(
        loaders[0].dataset[0][0].size(0),
        loaders[0].dataset[0][1].size(0),
        learning_rate,
        f'{network_name}_temp',
        config_dir,
    ).to(device)

    # Train for each epoch
    for epoch in range(epochs):
        epoch += 1

        # Train CNN
        train_val(device, loaders[0], decoder)
        loss = train_val(device, loaders[1], decoder, train=False)[0]
        decoder.scheduler.step(loss)
        trial.report(loss, epoch)

        # End bad trials early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        progress_bar(epoch, epochs, f'Loss: {loss:.3e}\tEpoch: {epoch}')

    # Final validation
    loss = train_val(device, loaders[1], decoder, train=False)[0]
    print(f'Validation loss: {loss:.3e}\n')

    return loss


def optimize_network(
        config: dict,
        loaders: tuple[DataLoader, DataLoader],
        device: torch.device) -> dict:
    """
    Optimizes the hyperparameters of the network using Optuna and plots the results

    Parameters
    ----------
    config : dictionary
        Configuration dictionary
    loaders : tuple[DataLoader, DataLoader]
        Train and validation dataloaders
    device : device
        Which device type PyTorch should use

    Returns
    -------
    dictionary
        Best trial hyperparameters
    """
    # Constants
    initial = 0
    load = config['optimization']['load']
    save = config['optimization']['save']
    num_trials = config['optimization']['trials']
    network_name = config['optimization']['name']
    min_trials = config['optimization']['pruning-minimum-trials']
    model_dir = config['output']['network-states-directory']

    # Start trials
    pruner = optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(
            n_startup_trials=min_trials,
            n_warmup_steps=10,
            n_min_trials=min_trials,
        ),
        patience=3,
    )
    study = optuna.create_study(direction='minimize', pruner=pruner)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # If load, load previous trial to continue optimization
    if load:
        try:
            initial, study, pruner = joblib.load(f'{model_dir}{network_name}_optimization')
        except FileNotFoundError:
            log.warning(f'{model_dir}{network_name}_optimization does not exist\n'
                  f'No state will be loaded')

    # Train network for each trial
    for i in range(initial, num_trials):
        study.optimize(lambda x: _objective(x, loaders, config, device), n_trials=1)

        if save:
            state = {'trial': i, 'study': study, 'pruner': pruner}
            joblib.dump(state, f'{model_dir}{network_name}_optimization')

    # Display results
    print(study.best_params)
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_contour(study).show()
    optuna.visualization.plot_parallel_coordinate(
        study,
        params=['learning_rate', 'filters_1', 'filters_2', 'conv_layers_1', 'conv_layers_2']
    ).show()

    return study.best_params


def main(config_path: str = './config.yaml'):
    """
    Main function for optimizing networks

    Parameters
    ----------
    config_path : string, default = '../config.yaml'
        File path to the configuration file
    """
    # Constants
    _, config = open_config(3, config_path=config_path)
    spectra_path = config['data']['spectra-path']
    params_path = config['data']['parameters-path']
    log_params = config['optimization']['log-parameters']

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

    optimize_network(config, loaders, device)


if __name__ == '__main__':
    main()
