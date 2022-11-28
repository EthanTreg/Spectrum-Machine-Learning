import os
import json

import torch
import joblib
import optuna

from src.spectrum_fit_cnn import train, validate
from src.utils.networks import Network
from src.utils.utils import progress_bar
from src.utils.data_utils import data_initialisation


def build_network(config_path: str, filters: list[int], conv_layers: list[int]):
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
    with open(f'{config_path[:-5]}_temp.json', 'w', encoding='utf-8') as file:
        json.dump(new_file, file, ensure_ascii=False, indent=4)


def objective(
        trial: optuna.trial.Trial,
        num_epochs: int,
        loaders: tuple,
        device: torch.device) -> float:
    """
    Objective function that trials parameters to optimize hyperparameters

    Parameters
    ----------
    trial : Trial
        Optuna trial object
    num_epochs : integer
        Number of epochs to train for
    loaders : tuple
        Train dataloader and validation dataloader
    device : device
        Which device type PyTorch should use

    Returns
    -------
    float
        Final validation loss of the trial network
    """
    # Variables
    config_dir = '../network_configs/'

    # Trial parameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    conv_layers_1 = trial.suggest_int('conv_layers_1', 1, 5)
    conv_layers_2 = trial.suggest_int('conv_layers_2', 1, 5)
    filters_1 = trial.suggest_int('filters_1', 1, 128)
    filters_2 = trial.suggest_int('filters_2', 1, 128)

    # Build temporary network for testing
    build_network(
        f'{config_dir}/Decoder.json',
        [filters_1, filters_2],
        [conv_layers_1, conv_layers_2]
    )

    # Initialize decoder
    decoder = Network(
        loaders[0].dataset[0][1].size(0),
        loaders[0].dataset[0][0].size(0),
        learning_rate,
        'Decoder',
        config_dir,
    ).to(device)

    # Train for each epoch
    for epoch in range(num_epochs):
        # Train CNN
        train(device, loaders[0], decoder)
        loss = validate(device, loaders[1], decoder)[0]
        decoder.scheduler.step(loss)
        trial.report(loss, epoch)

        # End bad trials early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        progress_bar(epoch, num_epochs, f'Loss: {loss:.3e}\tEpoch: {epoch}')

    # Final validation
    loss = validate(device, loaders[1], decoder)[0]
    print(f'\nValidation loss: {loss:.3e}')

    return loss


def main():
    """
    Main function for optimizing networks
    """
    # Variables
    load = True
    save = True
    num_epochs = 50
    num_trials = 30
    min_trials = 5
    synth_path = '../data/synth_spectra.npy'
    synth_params_path = '../data/synth_spectra_params.npy'
    log_params = [0, 2, 3, 4]

    # Constants
    initial = 0
    model_dir = '../model_states/'

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    # Initialize datasets
    loaders = data_initialisation(
        synth_path,
        synth_params_path,
        log_params,
        kwargs,
    )

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Find most recent file
    while os.path.exists(model_dir + f'trial_{initial}'):
        initial += 1

    if initial != 0 and load:
        study, pruner = joblib.load(model_dir + f'trial_{initial - 1}')
    else:
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

    for i in range(initial, num_trials):
        study.optimize(lambda x: objective(x, num_epochs, loaders, device), n_trials=1)

        if save:
            state = {'study': study, 'pruner': pruner}
            joblib.dump(state, model_dir + f'trial_{initial + i}')

    print(study.best_params)

    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_contour(study).show()
    optuna.visualization.plot_parallel_coordinate(
        study,
        params=['learning_rate', 'filters_1', 'filters_2', 'conv_layers_1', 'conv_layers_2']
    ).show()


if __name__ == '__main__':
    main()
