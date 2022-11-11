import json

import torch
import optuna

from src.spectrum_fit_cnn import train, validate
from src.utils.networks import Decoder
from src.utils.utils import progress_bar
from src.utils.data_utils import data_initialisation
from src.utils.network_utils import network_initialisation


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
    config_path = '../network_configs/decoder.json'

    # Trial parameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    conv_layers_1 = trial.suggest_int('conv_layers_1', 1, 5)
    conv_layers_2 = trial.suggest_int('conv_layers_2', 1, 5)
    filters_1 = trial.suggest_int('filters_1', 1, 128)
    filters_2 = trial.suggest_int('filters_2', 1, 128)

    build_network(config_path, [filters_1, filters_2], [conv_layers_1, conv_layers_2])

    # Initialize decoder
    decoder, optimizer, scheduler = network_initialisation(
        loaders[0].dataset[0][0].size(0),
        learning_rate,
        (loaders[0].dataset[0][1].size(0), f'{config_path[:-5]}_temp.json'),
        Decoder,
        device,
    )

    # Train for each epoch
    for epoch in range(num_epochs):
        # Train CNN
        train(device, loaders[0], decoder, optimizer)
        loss = validate(device, loaders[1], decoder)[0]
        scheduler.step(loss)
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
    num_epochs = 50
    num_trials = 5
    synth_path = '../data/synth_spectra.npy'
    synth_params_path = '../data/synth_spectra_params.npy'
    log_params = [0, 2, 3, 4]

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

    # Start trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=10, n_min_trials=2)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(lambda x: objective(x, num_epochs, loaders, device), n_trials=num_trials)

    print(study.best_params)

    fig1 = optuna.visualization.plot_param_importances(study)
    fig2 = optuna.visualization.plot_contour(study, params=['filters_1', 'filters_2'])
    fig3 = optuna.visualization.plot_contour(study, params=['conv_layers_1', 'conv_layers_2'])
    fig4 = optuna.visualization.plot_parallel_coordinate(
        study,
        params=['learning_rate', 'filters_1', 'filters_2', 'conv_layers_1', 'conv_layers_2']
    )
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()


if __name__ == '__main__':
    main()
