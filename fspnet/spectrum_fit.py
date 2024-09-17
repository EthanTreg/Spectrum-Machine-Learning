"""
Main spectral machine learning module
"""
import os
import pickle
from typing import Any, Self, BinaryIO

import torch
import numpy as np
import netloader.networks as nets
from netloader.network import Network
from netloader.utils import transforms
from netloader.utils.utils import save_name, get_device
from torch.utils.data import DataLoader
from torch import nn, Tensor
from numpy import ndarray

from fspnet.utils import plots
from fspnet.utils.utils import open_config
from fspnet.utils.data import SpectrumDataset, loader_init
from fspnet.utils.analysis import autoencoder_saliency, decoder_saliency, pyxspec_test


class AutoencoderNet(nn.Module):
    """
    Makes an autoencoder architecture from an encoder and decoder

    Attributes
    ----------
    name : str
        Name of the autoencoder
    checkpoints : list[Tensor]
        Outputs from each checkpoint
    net : ModuleList
        Network construction
    kl_loss : Tensor, default = 0
        KL divergence loss on the latent space, if using a sample layer

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the network
    to(*args, **kwargs) -> Network
        Moves and/or casts the parameters and buffers
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, name: str = 'autoencoder'):
        """
        Parameters
        ----------
        encoder : Module
            Encoder network
        decoder : Module
            Decoder network
        name : str, default = 'autoencoder'
            Name of the network
        """
        super().__init__()
        self.name: str = name
        self.checkpoints: list[torch.Tensor] = []
        self.kl_loss: torch.Tensor = torch.tensor(0.)
        self.net: nn.ModuleList = nn.ModuleList([encoder, decoder])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder

        Parameters
        ----------
        x : (N,...) list[Tensor] | Tensor
            Input tensor(s) with batch size N

        Returns
        -------
        (N,...) list[Tensor] | Tensor
            Output tensor from the network
        """
        x = self.net[0](x)
        self.checkpoints.append(x.clone())

        if hasattr(self.net[0], 'kl_loss'):
            self.kl_loss = self.net[0].kl_loss

        return self.net[1](x)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self.kl_loss = self.kl_loss.to(*args, **kwargs)
        self.net[0].to(*args, **kwargs)
        self.net[1].to(*args, **kwargs)
        return self


def pyxspec_tests(config: dict[str, Any], data: dict[str, ndarray]) -> None:
    """
    Tests the PGStats of the different fitting methods using PyXspec

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    data : dict[str, ndarray]
        ids : ndarray
            Files names of the FITS spectra corresponding to the parameters
        latent : ndarray
            Parameter predictions
        targets : ndarray
            Best fit parameters
    """
    # Initialize variables
    cpus: int = config['training']['cpus']
    python: str = config['training']['python-path']
    worker_dir: str = config['output']['worker-directory']
    default_params: list[float] = config['model']['default-parameters']
    worker_data: dict[str, Any] = {
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
    file: BinaryIO

    # Save worker variables
    with open(f'{worker_dir}worker_data.pickle', 'wb') as file:
        pickle.dump(worker_data, file)

    # Encoder validation performance
    print('\nTesting Encoder...')
    pyxspec_test(
        worker_dir,
        data['ids'],
        data['latent'],
        cpus=cpus,
        job_name='Encoder_output',
        python_path=python,
    )

    # Xspec performance
    print('\nTesting Xspec...')
    pyxspec_test(
        worker_dir,
        data['ids'],
        data['targets'],
        cpus=cpus,
        job_name='Xspec_output',
        python_path=python,
    )

    # Default performance
    print('\nTesting Defaults...')
    pyxspec_test(
        worker_dir,
        data['ids'],
        np.repeat([default_params], len(data['ids']), axis=0),
        cpus=cpus,
        python_path=python,
    )

    # Allow Xspec optimization
    worker_data['optimize'] = True
    with open(f'{worker_dir}worker_data.pickle', 'wb') as file:
        pickle.dump(worker_data, file)

    # Encoder + Xspec performance
    print('\nTesting Encoder + Fitting...')
    pyxspec_test(
        worker_dir,
        data['ids'],
        data['latent'],
        cpus=cpus,
        job_name='Encoder_Xspec_output',
        python_path=python,
    )

    # Default + Xspec performance
    print('\nTesting Defaults + Fitting...')
    pyxspec_test(
        worker_dir,
        data['ids'],
        data['targets'],
        cpus=cpus,
        job_name='Default_Xspec_output',
        python_path=python,
    )


def net_init(
        datasets: tuple[SpectrumDataset, SpectrumDataset],
        config: str | dict[str, Any] = '../config.yaml',
) -> tuple[nets.BaseNetwork, nets.BaseNetwork]:
    """
    Initialises the network

    Parameters
    ----------
    datasets : tuple[SpectrumDataset, SpectrumDataset]
        Encoder and decoder datasets
    config : string | dictionary, default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    tuple[BaseNetwork, BaseNetwork]
        Constructed decoder and autoencoder
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Load config parameters
    e_save_num = config['training']['encoder-save']
    e_load_num = config['training']['encoder-load']
    d_save_num = config['training']['decoder-save']
    d_load_num = config['training']['decoder-load']
    learning_rate = config['training']['learning-rate']
    encoder_name = config['training']['encoder-name']
    decoder_name = config['training']['decoder-name']
    description = config['training']['network-description']
    networks_dir = config['training']['network-configs-directory']
    log_params = config['model']['log-parameters']
    states_dir = config['output']['network-states-directory']
    device = get_device()[1]

    if d_load_num:
        decoder = nets.load_net(d_load_num, states_dir, decoder_name)
        decoder.description = description
        decoder.save_path = save_name(d_save_num, states_dir, decoder_name)
        transform = decoder.header['targets']
        param_transform = decoder.in_transform
    else:
        transform = transforms.MultiTransform([
            transforms.NumpyTensor(),
            transforms.MinClamp(dim=-1),
            transforms.Log(),
        ])
        transform.transforms.append(transforms.Normalise(
            transform(datasets[1].spectra),
            mean=False,
        ))
        param_transform = transforms.MultiTransform([
            transforms.NumpyTensor(),
            transforms.MinClamp(dim=0, idxs=log_params),
            transforms.Log(idxs=log_params),
        ])
        param_transform.transforms.append(transforms.Normalise(
            param_transform(datasets[1].params),
            dim=0,
        ))
        decoder = Network(
            decoder_name,
            networks_dir,
            list(datasets[1][0][1].shape),
            list(datasets[1][0][2].shape),
        )
        decoder = nets.Decoder(
            d_save_num,
            states_dir,
            decoder,
            learning_rate=learning_rate,
            description=description,
            verbose='epoch',
            transform=transform,
        )
        decoder.in_transform = param_transform

    if e_load_num:
        net = nets.load_net(e_load_num, states_dir, encoder_name)
        net.description = description
        net.save_path = save_name(e_save_num, states_dir, encoder_name)
    else:
        net = Network(
            encoder_name,
            networks_dir,
            list(datasets[0][0][2].shape),
            list(datasets[0][0][1].shape),
        )
        net = nets.Autoencoder(
            e_save_num,
            states_dir,
            AutoencoderNet(net, decoder.net, name=encoder_name),
            learning_rate=learning_rate,
            description=description,
            verbose='epoch',
            transform=transform,
            latent_transform=param_transform,
        )
        net.bound_loss = 0
        net.kl_loss = 0
        # net = nets.Encoder(
        #     e_save_num,
        #     states_dir,
        #     net,
        #     learning_rate=learning_rate,
        #     description=description,
        #     transform=param_transform,
        # )
        # net.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     net.optimiser,
        #     factor=0.5,
        #     min_lr=1e-5,
        # )
        # net._loss_function = encoder_loss

    for dataset in datasets:
        dataset.spectra, dataset.uncertainty = transform(
            dataset.spectra,
            uncertainty=dataset.uncertainty,
        )
        dataset.params = param_transform(dataset.params)
    return decoder.to(device), net.to(device)


def init(config: dict | str = '../config.yaml') -> tuple[
        tuple[DataLoader, DataLoader],
        tuple[DataLoader, DataLoader],
        nets.BaseNetwork,
        nets.BaseNetwork]:
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    config : dictionary | string, default = '../config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    tuple[tuple[Dataloader, Dataloader], tuple[Dataloader, Dataloader], BaseNetwork, BaseNetwork]
        Train & validation dataloaders for decoder and autoencoder, decoder, and autoencoder
    """
    if isinstance(config, str):
        _, config = open_config('main', config)

    # Load config parameters
    batch_size = config['training']['batch-size']
    val_frac = config['training']['validation-fraction']
    e_data_path = config['data']['encoder-data-path']
    d_data_path = config['data']['decoder-data-path']
    log_params = config['model']['log-parameters']

    # Fetch dataset & network
    e_dataset = SpectrumDataset(e_data_path, log_params)
    d_dataset = SpectrumDataset(d_data_path, log_params)
    decoder, net = net_init((e_dataset, d_dataset), config)

    # Initialise datasets
    e_loaders = loader_init(e_dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    d_loaders = loader_init(d_dataset, batch_size=batch_size, val_frac=val_frac, idxs=decoder.idxs)
    net.idxs = e_dataset.idxs
    decoder.idxs = d_dataset.idxs
    return e_loaders, d_loaders, decoder, net


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

    # Model variables
    log_params = config['model']['log-parameters']
    param_names = config['model']['parameter-names']

    # Output paths
    predictions_path = config['output']['parameter-predictions-path']
    states_dir = config['output']['network-states-directory']
    plots_dir = config['output']['plots-directory']
    worker_dir = config['output']['worker-directory']

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
    e_loaders, d_loaders, decoder, net = init(config)

    # Train decoder
    decoder.training(num_epochs, d_loaders)
    plots.plot_performance(
        'Loss',
        decoder.losses[1],
        plots_dir=f'{plots_dir}Decoder_',
        train=decoder.losses[0],
    )
    data = decoder.predict(d_loaders[-1])
    plots.plot_reconstructions(
        data['targets'][:, 0],
        data['preds'],
        plots_dir=f'{plots_dir}Decoder_',
    )

    net.net.net[1].requires_grad_(False)
    net.training(num_epochs, e_loaders)
    plots.plot_performance(
        'Loss',
        net.losses[1],
        plots_dir=f'{plots_dir}Autoencoder_',
        train=net.losses[0],
    )
    data = net.predict(e_loaders[0])
    plots.plot_reconstructions(
        data['inputs'][:, 0],
        data['preds'],
        plots_dir=f'{plots_dir}Autoencoder_',
    )

    # Plot linear weight mappings
    plots.plot_linear_weights(param_names, net.net, plots_dir=plots_dir)

    # WARNING: Not updated yet to use PyTorch-Network-Loader
    # plots.plot_encoder_pgstats(f'{plots_dir}{worker_dir}Encoder_Xspec_output.csv', config)
    # plots.plot_pgstat_iterations(
    #     [f'{worker_dir}Encoder_Xspec_output_60.csv',
    #      f'{worker_dir}Default_Xspec_output_60.csv'],
    #     ['Encoder', 'Defaults'],
    #     config,
    # )

    # Generate parameter predictions
    plots.plot_param_comparison(
        log_params,
        param_names,
        data['targets'],
        data['latent'],
        # data['preds'],
        plots_dir=plots_dir,
    )
    plots.plot_multi_plot(
        ['Target', 'Prediction'],
        [data['targets'], data['latent']],
        # [data['targets'], data['preds']],
        plots.plot_param_distribution,
        plots_dir=f'{plots_dir}Param_Distribution_',
        y_axis=False,
        log_params=log_params,
        param_names=param_names,
    )
    plots.plot_multi_plot(
        ['Targets', 'Predictions'],
        [data['targets'], data['latent']],
        # [data['targets'], data['preds']],
        plots.plot_param_pairs,
        plots_dir=f'{plots_dir}Pair_Plot_',
        log_params=log_params,
        param_names=param_names,
    )

    # Calculate saliencies
    # WARNING: Not updated yet to use PyTorch-Network-Loader
    # decoder_saliency(d_loaders[1], decoder)
    # saliency_output = autoencoder_saliency(e_loaders[1], encoder, decoder)
    # plots.plot_saliency(plots_dir, *saliency_output)

    if tests:
        pyxspec_tests(config, data)


if __name__ == '__main__':
    main()
