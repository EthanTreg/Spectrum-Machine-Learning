"""
Creates several plots
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fspnet.utils.data import load_x_data, data_normalization

MAJOR = 24
MINOR = 20
FIG_SIZE = (16, 9)


def _plot_loss(train_loss: list, val_loss: list):
    """
    Plots training and validation loss as a function of epochs

    Parameters
    ----------
    train_loss : list
        Training losses
    val_loss : list
        Validation losses
    """
    plt.figure(figsize=FIG_SIZE, constrained_layout=True)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR)
    plt.xlabel('Epoch', fontsize=MINOR)
    plt.ylabel('Loss', fontsize=MINOR)
    plt.yscale('log')
    plt.text(
        0.8, 0.75,
        f'Final loss: {val_loss[-1]:.3e}',
        fontsize=MINOR,
        transform=plt.gca().transAxes
    )

    legend = plt.legend(fontsize=MAJOR)
    legend.get_frame().set_alpha(None)


def _plot_reconstruction(
        x_data: ndarray,
        y_data: ndarray,
        y_recon: ndarray,
        axis: Axes) -> Axes:
    """
    Plots reconstructions and residuals

    Parameters
    ----------
    x_data : ndarray
        Energy values
    y_data : ndarray
        Spectrum
    y_recon : ndarray
        Reconstructed Spectrum
    axis : Axes
        Plot axes

    Returns
    -------
    Axes
        Spectrum axis
    """
    axis.scatter(x_data, y_recon - y_data, marker='x', label='Residual')
    axis.locator_params(axis='y', nbins=3)
    axis.tick_params(axis='both', labelsize=MINOR)
    axis.hlines(0, xmin=np.min(x_data), xmax=np.max(x_data), color='k')

    divider = make_axes_locatable(axis)
    axis_2 = divider.append_axes('top', size='150%', pad=0)

    axis_2.scatter(x_data, y_data, label='Spectrum')
    axis_2.scatter(x_data, y_recon, label='Reconstruction')
    axis_2.locator_params(axis='y', nbins=5)
    axis_2.tick_params(axis='y', labelsize=MINOR)
    axis_2.set_xticks([])

    return axis_2


def _plot_reconstructions(spectra: ndarray, outputs: ndarray):
    """
    Plots reconstructions and residuals for 4 spectra

    Parameters
    ----------
    spectra : ndarray
        True spectra
    outputs : ndarray
        Predicted spectra
    """
    x_data = load_x_data(spectra.shape[-1])

    # Initialize reconstructions plots
    _, axes = plt.subplots(2, 2, figsize=FIG_SIZE, sharex='col', gridspec_kw={
        'top': 0.92,
        'bottom': 0.08,
        'left': 0.09,
        'right': 0.99,
        'hspace': 0.05,
        'wspace': 0.15,
    })

    axes = axes.flatten()

    # Plot reconstructions
    for spectrum, output, axis in zip(spectra, outputs, axes):
        main_axis = _plot_reconstruction(x_data, spectrum, output, axis)

    plt.figtext(0.5, 0.02, 'Energy (keV)', ha='center', va='center', fontsize=MAJOR)
    plt.figtext(
        0.02,
        0.5,
        'Scaled Log Counts',
        ha='center',
        va='center',
        rotation='vertical',
        fontsize=MAJOR,
    )

    labels = np.hstack((main_axis.get_legend_handles_labels(), axes[0].get_legend_handles_labels()))
    legend = plt.figlegend(
        *labels,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.91),
        fontsize=MAJOR,
        markerscale=2,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)


def _plot_histogram(title: str, data: ndarray, data_twin: ndarray, axis: Axes) -> Axes:
    """
    Plots a histogram subplot with twin data

    Parameters
    ----------
    title : string
        Title of subplot
    data : ndarray
        Primary data to plot
    data_twin : ndarray
        Secondary data to plot
    axis : Axes
        Axis to plot on

    Returns
    -------
    Axes
        Twin axis
    """
    twin_axis = axis.twinx()

    axis.set_title(title, fontsize=MAJOR)
    axis.hist(data, bins=100, alpha=0.5, density=True, label='Target')
    axis.tick_params(labelsize=MINOR)
    twin_axis.hist(data_twin, bins=100, alpha=0.5, density=True, label='Predicted', color='orange')
    twin_axis.tick_params(labelsize=MINOR)

    return twin_axis


def plot_saliency(plots_dir: str, spectra: ndarray, predictions: ndarray, saliencies: ndarray):
    """
    Plots saliency map for the autoencoder

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    spectra : ndarray
        Target spectra
    predictions : ndarray
        Predicted spectra
    saliencies : ndarray
        Saliency
    """
    # Constants
    alpha = 0.8
    cmap = plt.cm.hot
    x_data = load_x_data(spectra.shape[1])

    x_regions = x_data[::12]
    saliencies = np.mean(saliencies.reshape(saliencies.shape[0], -1, 12), axis=-1)
    saliencies = data_normalization(saliencies, mean=False, axis=1)[0] * 0.9 + 0.05

    # Initialize e_saliency plots
    _, axes = plt.subplots(
        2, 4,
        figsize=FIG_SIZE,
        sharex='col',
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    axes = axes.flatten()

    # Plot each saliency map
    for i, (
            axis,
            spectrum,
            prediction,
            saliency
    ) in enumerate(zip(axes, spectra, predictions, saliencies)):
        for j, (x_region, saliency_region) in enumerate(zip(x_regions[:-1], saliency[:-1])):
            axis.axvspan(x_region, x_regions[j + 1], color=cmap(saliency_region), alpha=alpha)

        axis.axvspan(x_regions[-1], x_data[-1], color=cmap(saliency[-1]), alpha=alpha)

        axis.scatter(x_data, spectrum, label='Target')
        axis.scatter(x_data, prediction, color='g', label='Prediction')
        axis.text(0.9, 0.9, i + 1, fontsize=MAJOR, transform=axis.transAxes)
        axis.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)

    legend = plt.figlegend(
        *axes[0].get_legend_handles_labels(),
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.92),
        fontsize=MAJOR,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)

    plt.figtext(0.5, 0.02, 'Energy (keV)', ha='center', va='center', fontsize=MAJOR)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(f'{plots_dir}Saliency_Plot.png', transparent=False)


def plot_param_comparison(
        plots_dir: str,
        param_names: list[str],
        target: ndarray,
        predictions: ndarray):
    """
    Plots predictions against target for each parameter

    Parameters:
    ----------
    plots_dir : string
        Directory to save plots
    param_names : list[string]
        List of parameter names
    target : ndarray
        Target parameters
    predictions : ndarray
        Parameter predictions
    """
    _, axes = plt.subplot_mosaic('AABBCC;DDDEEE', constrained_layout=True, figsize=FIG_SIZE)

    # Plot each parameter
    for i, axis in enumerate(axes.values()):
        value_range = [
            min(np.min(target[:, i]), np.min(predictions[:, i])),
            max(np.max(target[:, i]), np.max(predictions[:, i]))
        ]
        axis.scatter(target[:1000, i], predictions[:1000, i], alpha=0.2)
        axis.plot(value_range, value_range, color='k')
        axis.set_title(param_names[i], fontsize=MAJOR)
        axis.tick_params(labelsize=MINOR)

    plt.savefig(f'{plots_dir}Parameter_Comparison.png', transparent=False)


def plot_param_distribution(name: str, data_paths: list[str], config: dict):
    """
    Plots histogram of each parameter for both true and predicted

    Parameters
    ----------
    name : string
        Name to save the plot
    data_paths : list[string]
        Path to the parameters
    config : dictionary
        Configuration dictionary
    """
    params = []
    log_params = config['model']['log-parameters']
    plots_dir = config['output']['plots-directory']
    param_names = config['model']['parameter-names']

    _, axes = plt.subplot_mosaic('AABBCC;DDDEEE', figsize=FIG_SIZE)

    for data_path in data_paths:
        if '.csv' in data_path:
            data = np.loadtxt(
                data_path,
                delimiter=',',
                usecols=range(1, 6),
            )
        elif '.pickle' in data_path:
            with open(data_path, 'rb') as file:
                data = np.array(pickle.load(file)['params'])
        else:
            data = np.load(data_path)

        if log_params:
            data[:, log_params] = np.log10(data[:, log_params])

        params.append(np.rollaxis(data, axis=1))

    # Plot subplots
    for (
        title,
        *param,
        axis,
    ) in zip(param_names, *params, axes.values()):
        twin_axis = _plot_histogram(title, *param, axis)

    legend = plt.figlegend(
        *np.hstack((axes['A'].get_legend_handles_labels(), twin_axis.get_legend_handles_labels())),
        fontsize=MAJOR,
        bbox_to_anchor=(0.95, 0.45),
    )
    legend.get_frame().set_alpha(None)
    plt.tight_layout()
    plt.savefig(plots_dir + name)


def plot_linear_weights(config: dict, weights: ndarray):
    """
    Plots the mappings of the weights from the lowest dimension
    to a high dimension for the linear layers

    Parameters
    ----------
    config : dictionary
        Configuration dictionary
    weights : ndarray
        Mappings of low dimension to high dimension
    """
    plots_dir = config['output']['plots-directory']
    param_names = config['model']['parameter-names']

    _, axes = plt.subplot_mosaic(
        'AABBCC;DDDEEE',
        figsize=FIG_SIZE,
        gridspec_kw={
            'top': 0.95,
            'bottom': 0.08,
            'left': 0.06,
            'right': 0.99,
            'hspace': 0.2,
            'wspace': 0.75,
        },
    )

    for title, weight, axis in zip(param_names, weights, axes.values()):
        axis.scatter(load_x_data(weight.size), weight)
        axis.set_title(title, fontsize=MAJOR)
        axis.tick_params(labelsize=MINOR)

    plt.figtext(0.5, 0.02, 'Energy (keV)', ha='center', va='center', fontsize=MAJOR)
    plt.savefig(f'{plots_dir}Linear_Weights_Mappings.png')


def plot_training(
    prefix: str,
    plots_dir: str,
    losses: tuple[list, list],
    spectra: ndarray,
    outputs: ndarray
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
    # Plot reconstructions if output is spectra reconstructions
    if spectra.shape[1] == outputs.shape[1]:
        _plot_reconstructions(spectra, outputs)
        plt.savefig(f'{plots_dir}{prefix}_Reconstructions.png', transparent=False)

    # Plot loss over epochs
    _plot_loss(losses[0], losses[1])
    plt.savefig(f'{plots_dir}{prefix}_Loss.png', transparent=False)


def plot_difference(
        x_data_1: ndarray,
        y_data_1: ndarray,
        x_data_2: ndarray,
        y_data_2: ndarray):
    """
    Plots the ratio between two data sets (set 1 / set 2)

    Parameters
    ----------
    x_data_1 : ndarray
        x values for first data set
    y_data_1 : ndarray
        y values for first data set
    x_data_2 : ndarray
        x values for second data set
    y_data_2 : ndarray
        y values for second data set
    """
    matching_indices = np.array((), dtype=int)

    for i in x_data_1:
        matching_indices = np.append(matching_indices, np.argmin(np.abs(x_data_2 - i)))

    diff = y_data_1 / y_data_2[matching_indices]

    plt.title('PyXspec compared to fits', fontsize=MAJOR)
    plt.scatter(x_data_1, diff)
    plt.xlabel('Energy (keV)', fontsize=MINOR)
    plt.ylabel('PyXspec / fits data', fontsize=MINOR)
    plt.text(
        0.05,
        0.2,
        f'Average ratio: {round(np.mean(diff), 3)}',
        fontsize=MINOR, transform=plt.gca().transAxes
    )


def plot_spectrum(x_bin: ndarray, y_bin: ndarray, x_px: ndarray, y_px: ndarray):
    """
    Plots the spectrum of PyXspec & fits data

    Parameters
    ----------
    x_bin : ndarray
        Binned x data from fits file
    y_bin : ndarray
        Binned y data from fits file
    x_px : ndarray
        x data from PyXspec
    y_px : ndarray
        y data from PyXspec
    """
    plt.title('Spectrum of PyXspec & fits', fontsize=MAJOR)
    plt.xlabel('Energy (keV)', fontsize=MINOR)
    plt.ylabel(r'Counts $s^{-1}$ $detector^{-1}$ $keV^{-1}$', fontsize=MINOR)
    plt.scatter(x_bin, y_bin, label='Fits data', marker='x')
    plt.scatter(x_px, y_px, label='PyXspec data')
    plt.xlim([0.15, 14.5])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=MINOR)
