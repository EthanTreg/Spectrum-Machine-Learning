"""
Creates several plots
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.axes import Axes
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fspnet.utils.data import load_x_data
from fspnet.utils.utils import data_normalization

MAJOR = 24
MINOR = 20


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
    plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
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


def _plot_reconstructions(
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
    twin_axis.hist(data_twin, bins=100, alpha=0.5, density=True, label='Predicted', color='orange')

    return twin_axis


def plot_saliency(plots_dir: str, spectra: ndarray, predictions: ndarray, saliencies: ndarray):
    """
    Plots saliency map for the autoencoder

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    spectra : Tensor
        Target spectra
    saliencies : Tensor
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
        2,
        4,
        figsize=(24, 12),
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
    _, axes = plt.subplot_mosaic('AABBCC;DDDEEE', constrained_layout=True, figsize=(16, 9))

    # Plot each parameter
    for i, axis in enumerate(axes.values()):
        value_range = [
            min(np.min(target[:, i]), np.min(predictions[:, i])),
            max(np.max(target[:, i]), np.max(predictions[:, i]))
        ]
        axis.scatter(target[:, i], predictions[:, i], alpha=0.8)
        axis.plot(value_range, value_range, color='k')
        axis.set_title(param_names[i])

    plt.savefig(f'{plots_dir}Parameter_Comparison.png', transparent=False)


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
    x_data = load_x_data(spectra.shape[-1])

    # Initialize reconstructions plots
    _, axes = plt.subplots(2, 2, figsize=(24, 12), sharex='col', gridspec_kw={
        'top': 0.92,
        'bottom': 0.07,
        'left': 0.08,
        'right': 0.99,
        'hspace': 0.05,
        'wspace': 0.15,
    })

    axes = axes.flatten()

    # Plot reconstructions
    for spectrum, output, axis in zip(spectra, outputs, axes):
        main_axis = _plot_reconstructions(x_data, spectrum, output, axis)

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
        bbox_to_anchor=(0.5, 0.92),
        fontsize=MAJOR,
        markerscale=2,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)

    plt.savefig(f'{plots_dir}{prefix}_Reconstructions.png', transparent=False)

    # Plot loss over epochs
    _plot_loss(losses[0], losses[1])
    plt.savefig(f'{plots_dir}{prefix}_Loss.png', transparent=False)


def plot_param_distribution(
        plots_dir: str,
        param_names: list[str],
        params: ndarray,
        loader: DataLoader):
    """
    Plots histogram of each parameter for both true and predicted

    Parameters
    ----------
    plots_dir : string
        Directory to plots
    param_names : list[string]
        Names for each parameter
    params : Tensor
        Parameter predictions from CNN
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    """
    log_params = loader.dataset.dataset.log_params
    param_transform = loader.dataset.dataset.transform[1]
    params_real = loader.dataset.dataset.params.cpu().numpy()
    # param_names = ['nH', r'$\Gamma', 'FracSctr', r'$T_{max}$', 'Norm']

    _, axes = plt.subplot_mosaic('AABBCC;DDDEEE', figsize=(16, 9))

    params[:, log_params] = np.log10(params[:, log_params])
    params = list(np.rollaxis(params, axis=1))
    params_real = params_real * param_transform[1] + param_transform[0]
    params_real = list(np.rollaxis(params_real, axis=1))

    # Plot subplots
    for i, (title, axis) in enumerate(zip(param_names, axes.values())):
        twin_axis = _plot_histogram(title, params_real[i], params[i], axis)

    legend = plt.figlegend(
        axes['A'].get_legend_handles_labels()[0] + twin_axis.get_legend_handles_labels()[0],
        axes['A'].get_legend_handles_labels()[1] + twin_axis.get_legend_handles_labels()[1],
        fontsize=MAJOR,
        bbox_to_anchor=(0.95, 0.45),
    )
    legend.get_frame().set_alpha(None)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}Param_Distribution')


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
