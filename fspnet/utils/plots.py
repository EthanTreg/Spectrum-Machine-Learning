"""
Creates several plots
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor
from torch.utils.data import DataLoader

from fspnet.utils.data import load_x_data

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
        x_data: np.ndarray,
        y_data: np.ndarray,
        y_recon: np.ndarray,
        axes: Axes):
    """
    Plots reconstructions for a given epoch

    Parameters
    ----------
    x_data : ndarray
        Energy values
    y_data : ndarray
        Spectrum
    y_recon : ndarray
        Reconstructed Spectrum
    axes : Axes
        Plot axes
    """
    axes.scatter(x_data, y_data, label='Spectrum')
    axes.scatter(x_data, y_recon, label='Reconstruction')
    axes.locator_params(axis='y', nbins=5)
    axes.tick_params(axis='both', labelsize=MINOR)


def _plot_histogram(title: str, data: np.ndarray, data_twin: np.ndarray, axis: Axes) -> Axes:
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
    axis.hist(data, bins=100, alpha=0.5, density=True, label='Real')
    twin_axis.hist(data_twin, bins=100, alpha=0.5, density=True, label='Predicted', color='orange')

    return twin_axis


def plot_saliency(plots_dir: str, spectra: Tensor, prediction: Tensor, saliency: Tensor):
    """
    Plots saliency map for the autoencoder

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    spectra : Tensor
        Target spectra
    prediction : Tensor
        Predicted spectra
    saliency : Tensor
        Saliency
    """
    x_data = load_x_data(spectra.size(1))

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
    for i, axis in enumerate(axes):
        axis.scatter(x_data, prediction[i], label='Prediction')
        axis.scatter(x_data, spectra[i], c=saliency[i], cmap=plt.cm.hot, label='Target')
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
    legend.legendHandles[1].set_color('orange')

    plt.figtext(0.5, 0.02, 'Energy (keV)', ha='center', va='center', fontsize=MAJOR)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(f'{plots_dir}Saliency_Plot.png', transparent=False)


def plot_initialization(
    prefix: str,
    plots_dir: str,
    losses: tuple[list, list],
    spectra: np.ndarray,
    outputs: np.ndarray
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
    _, axes = plt.subplots(4, 4, figsize=(24, 12), sharex='col', gridspec_kw={'hspace': 0})
    axes = axes.flatten()

    # Plot reconstructions
    for i in range(min(axes.size, spectra.shape[0])):
        _plot_reconstructions(x_data, spectra[i], outputs[i], axes[i])

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

    legend = plt.figlegend(
        *axes[0].get_legend_handles_labels(),
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=MAJOR,
        markerscale=2,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig(f'{plots_dir}{prefix} Reconstructions.png', transparent=False)

    # Plot loss over epochs
    _plot_loss(losses[0], losses[1])
    plt.savefig(f'{plots_dir}{prefix} Loss.png', transparent=False)


def plot_param_distribution(
        plots_dir: str,
        param_names: list[str],
        params: np.ndarray,
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

    _, axes = plt.subplot_mosaic('AABBCC;DDDEEE', figsize=(32, 18))

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


def diff_plot(
        x_data_1: np.ndarray,
        y_data_1: np.ndarray,
        x_data_2: np.ndarray,
        y_data_2: np.ndarray):
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

    plt.title('PyXspec compared to fits', fontsize=24)
    plt.scatter(x_data_1, diff)
    plt.xlabel('Energy (keV)', fontsize=20)
    plt.ylabel('PyXspec / fits data', fontsize=20)
    plt.text(
        0.05,
        0.2,
        f'Average ratio: {round(np.mean(diff), 3)}',
        fontsize=16, transform=plt.gca().transAxes
    )


def spectrum_plot(x_bin: np.ndarray, y_bin: np.ndarray, x_px: np.ndarray, y_px: np.ndarray):
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
    plt.title('Spectrum of PyXspec & fits', fontsize=24)
    plt.xlabel('Energy (keV)', fontsize=20)
    plt.ylabel(r'Counts $s^{-1}$ $detector^{-1}$ $keV^{-1}$', fontsize=20)
    plt.scatter(x_bin, y_bin, label='Fits data', marker='x')
    plt.scatter(x_px, y_px, label='PyXspec data')
    plt.xlim([0.15, 14.5])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=20)
