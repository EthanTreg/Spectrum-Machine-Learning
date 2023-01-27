import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor
from torch.utils.data import DataLoader

from src.utils.data_utils import load_x_data


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
    major = 28

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
        axis.text(0.9, 0.9, i + 1, fontsize=major, transform=axis.transAxes)
        axis.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)

    legend = plt.figlegend(
        *axes[0].get_legend_handles_labels(),
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.92),
        fontsize=major,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)
    legend.legendHandles[1].set_color('orange')

    plt.figtext(0.5, 0.02, 'Energy (keV)', ha='center', va='center', fontsize=major)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(f'{plots_dir}Saliency_Plot.png', transparent=False)


def plot_loss(train_loss: list, val_loss: list):
    """
    Plots training and validation loss as a function of epochs

    Parameters
    ----------
    train_loss : list
        Training losses
    val_loss : list
        Validation losses
    """
    major = 24
    minor = 20

    plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=minor)
    plt.ylabel('Loss', fontsize=minor)
    plt.yscale('log')
    plt.text(
        0.8, 0.75,
        f'Final loss: {val_loss[-1]:.3e}',
        fontsize=minor,
        transform=plt.gca().transAxes
    )

    legend = plt.legend(fontsize=major)
    legend.get_frame().set_alpha(None)


def plot_reconstructions(
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
    minor = 15

    axes.scatter(x_data, y_data, label='Spectrum')
    axes.scatter(x_data, y_recon, label='Reconstruction')
    axes.locator_params(axis='y', nbins=5)
    axes.tick_params(axis='both', labelsize=minor)


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
    major = 20

    x_data = load_x_data(spectra.shape[-1])

    # Initialize reconstructions plots
    _, axes = plt.subplots(4, 4, figsize=(24, 12), sharex='col', gridspec_kw={'hspace': 0})
    axes = axes.flatten()

    # Plot reconstructions
    for i in range(axes.size):
        plot_reconstructions(x_data, spectra[i], outputs[i], axes[i])

    plt.figtext(0.5, 0.02, 'Energy (keV)', ha='center', va='center', fontsize=major)
    plt.figtext(
        0.02,
        0.5,
        'Scaled Log Counts',
        ha='center',
        va='center',
        rotation='vertical',
        fontsize=major,
    )

    legend = plt.figlegend(
        *axes[0].get_legend_handles_labels(),
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=major,
        markerscale=2,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig(f'{plots_dir}{prefix} Reconstructions.png', transparent=False)

    # Plot loss over epochs
    plot_loss(losses[0], losses[1])
    plt.savefig(f'{plots_dir}{prefix} Loss.png', transparent=False)


def plot_histogram(title: str, data: np.ndarray, data_twin: np.ndarray, axis: Axes) -> Axes:
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
    major = 20
    twin_axis = axis.twinx()

    axis.set_title(title, fontsize=major)
    axis.hist(data, bins=100, alpha=0.5, density=True)
    twin_axis.hist(data_twin, bins=100, alpha=0.5, density=True, color='orange')

    return twin_axis


def plot_param_distribution(log_params: list[int], params: np.ndarray, loader: DataLoader):
    """
    Plots histogram of each parameter for both true and predicted

    Parameters
    ----------
    log_params : list[integer]
        Indices of parameters in logarithmic space
    params : Tensor
        Parameter predictions from CNN
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    """
    major = 20
    param_transform = loader.dataset.dataset.transform[1]
    titles = ['nH', r'$\Gamma', 'FracSctr', r'$T_{max}$', 'Norm']

    # Transform parameters
    params_real = loader.dataset.dataset.params.cpu() * param_transform[1] + param_transform[0]
    params_real[:, log_params] = 10 ** params_real[:, log_params]
    params_predicted = params * param_transform[1] + param_transform[0]
    params_predicted[:, log_params] = 10 ** params_predicted[:, log_params]

    params_real = [params_real[:, i] for i in range(params_real.shape[-1])]
    params_predicted = [params_predicted[:, i] for i in range(params_predicted.shape[-1])]

    # Remove extreme parameters
    params_real[0] = np.delete(params_real[0], np.argwhere(params_real[0] > 30))
    params_real[4] = np.delete(params_real[4], np.argwhere(params_real[4] > 6000))
    params_predicted[0] = np.delete(params_predicted[0], np.argwhere(params_predicted[0] > 30))
    params_predicted[4] = np.delete(params_predicted[4], np.argwhere(params_predicted[4] > 6000))

    _, axes = plt.subplot_mosaic('AABBCC;DDDEEE', figsize=(32, 18))

    # Plot subplots
    for i, (title, axis) in enumerate(zip(titles, axes.values())):
        twin_axis = plot_histogram(title, params_real[:, i], params_predicted[:, i], axis)

    legend = plt.figlegend(
        axes['A'].get_legend_handles_labels()[0] + twin_axis.get_legend_handles_labels()[0],
        axes['A'].get_legend_handles_labels()[1] + twin_axis.get_legend_handles_labels()[1],
        fontsize=major,
        bbox_to_anchor=(0.95, 0.45),
    )
    legend.get_frame().set_alpha(None)
    plt.tight_layout()
    plt.savefig('../Test')
