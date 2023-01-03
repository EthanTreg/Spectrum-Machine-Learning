import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from matplotlib.axes import Axes

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
        axis.text(0.9, 0.9, i + 1, fontsize=16, transform=axis.transAxes)
        axis.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)

    legend = plt.figlegend(
        *axes[0].get_legend_handles_labels(),
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.95),
        fontsize=16,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)
    legend.legendHandles[1].set_color('orange')

    plt.figtext(0.5, 0.02, 'Energy (keV)', ha='center', va='center', fontsize=16)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(f'{plots_dir}Saliency_Plot.png', transparent=True)


def plot_reconstructions(y_data: np.ndarray, y_recon: np.ndarray, axes: Axes):
    """
    Plots reconstructions for a given epoch

    Parameters
    ----------
    y_data : ndarray
        Spectrum
    y_recon : ndarray
        Reconstructed Spectrum
    axes : Axes
        Plot axes
    """
    x_data = load_x_data(y_data.size)

    axes.scatter(x_data, y_data, label='Spectrum')
    axes.scatter(x_data, y_recon, label='Reconstruction')
    axes.locator_params(axis='y', nbins=5)


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
    plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.yscale('log')
    plt.text(
        0.8, 0.75,
        f'Final loss: {val_loss[-1]:.3e}',
        fontsize=16,
        transform=plt.gca().transAxes
    )

    legend = plt.legend(fontsize=20)
    legend.get_frame().set_alpha(None)


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
    # Initialize reconstructions plots
    _, axes = plt.subplots(4, 4, figsize=(24, 12), sharex='col', gridspec_kw={'hspace': 0})
    axes = axes.flatten()

    # Plot reconstructions
    for i in range(axes.size):
        plot_reconstructions(spectra[i], outputs[i], axes[i])

    plt.figtext(0.5, 0.02, 'Energy (keV)', ha='center', va='center', fontsize=16)
    plt.figtext(
        0.02,
        0.5,
        'Scaled Log Counts',
        ha='center',
        va='center',
        rotation='vertical',
        fontsize=16,
    )

    legend = plt.figlegend(
        *axes[0].get_legend_handles_labels(),
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.95),
        fontsize=16,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    plt.savefig(f'{plots_dir}{prefix} Reconstructions.png', transparent=True)

    # Plot loss over epochs
    plot_loss(losses[0], losses[1])
    plt.savefig(f'{plots_dir}{prefix} Loss.png', transparent=True)
