"""
Creates several plots
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fspnet.utils.utils import subplot_grid
from fspnet.utils.analysis import param_comparison
from fspnet.utils.data import load_data, load_x_data, data_normalization

MAJOR = 24
MINOR = 20
FIG_SIZE = (16, 9)


def _legend(labels: ndarray, columns: int = 2):
    """
    Plots a legend across the top of the plot

    Parameters
    ----------
    labels : ndarray
        Legend matplotlib handels and labels as an array to be unpacked into handels and labels
    columns : integer, default = 2
        Number of columns for the legend
    """
    legend = plt.figlegend(
        *labels,
        loc='lower center',
        ncol=columns,
        bbox_to_anchor=(0.5, 0.91),
        fontsize=MAJOR,
        columnspacing=10,
    )
    legend.get_frame().set_alpha(None)

    for handle in legend.legendHandles:
        if isinstance(handle, matplotlib.collections.PathCollection):
            handle.set_sizes([100])


def _initialize_plot(
        subplots: str | tuple[int, int] | list,
        legend: bool = False,
        x_label: str = None,
        y_label: str = None,
        plot_kwargs: dict = None,
        gridspec_kw: dict = None) -> ndarray:
    """
    Initializes subplots using either mosaic or subplots

    Parameters
    ----------
    subplots : string | tuple[integer, integer] | list
        Argument for subplot or mosaic layout, mosaic will use string or list
        and subplots will use tuple
    legend : boolean, default = False,
        If the figure will have a legend at the top, then space will be made
    x_label : string, default = None
        X label for the plot
    y_label : string, default = None
        Y label for the plot
    plot_kwargs : dict, default = None
        Optional arguments for the subplot or mosaic function, excluding gridspec_kw
    gridspec_kw : dict, default = None
        Gridspec arguments for the subplot or mosaic function

    Returns
    -------
    ndarray
        Subplot axes
    """
    gridspec = {
        'top': 0.95,
        'bottom': 0.05,
        'left': 0.06,
        'right': 0.99,
        'hspace': 0.05,
        'wspace': 0.75,
    }

    if not plot_kwargs:
        plot_kwargs = {}

    # Gridspec commands for optional layouts
    if legend:
        gridspec['top'] = 0.92

    if x_label:
        gridspec['bottom'] = 0.08

    if y_label:
        gridspec['left'] = 0.09

    if gridspec_kw:
        gridspec = gridspec | gridspec_kw

    # Plots either subplot or mosaic
    if isinstance(subplots, tuple):
        _, axes = plt.subplots(
            *subplots,
            figsize=FIG_SIZE,
            gridspec_kw=gridspec,
            **plot_kwargs,
        )
    else:
        _, axes = plt.subplot_mosaic(
            subplots,
            figsize=FIG_SIZE,
            gridspec_kw=gridspec,
            **plot_kwargs,
        )

    if x_label:
        plt.figtext(0.5, 0.02, x_label, ha='center', va='center', fontsize=MAJOR)

    if y_label:
        plt.figtext(
            0.02,
            0.5,
            y_label,
            ha='center',
            va='center',
            rotation='vertical',
            fontsize=MAJOR,
        )

    return axes


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
    plt.yticks(fontsize=MINOR, minor=True)
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
    axis_2.locator_params(axis='y', nbins=3)
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
    axes = _initialize_plot(
        (2, 2),
        legend=True,
        x_label='Energy (keV)',
        y_label='Scaled Log Counts',
        plot_kwargs={'sharex': 'col'},
        gridspec_kw={'wspace': 0.2}
    ).flatten()

    # Plot reconstructions
    for spectrum, output, axis in zip(spectra, outputs, axes):
        main_axis = _plot_reconstruction(x_data, spectrum, output, axis)

    labels = np.hstack((main_axis.get_legend_handles_labels(), axes[0].get_legend_handles_labels()))
    _legend(labels, columns=3)


def _plot_histogram(
        data: ndarray,
        axis: Axes,
        log: bool = False,
        labels: list[str] = None,
        hist_kwargs: dict = None,
        data_twin: ndarray = None) -> None | Axes:
    """
    Plots a histogram subplot with twin data if provided

    Parameters
    ----------
    data : ndarray
        Primary data to plot
    axis : Axes
        Axis to plot on
    log : boolean, default = False
        If data should be plotted on a log scale, expects linear data
    labels : list[string], default = None
        Labels for data and, if provided, data_twin
    hist_kwargs : dictionary, default = None
        Optional keyword arguments for plotting the histogram
    data_twin : ndarray, default = None
        Secondary data to plot

    Returns
    -------
    None | Axes
        If data_twin is provided, returns twin axis
    """
    bins = bin_num = 100

    if not labels:
        labels = ['', '']

    if not hist_kwargs:
        hist_kwargs = {}

    if log:
        bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), bin_num)
        axis.set_xscale('log')

    axis.hist(data, bins=bins, alpha=0.5, label=labels[0], **hist_kwargs)
    axis.tick_params(labelsize=MINOR)
    axis.ticklabel_format(axis='y', scilimits=(-2, 2))

    if data_twin is not None:
        twin_axis = axis.twinx()
        _plot_histogram(
            data_twin,
            twin_axis,
            log=log,
            labels=[labels[1]],
            hist_kwargs={'color': 'orange'},
        )
        return twin_axis

    return None


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
    cmap = plt.cm.hot
    x_data = load_x_data(spectra.shape[1])

    x_regions = x_data[::12]
    saliencies = np.mean(saliencies.reshape(saliencies.shape[0], -1, 12), axis=-1)
    saliencies = data_normalization(saliencies, mean=False, axis=1)[0] * 0.9 + 0.05

    # Initialize saliency plots
    axes = _initialize_plot(
        (2, 2),
        legend=True,
        x_label='Energy (keV)',
        gridspec_kw={'left': 0.01, 'bottom': 0.05, 'hspace': 0, 'wspace': 0},
        plot_kwargs={'sharex': 'col'},
    ).flatten()

    # Plot each saliency map
    for i, (
            axis,
            spectrum,
            prediction,
            saliency
    ) in enumerate(zip(axes, spectra, predictions, saliencies)):
        for j, (x_region, saliency_region) in enumerate(zip(x_regions[:-1], saliency[:-1])):
            axis.axvspan(x_region, x_regions[j + 1], color=cmap(saliency_region))

        axis.axvspan(x_regions[-1], x_data[-1], color=cmap(saliency[-1]))

        axis.scatter(x_data, spectrum, label='Target')
        axis.scatter(x_data, prediction, color='g', label='Prediction')
        axis.text(0.96, 0.9, i + 1, fontsize=MAJOR, transform=axis.transAxes)
        axis.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)

    _legend(axes[0].get_legend_handles_labels())
    plt.savefig(f'{plots_dir}Saliency_Plot.png', transparent=False)


def plot_param_pairs(data_paths: list[str], labels: list[str], config: dict):
    """
    Plots a pair plot to compare the distributions and comparisons between parameters

    Parameters
    ----------
    data_paths : list[string]
        Paths to the parameters to compare, can have length 1 or 2
    labels : list[string]
        Labels for the data of the same length as data_paths
    config : dictionary
        Configuration dictionary
    """
    log_params = config['model']['log-parameters']
    param_names = config['model']['parameter-names']
    plots_dir = config['output']['plots-directory']

    # Load data & shuffle
    if len(data_paths) == 2:
        params = param_comparison(data_paths)
    else:
        params = load_data(data_paths[0], load_kwargs={'usecols': range(1, 6)})

        if '.pickle' in data_paths[0]:
            params = np.array(params['params'])

        np.random.shuffle(params)
        params = [np.swapaxes(params, 0, 1)]

    # Initialize pair plots
    axes = _initialize_plot(
        (params[0].shape[0],) * 2,
        legend=True,
        x_label=' ',
        y_label=' ',
        plot_kwargs={'sharex': 'col'},
        gridspec_kw={'hspace': 0, 'wspace': 0},
    )

    # Loop through each subplot
    for i, (axes_col, *y_param) in enumerate(zip(axes, *params)):
        for j, (axis, *x_param) in enumerate(zip(axes_col, *params)):
            log = False

            # Share y-axis for all scatter plots
            if i != j and i == 0:
                axis.sharey(axes_col[1])
            elif i != j:
                axis.sharey(axes_col[0])

            # Hide ticks for plots that aren't in the first column or bottom row
            if j == 0 or (j == 1 and i == 0):
                axis.tick_params(axis='y', labelsize=MINOR)
            else:
                axis.tick_params(labelleft=False)

            if i == len(axes_col) - 1:
                axis.tick_params(axis='x', labelsize=MINOR)
            else:
                axis.tick_params(labelbottom=False)

            # Convert axis for logged parameters to log scale
            if j in log_params:
                log = True
                axis.set_xscale('log')
                axis.locator_params(axis='x', numticks=3)
            else:
                axis.locator_params(axis='x', nbins=3)

            if i in log_params and i != j:
                axis.set_yscale('log')
                axis.locator_params(axis='y', numticks=3)
            else:
                axis.locator_params(axis='y', nbins=3)

            # Plot scatter plots & histograms
            if i == j:
                twin_axis = _plot_histogram(x_param[0], axis, log=log, data_twin=x_param[1])
                axis.tick_params(labelleft=False, left=False)
                twin_axis.tick_params(labelright=False, right=False)
            elif j < i:
                axis.scatter(
                    x_param[0][:1000],
                    y_param[0][:1000],
                    s=4,
                    alpha=0.2,
                    label=labels[0],
                )
            else:
                axis.remove()

            if j < i and len(x_param) > 1:
                axis.scatter(
                    x_param[1][:1000],
                    y_param[1][:1000],
                    s=4,
                    alpha=0.2,
                    color='orange',
                    label=labels[1],
                )

            # Bottom row parameter names
            if i == len(axes_col) - 1:
                axis.set_xlabel(param_names[j], fontsize=MINOR)

            # Left column parameter names
            if j == 0:
                axis.set_ylabel(param_names[i], fontsize=MINOR)

    _legend(axes[-1, 0].get_legend_handles_labels())
    plt.savefig(f'{plots_dir}Parameter_Pairs.png')


def plot_param_comparison(
        plots_dir: str,
        param_names: list[str],
        config: dict):
    """
    Plots prediction against target for each parameter

    Parameters:
    ----------
    plots_dir : string
        Directory to save plots
    param_names : list[string]
        List of parameter names
    config: dictionary
        Configuration dictionary
    """
    log_params = config['model']['log-parameters']
    target, prediction = param_comparison((
        config['data']['encoder-data-path'],
        config['output']['parameter-predictions-path'],
    ))

    _, axes = plt.subplot_mosaic(
        subplot_grid(len(param_names)),
        constrained_layout=True,
        figsize=FIG_SIZE,
    )

    # Plot each parameter
    for i, (name, axis, target_param, predicted_param) in enumerate(zip(
        param_names,
        axes.values(),
        target,
        prediction)):
        value_range = [
            min(np.min(target_param), np.min(predicted_param)),
            max(np.max(target_param), np.max(predicted_param))
        ]
        axis.scatter(target_param, predicted_param, alpha=0.2)
        axis.plot(value_range, value_range, color='k')
        axis.set_title(name, fontsize=MAJOR)
        axis.tick_params(labelsize=MINOR)

        if i in log_params:
            axis.set_xscale('log')
            axis.set_yscale('log')

    plt.savefig(f'{plots_dir}Parameter_Comparison.png', transparent=False)


def plot_param_distribution(name: str, data_paths: list[str], labels: list[str], config: dict):
    """
    Plots histogram of each parameter for both true and predicted

    Parameters
    ----------
    name : string
        Name to save the plot
    data_paths : list[string]
        Paths to the parameters to plot
    labels : list[string]
        Legend labels for each data path
    config : dictionary
        Configuration dictionary
    """
    params = []
    log_params = config['model']['log-parameters']
    param_names = config['model']['parameter-names']
    plots_dir = config['output']['plots-directory']

    axes = _initialize_plot(
        subplot_grid(len(param_names)),
        legend=True,
        gridspec_kw={'bottom': 0.1, 'right': 0.95, 'hspace': 0.25},
    )

    for data_path in data_paths:
        data = load_data(data_path, load_kwargs={'usecols': range(1, 6)})

        if '.pickle' in data_path:
            data = data['params']

        params.append(np.swapaxes(data, 0, 1))

    # Plot subplots
    for i, (title, *param, axis) in enumerate(zip(param_names, *params, axes.values())):
        log = False

        if i in log_params:
            axis.set_xscale('log')
            log = True

        twin_axis = _plot_histogram(param[0], axis, log=log, labels=labels, data_twin=param[1])
        axis.set_xlabel(title, fontsize=MAJOR)

    _legend(np.hstack((
        axes[0].get_legend_handles_labels(),
        twin_axis.get_legend_handles_labels(),
    )))
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

    axes = _initialize_plot('AABBCC;DDDEEE', x_label='Energy (keV)', gridspec_kw={'hspace': 0.2})

    for title, weight, axis in zip(param_names, weights, axes.values()):
        axis.scatter(load_x_data(weight.size), weight)
        axis.set_title(title, fontsize=MAJOR)
        axis.tick_params(labelsize=MINOR)

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

    plt.figure(figsize=FIG_SIZE, constrained_layout=True)
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
