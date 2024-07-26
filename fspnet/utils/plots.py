"""
Creates several plots
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fspnet.utils.network import Network
from fspnet.utils.utils import subplot_grid
from fspnet.utils.data import load_params, load_x_data
from fspnet.utils.analysis import param_comparison, linear_weights, encoder_pgstats

MAJOR = 28
MINOR = 24
FIG_SIZE = (16, 9)
SCATTER_NUM = 1000


def _legend(
        labels: list | ndarray,
        fig: Figure,
        columns: int = 2,
        loc: str = 'outside upper center') -> mpl.legend.Legend:
    """
    Plots a legend across the top of the plot

    Parameters
    ----------
    labels : list | ndarray
        Legend matplotlib handles and labels as an array to be unpacked into handles and labels
    fig : Figure
        Figure to add legend to
    columns : integer, default = 2
        Number of columns for the legend
    loc : string, default = 'outside upper center'
        Location to place the legend

    Returns
    -------
    Legend
        Legend object
    """
    legend = fig.legend(
        *labels,
        loc=loc,
        ncol=columns,
        fontsize=MAJOR,
        borderaxespad=0.2,
    )

    for handle in legend.legend_handles:
        handle.set_alpha(1)

        if isinstance(handle, mpl.collections.PathCollection):
            handle.set_sizes([100])

    return legend


def _init_plot(
        subplots: str | tuple[int, int] | list | ndarray,
        x_label: str = None,
        y_label: str = None,
        **kwargs) -> tuple[Figure, dict | ndarray]:
    """
    Initialises subplots using either mosaic or subplots

    Parameters
    ----------
    subplots : string | tuple[integer, integer] | list | ndarray
        Argument for subplot or mosaic layout, mosaic will use string or list
        and subplots will use tuple
    legend : boolean, default = False,
        If the figure will have a legend at the top, then space will be made
    x_label : string, default = None
        X label for the plot
    y_label : string, default = None
        Y label for the plot
    **kwargs
        Optional arguments for the subplot or mosaic function

    Returns
    -------
    tuple[Figure, dictionary | ndarray]
        Figure and subplot axes
    """
    # Plots either subplot or mosaic
    if isinstance(subplots, tuple):
        fig, axes = plt.subplots(
            *subplots,
            figsize=FIG_SIZE,
            constrained_layout=True,
            **kwargs,
        )
    else:
        fig, axes = plt.subplot_mosaic(
            subplots,
            figsize=FIG_SIZE,
            layout='constrained',
            **kwargs,
        )

    if x_label:
        fig.supxlabel(x_label, fontsize=MAJOR)

    if y_label:
        fig.supylabel(y_label, fontsize=MAJOR)

    return fig, axes


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
        0.7, 0.75,
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
    major_axis = divider.append_axes('top', size='150%', pad=0)

    major_axis.scatter(x_data, y_data, label='Spectrum')
    major_axis.scatter(x_data, y_recon, label='Reconstruction')
    major_axis.locator_params(axis='y', nbins=3)
    major_axis.tick_params(axis='y', labelsize=MINOR)
    major_axis.set_xticks([])

    return major_axis


def _plot_reconstructions(
        spectra: ndarray,
        outputs: ndarray) -> tuple[list[plt.Axes], ndarray, Figure, mpl.legend.Legend]:
    """
    Plots reconstructions and residuals for 4 spectra

    Parameters
    ----------
    spectra : ndarray
        True spectra
    outputs : ndarray
        Predicted spectra

    Returns
    -------
    tuple[list[Axes], ndarray, Figure, Legend]
        Major axes, minor axes, figure, and legend
    """
    major_axes = []
    x_data = load_x_data(spectra.shape[-1])

    # Initialize reconstructions plots
    fig, axes = _init_plot(
        (2, 2),
        x_label='Energy (keV)',
        y_label='Scaled Log Counts',
        sharex='col',
    )
    axes = axes.flatten()

    # Plot reconstructions
    for spectrum, output, axis in zip(spectra, outputs, axes):
        major_axes.append(_plot_reconstruction(x_data, spectrum, output, axis))

    labels = np.hstack((
        major_axes[0].get_legend_handles_labels(),
        axes[-1].get_legend_handles_labels()),
    )
    legend = _legend(labels, fig, columns=3)

    return major_axes, axes, fig, legend


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
    bins = 8
    x_data = load_x_data(spectra.shape[1])

    x_regions = x_data[::bins]
    saliencies = np.mean(saliencies.reshape(saliencies.shape[0], -1, bins), axis=-1)

    # Initialize saliency plots
    major_axes, minor_axes, fig, legend = _plot_reconstructions(spectra, predictions)
    legend.remove()

    for axis, saliency in zip(major_axes, saliencies):
        twin_axis = axis.twinx()
        twin_axis.step(x_regions, saliency, color='green', label='Saliency')
        twin_axis.tick_params(right=False, labelright=False)

    labels = np.hstack((
        axes[-1].get_legend_handles_labels(),
        twin_axis.get_legend_handles_labels(),
        minor_axes[0].get_legend_handles_labels()),
    )
    _legend(labels, fig, columns=4)
    plt.savefig(f'{plots_dir}Saliency_Plot.png')


def plot_param_pairs(
        data_paths: tuple[str] | tuple[str, str],
        config: dict,
        labels: tuple[str, str] = None):
    """
    Plots a pair plot to compare the distributions and comparisons between parameters

    Parameters
    ----------
    data_paths : tuple[string] | tuple[string, string]
        Paths to the parameters to compare, can have length 1 or 2
    config : dictionary
        Configuration dictionary
    labels : tuple[string, string], default = None
        Labels for the legend if provided
    """
    log_params = config['model']['log-parameters']
    param_names = config['model']['parameter-names']
    plots_dir = config['output']['plots-directory']

    if not labels:
        labels = [None]

    # Load data & shuffle
    if len(data_paths) == 2:
        params = param_comparison(data_paths)
    else:
        params = load_params(data_paths[0], load_kwargs={'usecols': range(1, 6)})[1]
        np.random.shuffle(params)
        params = [np.swapaxes(params, 0, 1)]

    # Initialize pair plots
    fig, axes = _init_plot((params[0].shape[0],) * 2, sharex='col')

    # Loop through each subplot
    for i, (axes_col, *y_param) in enumerate(zip(axes, *params)):
        for j, (axis, *x_param) in enumerate(zip(axes_col, *params)):
            log = False

            if len(x_param) > 1:
                data_twin = x_param[1]
            else:
                data_twin = None

            # Share y-axis for all scatter plots
            if i != j and i == 0:
                axis.sharey(axes_col[1])
            elif i != j:
                axis.sharey(axes_col[0])

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
                twin_axis = _plot_histogram(x_param[0], axis, log=log, data_twin=data_twin)
                axis.tick_params(which='both', labelleft=False, left=False)
            elif j < i:
                axis.scatter(
                    x_param[0][:SCATTER_NUM],
                    y_param[0][:SCATTER_NUM],
                    s=4,
                    alpha=0.2,
                    label=labels[0],
                )
            else:
                axis.remove()

            if twin_axis:
                twin_axis.tick_params(which='both', labelright=False, right=False)

            if j < i and len(x_param) > 1:
                axis.scatter(
                    x_param[1][:SCATTER_NUM],
                    y_param[1][:SCATTER_NUM],
                    s=4,
                    alpha=0.2,
                    color='orange',
                    label=labels[1],
                )

            # Hide ticks for plots that aren't in the first column or bottom row
            if j == 0:
                axis.tick_params(axis='y', labelsize=MINOR)
            else:
                axis.tick_params(which='both', left=False, labelleft=False)

            if i == len(axes_col) - 1:
                axis.tick_params(axis='x', labelsize=MINOR)
            else:
                axis.tick_params(which='both', bottom=False)

            # Bottom row parameter names
            if i == len(axes_col) - 1:
                axis.set_xlabel(param_names[j], fontsize=MAJOR)

            # Left column parameter names
            if j == 0 and i != 0:
                axis.set_ylabel(param_names[i], fontsize=MAJOR)

    if labels[0]:
        _legend(axes[-1, 0].get_legend_handles_labels(), fig, loc='upper center')

    plt.savefig(f'{plots_dir}Parameter_Pairs.png')


def plot_param_comparison(config: dict):
    """
    Plots prediction against target for each parameter

    Parameters:
    ----------
    config: dictionary
        Configuration dictionary
    """
    log_params = config['model']['log-parameters']
    param_names = config['model']['parameter-names']
    plots_dir = config['output']['plots-directory']

    target, prediction = param_comparison((
        config['data']['encoder-data-path'],
        config['output']['parameter-predictions-path'],
    ))
    colours = np.array(['blue'] * SCATTER_NUM, dtype=object)

    fig, axes = _init_plot(subplot_grid(len(param_names)))

    # Highlight parameters that are maximally or minimally constrained
    for target_param in target[:, :SCATTER_NUM]:
        colours[np.argwhere(
            (target_param == np.max(target_param)) |
            (target_param == np.min(target_param))
        )] = 'red'

    # Plot each parameter
    for i, (name, axis, target_param, predicted_param) in enumerate(zip(
        param_names,
        axes.values(),
        target[:, :SCATTER_NUM],
        prediction[:, :SCATTER_NUM])):
        value_range = [np.min(target_param), np.max(target_param)]

        axis.scatter(target_param, predicted_param, color=colours, alpha=0.2)
        axis.plot(value_range, value_range, color='k')
        axis.set_title(name, fontsize=MAJOR)
        axis.tick_params(labelsize=MINOR)
        axis.xaxis.get_offset_text().set_visible(False)
        axis.yaxis.get_offset_text().set_size(MINOR)

        if i in log_params:
            axis.set_xscale('log')
            axis.set_yscale('log')

    _legend([[
        plt.scatter([], [], color='blue'),
        plt.scatter([], [], color='red'),
    ], ['Free', 'Pegged']], fig)
    plt.savefig(f'{plots_dir}Parameter_Comparison.png')


def plot_param_distribution(
        name: str,
        data_paths: list[str],
        config: dict,
        y_axis: bool = True,
        labels: list[str] = None):
    """
    Plots histogram of each parameter for both true and predicted

    Parameters
    ----------
    name : string
        Name to save the plot
    data_paths : list[string]
        Paths to the parameters to plot
    config : dictionary
        Configuration dictionary
    y_axis : boolean, default = True
        If the y-axis ticks should be plotted
    labels : list[string], default = None
        Legend labels for each data path
    """
    params = []
    log_params = config['model']['log-parameters']
    param_names = config['model']['parameter-names']
    plots_dir = config['output']['plots-directory']

    fig, axes = _init_plot(subplot_grid(len(param_names)))

    for data_path in data_paths:
        params.append(np.swapaxes(load_params(data_path)[1], 0, 1))

    # Plot subplots
    for i, (title, *param, axis) in enumerate(zip(param_names, *params, axes.values())):
        log = False

        if len(param) > 1:
            data_twin = param[1]
        else:
            data_twin = None

        if i in log_params:
            axis.set_xscale('log')
            log = True

        twin_axis = _plot_histogram(param[0], axis, log=log, labels=labels, data_twin=data_twin)
        axis.set_xlabel(title, fontsize=MAJOR)

        if not y_axis:
            axis.tick_params(labelleft=False, left=False)
            twin_axis.tick_params(labelright=False, right=False)

    if twin_axis:
        labels = np.hstack((
            axes[0].get_legend_handles_labels(),
            twin_axis.get_legend_handles_labels(),
        ))
    elif labels:
        labels = axes[0].get_legend_handles_labels()

    if labels is not None:
        _legend(labels, fig)

    plt.savefig(plots_dir + name)


def plot_linear_weights(config: dict, network: Network):
    """
    Plots the mappings of the weights from the lowest dimension
    to a high dimension for the linear layers

    Parameters
    ----------
    config : dictionary
        Configuration dictionary
    network : Network
        Neural network to calculate the linear weights mapping
    """
    plots_dir = config['output']['plots-directory']
    param_names = config['model']['parameter-names']
    weights = linear_weights(network)

    _, axes = _init_plot(subplot_grid(len(param_names)), x_label='Energy (keV)')

    for title, weight, axis in zip(param_names, weights, axes.values()):
        axis.scatter(load_x_data(weight.size), weight)
        axis.set_title(title, fontsize=MAJOR)
        axis.tick_params(labelsize=MINOR)
        axis.set_yticks([])

    plt.savefig(f'{plots_dir}Linear_Weights_Mappings.png')


def plot_encoder_pgstats(loss_file: str, config: dict):
    """
    Plots the encoder's PGStat against the corresponding spectrum maximum

    Parameters
    ----------
    loss_file : string
        Path to the file that contains the PGStats for each spectrum
    config : dictionary
        Configuration dictionary
    """
    plots_dir = config['output']['plots-directory']
    spectra_file = config['data']['encoder-data-path']

    losses, spectra_max = encoder_pgstats(loss_file, spectra_file)
    losses = losses[:SCATTER_NUM]
    spectra_max = spectra_max[:SCATTER_NUM]

    plt.figure(figsize=FIG_SIZE, constrained_layout=True)
    plt.scatter(spectra_max, losses)
    plt.xlabel(
        r'Spectrum Maximum $(\mathrm{Counts}\ s^{-1}\ \mathrm{detector}^{-1}\ keV^{-1})$',
        fontsize=MAJOR,
    )
    plt.ylabel('PGStat', fontsize=MAJOR)
    plt.yscale('log')
    plt.xticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR)
    plt.savefig(f'{plots_dir}Encoder_PGStats.png')


def plot_pgstat_iterations(loss_files: list[str], names: list[str], config: dict):
    """
    Plots the median PGStat for every step iterations of Xspec fitting from the provided loss files

    Parameters
    ----------
    loss_files : list[string]
        Files with losses for each step
    names : list[string]
        Plot name for each loss file
    config : dictionary
        Configuration dictionary
    """
    step = config['model']['step']
    num_params = config['model']['parameters-number']
    plots_dir = config['output']['plots-directory']
    losses = []

    for loss_file in loss_files:
        data = np.loadtxt(loss_file, delimiter=',', dtype=str)
        losses.append(np.median(data[:, num_params + 1:].astype(float), axis=0))

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)

    for name, loss in zip(names, losses):
        plt.plot(np.arange(loss.size) * step + step, loss, label=name)

    plt.xlabel('Xspec Fitting Iterations', fontsize=MAJOR)
    plt.ylabel('Median Reduced PGStat', fontsize=MAJOR)
    plt.xticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR)
    plt.yscale('log')
    _legend(plt.gca().get_legend_handles_labels(), fig)
    plt.savefig(f'{plots_dir}Iteration_Median_PGStat.png')


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
        plt.savefig(f'{plots_dir}{prefix}_Reconstructions.png')

    # Plot loss over epochs
    _plot_loss(losses[0], losses[1])
    plt.savefig(f'{plots_dir}{prefix}_Loss.png')


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
