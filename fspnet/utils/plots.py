"""
Creates several plots
"""
from typing import Any, Callable

import torch.nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.figure import FigureBase
from matplotlib.colors import XKCD_COLORS
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fspnet.utils.data import load_x_data
from fspnet.utils.analysis import linear_weights
from fspnet.utils.utils import legend_marker, subplot_grid

MAJOR: int = 24
MINOR: int = 20
SCATTER_NUM: int = 1000
MARKERS: list[str] = list(Line2D.markers.keys())
COLOURS: list[str] = list(XKCD_COLORS.values())[::-1]
RECTANGLE: tuple[int, int] = (16, 9)
SQUARE: tuple[int, int] = (10, 10)
HI_RES: tuple[int, int] = (32, 18)
HI_RES_SQUARE: tuple[int, int] = (20, 20)


def _init_plot(
        titles: str | list[str] | None = None,
        x_labels: str | list[str] | None = None,
        y_labels: str | list[str] | None = None,
        fig_size: tuple[int, int] = RECTANGLE,
        subfigures: tuple[int, int] | None = None,
        **kwargs: Any) -> tuple[FigureBase, ndarray[FigureBase] | None]:
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
    fig_size : tuple[integer, integer]
        Size of the figure
    **kwargs
        Optional arguments for the subplot or mosaic function

    Returns
    -------
    tuple[FigureBase, ndarray[FigureBase] | None]
        FigureBase and subplot axes
    """
    title: str
    x_label: str
    y_label: str
    subfigs: ndarray[FigureBase] | None = None
    subfig: FigureBase
    fig: FigureBase = plt.figure(constrained_layout=True, figsize=fig_size)

    if subfigures:
        subfigs = fig.subfigures(*subfigures, **kwargs)

    if isinstance(titles, list) and subfigs is not None:
        for title, subfig in zip(titles, subfigs.flatten()):
            subfig.suptitle(title, fontsize=MAJOR)
    else:
        fig.suptitle(titles, fontsize=MAJOR)

    if isinstance(x_labels, list) and subfigs is not None:
        for x_label, subfig in zip(x_labels, subfigs.flatten()):
            subfig.supxlabel(x_label, fontsize=MAJOR)
    else:
        fig.supxlabel(x_labels, fontsize=MAJOR)

    if isinstance(y_labels, list) and subfigs is not None:
        for y_label, subfig in zip(y_labels, subfigs.flatten()):
            subfig.supylabel(y_label, fontsize=MAJOR)
    else:
        fig.supylabel(y_labels, fontsize=MAJOR)

    return fig, subfigs


def _init_subplots(
        subplots: str | tuple[int, int] | list | ndarray,
        fig: FigureBase | None = None,
        fig_size: tuple[int, int] = RECTANGLE,
        **kwargs: Any) -> tuple[dict[str, Axes] | ndarray[Axes], FigureBase]:
    """
    Generates subplots within a figure or sub-figure

    Parameters
    ----------
    subplots : str | tuple[int, int] | list | ndarray
        Parameters for subplots or subplot_mosaic
    fig : FigureBase | None, default = None
        FigureBase to add subplots to
    fig_size : tuple[integer, integer]
        Size of the figure, only used if fig is None

    **kwargs
        Optional kwargs to pass to subplots or subplot_mosaic

    Returns
    -------
    tuple[dict[str, Axes] | ndarray[Axes], FigureBase]
        Dictionary or array of subplot axes and figure
    """
    axes: dict[str, Axes] | ndarray[Axes]

    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=fig_size)

    # Plots either subplot or mosaic
    if isinstance(subplots, tuple):
        axes = fig.subplots(*subplots, **kwargs)
    else:
        axes = fig.subplot_mosaic(subplots, **kwargs)

    return axes, fig


def _legend(
        labels: list[str] | ndarray,
        fig: FigureBase,
        columns: int = 2,
        loc: str = 'outside upper center') -> Legend:
    """
    Plots a legend across the top of the plot

    Parameters
    ----------
    labels : (2,L) list | ndarray[np.str_]
        Legend matplotlib handles and labels of size L to be unpacked into handles and labels
    fig : FigureBase
        FigureBase to add legend to
    columns : integer, default = 2
        Number of columns for the legend
    loc : string, default = 'outside upper center'
        Location to place the legend

    Returns
    -------
    Legend
        Legend object
    """
    rows: int
    fig_size: float = fig.get_size_inches()[0] * fig.dpi
    handle: mpl.artist.Artist
    legend_offset: float
    legend: Legend = fig.legend(
        *labels,
        loc=loc,
        ncol=columns,
        fontsize=MAJOR,
        borderaxespad=0.2,
    )

    legend_offset = float(np.array(legend.get_window_extent())[0, 0])

    if legend_offset < 0:
        legend.remove()
        rows = np.abs(legend_offset) // fig_size + 2
        columns = np.ceil(len(labels[1]) / rows)
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

    major_axis: Axes = make_axes_locatable(axis).append_axes('top', size='150%', pad=0)
    major_axis.scatter(x_data, y_data, label='Spectrum')
    major_axis.scatter(x_data, y_recon, label='Reconstruction')
    major_axis.locator_params(axis='y', nbins=3)
    major_axis.tick_params(axis='y', labelsize=MINOR)
    major_axis.set_xticks([])
    return major_axis


def _plot_histogram(
        data: ndarray,
        axis: Axes,
        log: bool = False,
        bin_num: int = 100,
        alpha: float | None = None,
        colour: str = 'blue',
        labels: list[str] | None = None,
        data_range: tuple[float, float] | None = None,
        data_twin: ndarray | None = None,
        **kwargs) -> Axes | None:
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
    bin_num : integer, default = 100
        Number of bins
    alpha : float, default = 0.2 if density, 0.5 if data_twin is provided; otherwise, 1
        Transparency of the histogram, gets halved if data_twin is provided
    colour : string
        Colour of the histogram or density plot
    labels : list[string], default = None
        Labels for data and, if provided, data_twin
    data_range : tuple[float, float], default = None
        x-axis data range, required if density is True
    data_twin : ndarray, default = None
        Secondary data to plot

    **kwargs
        Optional keyword arguments that get passed to Axes.plot and Axes.fill_between if density is
        True, else to Axes.hist

    Returns
    -------
    None | Axes
        If data_twin is provided, returns twin axis
    """
    bins: int | ndarray = bin_num
    twin_axis: Axes

    if not labels:
        labels = ['', '']

    if alpha is None and data_twin is None:
        alpha = 1
    else:
        alpha = 0.5

    if log:
        bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), bin_num)
        axis.set_xscale('log')

    axis.hist(
        data,
        bins=bins,
        alpha=alpha,
        label=labels[0],
        color=colour,
        range=data_range,
        **kwargs,
    )

    axis.tick_params(labelsize=MINOR)
    axis.ticklabel_format(axis='y', scilimits=(-2, 2))

    if data_twin is not None:
        twin_axis = axis.twinx()
        _plot_histogram(
            data_twin,
            twin_axis,
            log=log,
            bin_num=bin_num,
            alpha=alpha,
            colour='orange',
            labels=[labels[1]],
            **kwargs,
        )
        return twin_axis
    return None


def plot_encoder_pgstats(loss_file: str, config: dict) -> None:
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

    plt.figure(figsize=RECTANGLE, constrained_layout=True)
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


def plot_linear_weights(
        param_names: list[str],
        net: torch.nn.Module,
        plots_dir: str | None = None) -> None:
    """
    Plots the mappings of the weights from the lowest dimension
    to a high dimension for the linear layers

    Parameters
    ----------
    param_names : list[str], default = None
        Names of the parameters
    net : ModuleList
        Neural network to calculate the linear weights mapping
    plots_dir : str, default = None
        Directory to save plots
    """
    title: str
    axes: dict[str, Axes]
    weight: ndarray
    weights: ndarray = linear_weights(net)
    axis: Axes

    fig, _ = _init_plot(x_labels='Energy (keV)')
    axes, _ = _init_subplots(subplot_grid(len(param_names)), fig=fig)

    for title, weight, axis in zip(param_names, weights, axes.values()):
        axis.scatter(load_x_data(len(weight)), weight)
        axis.set_title(title, fontsize=MAJOR)
        axis.tick_params(labelsize=MINOR)
        axis.set_yticks([])

    if plots_dir:
        plt.savefig(f'{plots_dir}Linear_Weights_Mappings.png')


def plot_multi_plot(
        labels: list[str],
        data: list[ndarray],
        plot_func: Callable[..., dict[str, Axes] | ndarray],
        plots_dir: str | None = None,
        **kwargs: Any) -> None:
    """
    Plots multiple datasets onto the same plot

    Parameters
    ----------
    labels : list[str]
        Labels for the different datasets
    data : list[(N,L) ndarray]
        Datasets to plot
    plot_func : Callable[..., dict[str, Axes] | ndarray]
        Plotting function to plot multiple datasets onto
    plots_dir : string, default = None
        Directory to save plots
    """
    datum: ndarray
    axes: ndarray | None = None

    for colour, datum in zip(COLOURS, data):
        axes = plot_func(datum, axes=axes, alpha=len(data) ** -1, colour=colour, **kwargs)

    _legend(legend_marker(COLOURS, labels), plt.gcf(), columns=len(labels))

    if plots_dir:
        plt.savefig(f'{plots_dir}Multi_Dataset_Plot.png')


def plot_param_comparison(
        log_params: list[int],
        param_names: list[str],
        targets: ndarray,
        preds: ndarray,
        plots_dir: str | None = None) -> None:
    """
    Plots prediction against target for each parameter

    Parameters:
    ----------
    log_params : list[int], default = None
        Which parameters should be in log space
    param_names : list[str], default = None
        Names of the parameters
    targets : (N,L) ndarray
        Target parameters
    preds : (N,L) ndarray
        Predicted parameters
    plots_dir : str, default = None
        Directory to save plots
    """
    i: int
    name: str
    value_range: tuple[float, float]
    axes: dict[str, Axes]
    pred_param: ndarray
    target_param: ndarray
    colours: ndarray = np.array(['blue'] * SCATTER_NUM, dtype=object)
    axis: Axes
    fig: FigureBase

    axes, fig = _init_subplots(subplot_grid(len(param_names)))
    targets = targets[:SCATTER_NUM].swapaxes(0, 1)
    preds = preds[:SCATTER_NUM].swapaxes(0, 1)

    # Highlight parameters that are maximally or minimally constrained
    for target_param in targets:
        colours[np.argwhere(
            (target_param == np.max(target_param)) |
            (target_param == np.min(target_param))
        )] = 'red'

    # Plot each parameter
    for i, (name, axis, target_param, pred_param) in enumerate(zip(
            param_names,
            axes.values(),
            targets,
            preds)):
        value_range = (np.min(target_param), np.max(target_param))

        axis.scatter(target_param, pred_param, color=colours, alpha=0.2)
        axis.plot(value_range, value_range, color='k')
        axis.set_title(name, fontsize=MAJOR)
        axis.tick_params(labelsize=MINOR)
        axis.xaxis.get_offset_text().set_visible(False)
        axis.yaxis.get_offset_text().set_size(MINOR)

        if i in log_params:
            axis.set_xscale('log')
            axis.set_yscale('log')

    _legend(legend_marker(['blue', 'red'], ['Free', 'Pegged']), fig)

    if plots_dir:
        plt.savefig(f'{plots_dir}Parameter_Comparison.png')


def plot_param_distribution(
        data: ndarray,
        log_params: list[int],
        param_names: list[str],
        y_axis: bool = True,
        plots_dir: str | None = None,
        axes: dict[str, Axes] | None = None,
        **kwargs: Any) -> dict[str, Axes]:
    """
    Plots histogram of each parameter for both true and predicted

    Parameters
    ----------
    data : (N,L) ndarray
        Data to plot the distributions for
    log_params : list[int], default = None
        Which parameters should be in log space
    param_names : list[str], default = None
        Names of the parameters
    y_axis : boolean, default = True
        If the y-axis ticks should be plotted
    plots_dir : str, default = None
        Directory to save plots
    axes : dict[str, Axes], default = None
        Axes to use for plotting L parameters

    **kwargs
        Optional keyword arguments to pass to _plot_histogram

    Returns
    -------
    dict[str, Axes]
        Plot axes
    """
    log: bool
    i: int
    title: str
    datum: ndarray
    axis: Axes

    data = data[:SCATTER_NUM].swapaxes(0, 1)

    if axes is None:
        axes, _ = _init_subplots(subplot_grid(len(param_names)))

    # Plot subplots
    for i, (title, datum, axis) in enumerate(zip(param_names, data, axes.values())):
        log = False

        if i in log_params:
            log = True
            axis.set_xscale('log')

        _plot_histogram(datum, axis, log=log, **kwargs)
        axis.set_xlabel(title, fontsize=MAJOR)

        if not y_axis:
            axis.tick_params(labelleft=False, left=False)

    if plots_dir:
        plt.savefig(f'{plots_dir}Parameter_Distribution.png')

    return axes


def plot_param_pairs(
        data: ndarray,
        plots_dir: str | None = None,
        log_params: list[int] | None = None,
        param_names: list[str | None] | None = None,
        ranges: ndarray | None = None,
        axes: ndarray | None = None,
        **kwargs) -> ndarray:
    """
    Plots a pair plot to compare the distributions and comparisons between parameters

    Parameters
    ----------
    data : (N,L) ndarray
        Data to plot parameter pairs for N data points and L parameters
    plots_dir : string, default = None
        Directory to save plots
    log_params : list[int], default = None
        Which parameters should be in log space
    param_names : list[str], default = None
        Names of the parameters
    ranges : (L,2) ndarray, default = None
        Ranges for L parameters, required if using kwargs to plot densities
    axes : (L,L) ndarray, default = None
        Axes to use for plotting L parameters

    **kwargs
        Optional keyword arguments passed to _plot_histogram

    Returns
    -------
    ndarray
        Plot axes
    """
    log: bool
    i: int
    j: int
    x_data: ndarray
    y_data: ndarray
    x_range: ndarray
    y_range: ndarray
    axes_row: ndarray
    axis: Axes

    data = data[:SCATTER_NUM].swapaxes(0, 1)

    if log_params is None:
        log_params = []

    if param_names is None:
        param_names = [None] * data.shape[0]

    if ranges is None:
        ranges = [None] * data.shape[0]

    if axes is None:
        axes, _ = _init_subplots((data.shape[0],) * 2, sharex='col', fig_size=HI_RES)

    # Loop through each subplot
    for i, (axes_row, y_data, y_range) in enumerate(zip(axes, data, ranges)):
        for j, (axis, x_data, x_range) in enumerate(zip(axes_row, data, ranges)):
            log = False

            # Share y-axis for all scatter plots
            if i not in {0, j}:
                axis.sharey(axes_row[0])

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

            # Hide ticks for plots that aren't in the first column or bottom row
            if j == 0:
                axis.tick_params(axis='y', labelsize=MINOR)
            else:
                axis.tick_params(labelleft=False, left=False)

            if i == axes.shape[0] - 1:
                axis.tick_params(axis='x', labelsize=MINOR)
            else:
                axis.tick_params(labelbottom=False, bottom=False)

            if x_range is not None and j < i:
                axis.set_xlim(x_range)
                axis.set_ylim(y_range)

            # Plot scatter plots & histograms
            if i == j:
                _plot_histogram(x_data, axis, log=log, data_range=x_range, **kwargs)
                axis.tick_params(labelleft=False, left=False)
            elif j < i:
                axis.scatter(
                    x_data[:SCATTER_NUM],
                    y_data[:SCATTER_NUM],
                    s=4,
                    alpha=0.2,
                )
            else:
                axis.set_visible(False)

            # Bottom row parameter names
            if param_names[0] and i == len(axes_row) - 1:
                axis.set_xlabel(param_names[j], fontsize=MAJOR)

            # Left column parameter names
            if param_names[0] and j == 0 and i != 0:
                axis.set_ylabel(param_names[i], fontsize=MAJOR)

    if plots_dir:
        plt.savefig(f'{plots_dir}Parameter_Pairs.png')

    return axes


def plot_performance(
        y_label: str,
        val: list[Any] | ndarray,
        log_y: bool = True,
        plots_dir: str | None = None,
        train: list[Any] | ndarray | None = None) -> None:
    """
    Plots training and validation performance as a function of epochs

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    y_label : string
        Performance metric
    val : list[Any] | ndarray
        Validation performance
    log_y : boolean, default = True
        If y-axis should be logged
    plots_dir : str, default = None
        Directory to save plots
    train : list[Any] | ndarray, default = None
        Training performance
    """
    plt.figure(figsize=RECTANGLE, constrained_layout=True)

    if train is not None:
        plt.plot(train, label='Training')

    plt.plot(val, label='Validation')
    plt.xticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR, minor=True)
    plt.xlabel('Epoch', fontsize=MINOR)
    plt.ylabel(y_label, fontsize=MINOR)
    plt.text(
        0.8, 0.75,
        f'Final: {val[-1]:.3e}',
        fontsize=MINOR,
        transform=plt.gca().transAxes
    )

    if log_y and np.min(val) > 0:
        plt.yscale('log')

    plt.legend(fontsize=MAJOR)

    if plots_dir:
        plt.savefig(f'{plots_dir}Performance.png')


def plot_pgstat_iterations(loss_files: list[str], names: list[str], config: dict) -> None:
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


def plot_reconstructions(
        spectra: ndarray,
        outputs: ndarray,
        plots_dir: str | None = None) -> tuple[list[Axes], ndarray, FigureBase, Legend]:
    """
    Plots reconstructions and residuals for 4 spectra

    Parameters
    ----------
    spectra : ndarray
        True spectra
    outputs : ndarray
        Predicted spectra
    plots_dir : str, default = None
        Directory to save plots

    Returns
    -------
    tuple[list[Axes], ndarray, FigureBase, Legend]
        Major axes, minor axes, figure, and legend
    """
    major_axes: list[Axes] = []
    axes: ndarray
    labels: ndarray
    output: ndarray
    spectrum: ndarray
    x_data: ndarray = load_x_data(spectra.shape[-1])
    axis: Axes
    fig: FigureBase
    legend: Legend

    # Initialize reconstructions plots
    fig, _ = _init_plot(x_label='Energy (keV)', y_label='Scaled Log Counts')
    axes, _ = _init_subplots((2, 2), fig=fig, sharex='col')
    axes = axes.flatten()

    # Plot reconstructions
    for spectrum, output, axis in zip(spectra, outputs, axes):
        major_axes.append(_plot_reconstruction(x_data, spectrum, output, axis))

    labels = np.hstack((
        major_axes[0].get_legend_handles_labels(),
        axes[-1].get_legend_handles_labels()),
    )
    legend = _legend(labels, fig, columns=3)

    if plots_dir:
        plt.savefig(f'{plots_dir}Reconstructions.png')

    return major_axes, axes, fig, legend


def plot_saliency(
        spectra: ndarray,
        predictions: ndarray,
        saliencies: ndarray,
        plots_dir: str | None = None) -> None:
    """
    Plots saliency map for the autoencoder

    Parameters
    ----------
    spectra : ndarray
        Target spectra
    predictions : ndarray
        Predicted spectra
    saliencies : ndarray
        Saliency
    plots_dir : str, default = None
        Directory to save plots
    """
    bins: int = 8
    labels: ndarray
    saliency: ndarray
    x_regions: ndarray
    major_axes: ndarray
    minor_axes: ndarray
    x_data: ndarray = load_x_data(spectra.shape[1])
    fig: FigureBase
    legend: Legend
    twin_axis: Axes
    axis: Axes

    x_regions = x_data[::bins]
    saliencies = np.mean(saliencies.reshape(saliencies.shape[0], -1, bins), axis=-1)

    # Initialize saliency plots
    major_axes, minor_axes, fig, legend = plot_reconstructions(spectra, predictions)
    legend.remove()

    for axis, saliency in zip(major_axes, saliencies):
        twin_axis = axis.twinx()
        twin_axis.step(x_regions, saliency, color='green', label='Saliency')
        twin_axis.tick_params(right=False, labelright=False)

    labels = np.hstack((
        major_axes[-1].get_legend_handles_labels(),
        twin_axis.get_legend_handles_labels(),
        minor_axes[0].get_legend_handles_labels(),
    ))
    _legend(labels, fig, columns=4)

    if plots_dir:
        plt.savefig(f'{plots_dir}Saliency_Plot.png')
