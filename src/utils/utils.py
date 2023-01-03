from __future__ import annotations

import os
from typing import TYPE_CHECKING
from multiprocessing import Queue, Value

import xspec
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils.plot_utils import plot_saliency

if TYPE_CHECKING:
    from src.utils.networks import Network


class PyXspecFitting:
    """
    Handles fitting parameters to spectrum using a model and calculates fit statistic

    Attributes
    ----------
    optimize : bool
        Whether to optimize the fit
    model : string
        Model to use for PyXspec
    dirs : list[string]
        Root directory & directory for spectral data
    fix_params : ndarray
        Parameter number & value of fixed parameters
    fix_params_index: ndarray
        Index to insert fixed parameters into network parameter output
    param_limits : tensor
        Parameter lower & upper limits

    Methods
    -------
    fit_statistic(params)
        Calculates fit statistic from spectrum and parameter predictions
    fit_loss(total, names, params, counter, queue)
        Evaluates the loss using PyXspec and each predicted parameter
    """
    def __init__(self, model: str, dirs: list[str], fix_params: np.ndarray):
        """
        Parameters
        ----------
        model : string
            Model to use for PyXspec
        dirs : list[string]
            Root directory & directory for spectral data
        fix_params : ndarray
            Parameter number (starting from 1) & value of fixed parameters
        """
        super().__init__()
        self.optimize = False
        self.dirs = dirs
        self.fix_params = fix_params
        self.param_limits = np.empty((0, 2))
        self.fix_params_index = self.fix_params[:, 0] - np.arange(self.fix_params.shape[1]) - 1

        # PyXspec initialization
        xspec.Xset.chatter = 0
        xspec.Xset.logChatter = 0
        xspec.AllModels.lmod('simplcutx', dirPath='../../../Documents/Xspec_Models/simplcutx')
        self.model = xspec.Model(model)
        xspec.AllModels.setEnergies('0.002 500 1000 log')
        xspec.Fit.statMethod = 'pgstat'
        xspec.Fit.query = 'no'

        # Generate parameter limits
        for j in range(self.model.nParameters):
            j += 1

            if j in fix_params[:, 0]:
                self.model(j).frozen = True
            else:
                limits = np.array(self.model(j).values)[[2, 5]]
                self.param_limits = np.vstack((self.param_limits, limits))

    def fit_statistic(self, params: np.ndarray) -> float:
        """
        Forward function of PyXspec loss
        Parameters
        ----------
        params : ndarray
            Parameter predictions from CNN
        Returns
        -------
        float
            Loss value
        """
        # Merge fixed & free parameters
        params = np.insert(params, self.fix_params_index, self.fix_params[:, 1])

        # Update model parameters
        self.model.setPars(params.tolist())

        # If optimization is enabled, then try unless there is an error
        if self.optimize:
            try:
                xspec.Fit.perform()
            except Exception:
                self.model.setPars(params.tolist())

        # Calculate fit statistic loss
        return xspec.Fit.statistic

    def fit_loss(
            self,
            total: int,
            names: list[str],
            params: np.ndarray,
            counter: Value,
            queue: Queue):
        """
        Custom loss function using PyXspec to calculate statistic for Poisson data
        and Gaussian background (PGStat) and L1 loss for parameters that exceed limits

        params
        ----------
        total : int
            Total number of spectra
        names : list[string]
            Spectra names
        params : ndarray
            Output from CNN of parameter predictions between 0 & 1
        counter : Value
            Number of spectra fitted
        queue : Queue
            Multiprocessing queue to add PGStat loss
        """
        losses = []
        os.chdir(self.dirs[1])

        # Limit parameters
        margin = np.minimum(1e-6, 1e-6 * (self.param_limits[:, 1] - self.param_limits[:, 0]))
        param_min = self.param_limits[:, 0] + margin
        param_max = self.param_limits[:, 1] - margin

        # Loop through each spectrum in the batch
        for i, name in enumerate(names):
            params = np.clip(params, a_min=param_min, a_max=param_max)

            # Load spectrum
            xspec.Spectrum(name)
            xspec.AllData.ignore('0-0.3 10.0-**')

            # Calculate fit statistic
            losses.append(self.fit_statistic(params[i]))

            xspec.AllData.clear()

            # Increase progress
            with counter.get_lock():
                counter.value += 1

            progress_bar(counter.value, total)

        # Average loss of batch
        queue.put(sum(losses) / len(names))

        os.chdir(self.dirs[0])


def progress_bar(i: int, total: int, text: str = ''):
    """
    Terminal progress bar

    Parameters
    ----------
    i : integer
        Current progress
    total : integer
        Completion number
    text : string, default = '
        Optional text to place at the end of the progress bar
    """
    length = 50
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = '█' * filled + '-' * (length - filled)
    print(f'\rProgress: |{bar_fill}| {int(percent)}%\t{text}\t', end='')

    if i == total:
        print()


def calculate_saliency(
        plots_dir: str,
        loaders: tuple[DataLoader, DataLoader],
        device: torch.device,
        encoder: Network,
        decoder: Network):
    """
    Generates saliency values for autoencoder & decoder
    Prints stats on decoder parameter significance

    Parameters
    ----------
    plots_dir : string
        Directory to save plots
    loaders : tuple[DataLoader, DataLoader]
        Autoencoder & decoder validation dataloaders
    device : device
        Which device type PyTorch should use
    encoder : Network
        Encoder half of the network
    decoder : Network
        Decoder half of the network
    """
    # Constants
    d_spectra, d_parameters, _ = next(iter(loaders[0]))
    e_spectra = next(iter(loaders[1]))[0][:8].to(device)

    # Network initialization
    decoder.train()
    encoder.train()
    d_spectra = d_spectra.to(device)
    d_parameters = d_parameters.to(device)
    d_parameters.requires_grad_()
    e_spectra.requires_grad_()

    # Generate predictions
    d_output = decoder(d_parameters)
    e_output = decoder(encoder(e_spectra))

    # Perform backpropagation
    d_loss = torch.nn.MSELoss()(d_output, d_spectra)
    e_loss = torch.nn.MSELoss()(e_output, e_spectra)
    d_loss.backward()
    e_loss.backward()

    # Calculate saliency
    d_saliency = d_parameters.grad.data.abs().cpu()
    e_saliency = e_spectra.grad.data.abs().cpu()

    # Measure impact of input parameters on decoder output
    parameter_saliency = torch.mean(d_saliency, dim=0)
    parameter_impact = parameter_saliency / torch.min(parameter_saliency)
    parameter_std = torch.std(d_saliency, dim=0) / torch.min(parameter_saliency)

    print(
        f'\nParameter impact on decoder:\n{parameter_impact.tolist()}'
        f'\nParameter spread:\n{parameter_std.tolist()}\n'
    )

    e_spectra = e_spectra.cpu().detach()
    e_output = e_output.cpu().detach()
    e_saliency = e_saliency.cpu()

    plot_saliency(plots_dir, e_spectra, e_output, e_saliency)
