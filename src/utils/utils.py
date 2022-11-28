from multiprocessing import Queue, Value

import xspec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class PyXspecFitting:
    """
    Handles fitting parameters to spectrum using a model and calculates fit statistic

    Attributes
    ----------
    model : string
        Model to use for PyXspec
    fix_params : ndarray
        Parameter number & value of fixed parameters
    param_limits : tensor
        Parameter lower & upper limits

    Methods
    -------
    fit_statistic(params)
        Calculates fit statistic from spectrum and parameter predictions
    fit_loss(total, names, params, counter, queue)
        Evaluates the loss using PyXspec and each predicted parameter
    """
    def __init__(self, model: str, fix_params: np.ndarray):
        """
        Parameters
        ----------
        model : string
            Model to use for PyXspec
        fix_params : ndarray
            Parameter number & value of fixed parameters
        """
        super().__init__()
        self.fix_params = fix_params
        self.param_limits = np.empty((0, 2))

        # PyXspec initialization
        xspec.Xset.chatter = 0
        xspec.Xset.logChatter = 0
        xspec.AllModels.lmod('simplcutx', dirPath='../../../Documents/Xspec_Models/simplcutx')
        self.model = xspec.Model(model)
        xspec.AllModels.setEnergies('0.002 500 1000 log')
        xspec.Fit.statMethod = 'pgstat'

        # Generate parameter limits
        for j in range(self.model.nParameters):
            if j + 1 not in fix_params[:, 0]:
                limits = np.array(self.model(j + 1).values)[[2, 5]]
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
        fix_params_index = self.fix_params[0] - np.arange(self.fix_params.shape[1])
        params = np.insert(params, fix_params_index, self.fix_params[1])

        # Update model parameters
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
        loss = 0

        # Loop through each spectrum in the batch
        for i, name in enumerate(names):
            # Limit parameters
            param_min = self.param_limits[:, 0] + 1e-6 * (
                    self.param_limits[:, 1] - self.param_limits[:, 0])
            param_max = self.param_limits[:, 1] - 1e-6 * (
                    self.param_limits[:, 1] - self.param_limits[:, 0])

            params = np.clip(params, a_min=param_min, a_max=param_max)

            # Load spectrum
            xspec.Spectrum(name)

            # Calculate fit statistic
            loss += self.fit_statistic(params[i])

            xspec.AllData.clear()

            # Increase progress
            with counter.get_lock():
                counter.value += 1

            progress_bar(counter.value, total)

        # Average loss of batch
        queue.put(loss / len(names))


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
    x_data = np.load('../data/spectra_x_axis.npy')

    # Make sure x data size is even
    if x_data.size % 2 != 0:
        x_data = np.append(x_data[:-2], np.mean(x_data[-2:]))

    # Make sure x data size is of the same size as y data
    if x_data.size != y_data.size and x_data.size % y_data.size == 0:
        x_data = x_data.reshape(int(x_data.size / y_data.size), - 1)
        x_data = np.mean(x_data, axis=0)

    axes.scatter(x_data, y_data, label='Spectrum')
    axes.scatter(x_data, y_recon, label='Reconstruction')
    axes.set_xlabel('Energy (keV)', fontsize=12)
    axes.set_ylabel(r'$log_{10}$ Counts ($s^{-1}$ $detector^{-1}$ $keV^{-1}$)', fontsize=12)
    axes.legend(fontsize=16)


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
    plt.legend(fontsize=20)
