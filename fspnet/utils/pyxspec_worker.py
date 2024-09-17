"""
Worker for fitting spectral parameters using PyXspec
"""
import os

import xspec
import numpy as np
from mpi4py import MPI
from numpy import ndarray
from netloader.utils.utils import progress_bar

from fspnet.utils.workers import initialize_worker, initialize_pyxspec


class PyXspecFitting:
    """
    Handles fitting parameters to spectrum using a model and calculates fit statistic

    Attributes
    ----------
    optimize : bool
        Whether to optimize the fit
    model : Model
        PyXspec model
    fixed_params : ndarray
        Parameter number & value of fixed parameters
    param_limits : tensor
        Parameter lower & upper limits

    Methods
    -------
    fit_loss(total, names, params, counter, queue)
        Evaluates the loss using PyXspec and each predicted parameter
    """
    def __init__(
            self,
            model_name: str,
            fixed_params: dict,
            optimize: bool = False,
            iterations: int = 10,
            custom_model: str = None,
            model_dir: str = None):
        """
        Parameters
        ----------
        model_name : string
            Model to use for PyXspec
        fixed_params : dictionary
            Parameter number (starting from 1) & value of fixed parameters
        optimize : boolean, default = False
            Whether Xspec fitting should be performed
        iterations : integer, default = 10
            Number of iterations to perform if optimize is True
        custom_model : string, default = None
            Name of a custom model to load
        model_dir : string, default = None
            Path to the directory containing the model for PyXspec to load
        """
        super().__init__()
        self.optimize = optimize
        self.fixed_params = fixed_params
        self.param_limits = np.empty((0, 2))

        # PyXspec initialization
        self.model = initialize_pyxspec(
            model_name,
            iterations=iterations,
            custom_model=custom_model,
            model_dir=model_dir
        )

        self.model(2).values = [*self.model(2).values[:5], 4.5]

        # Generate parameter limits
        for j in range(self.model.nParameters):
            j += 1

            if j in self.fixed_params:
                self.model(j).frozen = True
            else:
                limits = np.array(self.model(j).values)[[2, 5]]
                self.param_limits = np.vstack((self.param_limits, limits))

    def _fit_statistic(self, params: ndarray) -> float:
        """
        Measures the quality of fit for the given parameters

        If optimize is true, then several of iterations of fitting will be performed

        Parameters
        ----------
        params : ndarray
            params to calculate the fit statistics for

        Returns
        -------
        float
            PGStat loss divided by the degrees of freedom
        """
        # Merge fixed & free parameters
        param_indices = np.arange(params.size + len(self.fixed_params)) + 1
        param_indices = np.setdiff1d(param_indices, list(self.fixed_params.keys()))
        params = dict(zip(param_indices.tolist(), params)) | self.fixed_params

        # Update model parameters
        self.model.setPars(params)

        # If optimization is enabled, then try unless there is an error
        if self.optimize:
            try:
                xspec.Fit.perform()
            except Exception:
                self.model.setPars(params)

        # Calculate fit statistic loss
        return xspec.Fit.statistic / xspec.Fit.dof

    def _iterative_fit(self, params: ndarray, step: int = 2) -> list[float]:
        """
        Iteratively fits the model to track the fit statistics after every step iterations

        Parameters
        ----------
        params : ndarray
            params to calculate the fit statistics for
        step : integer, default = 2
            Number of fit iterations to perform before calculating the fit statistic

        Returns
        -------
        list[float]
            Fit statistics after every step
        """
        losses = []
        iterations = xspec.Fit.nIterations
        xspec.Fit.nIterations = step
        free_param_idxs = np.arange(self.model.nParameters) + 1
        free_param_idxs = np.setdiff1d(free_param_idxs, list(self.fixed_params.keys()))

        # Loop through every step iterations and calculate fit statistic
        for _ in range(0, iterations, step):
            losses.append(self._fit_statistic(params))

            # Save free parameters
            for i, idx in enumerate(free_param_idxs):
                params[i] = self.model(int(idx)).values[0]

        xspec.Fit.nIterations = iterations

        return losses

    def fit_loss(self, names: ndarray, params: ndarray, step: int = 2) -> list[float]:
        """
        Custom loss function using PyXspec to calculate statistic for Poisson data
        and Gaussian background (PGStat) and L1 loss for parameters that exceed limits

        Parameters
        ----------
        names : list[string]
            Spectra names
        params : ndarray
            params to calculate the fit statistics for
        step : integer, default = 2
            Number of fit iterations to perform before calculating the fit statistic

        Returns
        -------
        list[float]
            Loss for each spectra
        """
        losses = []

        # Limit parameters
        margin = np.minimum(1e-6, 1e-6 * (self.param_limits[:, 1] - self.param_limits[:, 0]))
        param_min = self.param_limits[:, 0] + margin
        param_max = self.param_limits[:, 1] - margin
        params = np.clip(params, a_min=param_min, a_max=param_max)

        # Loop through each spectrum in the batch
        for i, (spectrum_params, name) in enumerate(zip(params, names)):
            # Load spectrum
            xspec.Spectrum(name)
            xspec.AllData.ignore('0.0-0.3 10.0-**')

            # Calculate fit statistic
            losses.append(self._fit_statistic(spectrum_params))
            # losses.append(self._iterative_fit(spectrum_params, step=step))

            xspec.AllData.clear()

            if MPI.COMM_WORLD.Get_size() == 1:
                progress_bar(i, len(names))
            else:
                print('update')

        # Average loss of batch
        return losses


def worker():
    """
    Worker for calculating the PGStat loss of a batch of data
    Supports both multiprocessing using MPI or single
    processing if system doesn't have multiple threads
    Access data through
    """
    rank, _, worker_dir, data = initialize_worker()

    # Retrieve data from files
    job = np.loadtxt(f'{worker_dir}worker_{rank}_job.csv', delimiter=',', dtype=str)
    names = job[:, 0]
    params = job[:, 1:].astype(float)

    # Initialize PyXspec model
    model = PyXspecFitting(
        data['model'],
        data['fix_params'],
        optimize=data['optimize'],
        iterations=data['iterations'],
        custom_model=data['custom_model'],
        model_dir=data['model_dir'],
    )

    # Calculate average loss of the batch
    os.chdir(data['dirs'][1])
    losses = model.fit_loss(names, params, step=data['step'])
    os.chdir(data['dirs'][0])

    # Save results
    job = np.hstack((job, np.expand_dims(losses, axis=1)))
    # job = np.hstack((job, losses))
    np.savetxt(f'{worker_dir}worker_{rank}_job.csv', job, delimiter=',', fmt='%s')

    print(f'Worker {rank} done')


if __name__ == "__main__":
    worker()
