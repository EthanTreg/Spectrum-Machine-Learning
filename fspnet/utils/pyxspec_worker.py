"""
Worker for fitting spectral parameters using PyXspec
"""
import os
import argparse

import xspec
import numpy as np
from mpi4py import MPI


class PyXspecFitting:
    """
    Handles fitting parameters to spectrum using a model and calculates fit statistic

    Attributes
    ----------
    optimize : bool
        Whether to optimize the fit
    model : Model
        PyXspec model
    fix_params : ndarray
        Parameter number & value of fixed parameters
    fix_params_index: ndarray
        Index to insert fixed parameters into network parameter output
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
            fix_params: np.ndarray,
            optimize: bool = False,
            iterations: int = 10,
            custom_model: str = None,
            model_dir: str = None):
        """
        Parameters
        ----------
        model_name : string
            Model to use for PyXspec
        fix_params : ndarray
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
        self.fix_params = fix_params
        self.param_limits = np.empty((0, 2))
        self.fix_params_index = self.fix_params[:, 0] - np.arange(self.fix_params.shape[1]) - 1

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

            if j in fix_params[:, 0]:
                self.model(j).frozen = True
            else:
                limits = np.array(self.model(j).values)[[2, 5]]
                self.param_limits = np.vstack((self.param_limits, limits))

    def _fit_statistic(self, params: np.ndarray) -> float:
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
                params[-1] = 10
                self.model.setPars(params.tolist())

                try:
                    xspec.Fit.perform()
                except Exception:
                    self.model.setPars(params.tolist())

        # Calculate fit statistic loss
        return xspec.Fit.statistic

    def fit_loss(
            self,
            names: list[str],
            params: np.ndarray) -> list[float]:
        """
        Custom loss function using PyXspec to calculate statistic for Poisson data
        and Gaussian background (PGStat) and L1 loss for parameters that exceed limits

        Parameters
        ----------
        names : list[string]
            Spectra names
        params : ndarray
            Output from CNN of parameter predictions between 0 & 1

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
        for spectrum_params, name in zip(params, names):

            # Load spectrum
            xspec.Spectrum(name)
            xspec.AllData.ignore('0-0.3 10.0-**')

            # Calculate fit statistic
            losses.append(self._fit_statistic(spectrum_params))

            xspec.AllData.clear()

        # Average loss of batch
        return losses


def initialize_pyxspec(
        model_name: str,
        iterations: int = 10,
        custom_model: str = None,
        model_dir: str = None) -> xspec.Model:
    """
    Initialises PyXspec

    Parameters
    ----------
    model_name : string
        Name of model to load
    iterations : integer, default = 10
        Number of fitting iterations to perform during fit operation
    custom_model : string, default = None
        Name of a custom model to load
    model_dir : string, default = None
        Path to the directory containing the model for PyXspec to load

    Returns
    -------
    Model
        PyXspec model
    """
    # Prevent PyXspec terminal output
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0

    # Load custom model
    if custom_model:
        xspec.AllModels.lmod(custom_model, dirPath=model_dir)

    model = xspec.Model(model_name)
    xspec.AllModels.setEnergies('0.002 500 1000 log')
    xspec.Xset.abund = 'wilm'

    # Fit options
    xspec.Fit.statMethod = 'pgstat'
    xspec.Fit.query = 'no'
    xspec.Fit.nIterations = iterations

    return model


def worker():
    """
    Worker for calculating the PGStat loss of a batch of data
    Supports both multiprocessing using MPI or single
    processing if system doesn't have multiple threads
    Access data through
    """
    # Initialize worker
    rank = 0
    cpus = os.cpu_count()
    parser = argparse.ArgumentParser()

    # Retrieve script arguments
    parser.add_argument('working_dir')
    parser.add_argument('worker_dir')
    args = parser.parse_args()
    worker_dir = args.worker_dir

    # Change working directory
    os.chdir(args.working_dir)

    if cpus != 1:
        # Get worker rank
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print(f'Worker {rank} starting...')

    # Retrieve data from files
    data = np.load(f'{worker_dir}worker_data.npy', allow_pickle=True).item()
    job = np.loadtxt(f'{worker_dir}worker_{rank}_job.csv', delimiter=',', dtype=str)
    names = job[:, 0]
    params = job[:, 1:].astype(float)

    # Initialize PyXspec model
    model = PyXspecFitting(
        data['model'],
        data['fix_params'],
        optimize=data['optimize'],
        custom_model=data['custom_model'],
        model_dir=data['model_dir'],
    )

    # Calculate average loss of the batch
    os.chdir(data['dirs'][1])
    losses = model.fit_loss(names, params)
    os.chdir(data['dirs'][0])

    # Save results
    job = np.hstack((job, np.expand_dims(losses, axis=1)))
    np.savetxt(f'{worker_dir}worker_{rank}_job.csv', job, delimiter=',', fmt='%s')

    print(f'Worker {rank} done')


if __name__ == "__main__":
    worker()
