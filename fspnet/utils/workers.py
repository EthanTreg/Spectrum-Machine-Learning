"""
Utility function for workers that use MPI
"""
import pickle
from argparse import ArgumentParser

import xspec
from mpi4py import MPI


def initialize_worker() -> tuple[int, int, str, dict]:
    """
    Initializes MPI workers

    Returns
    -------
    tuple[integer, integer, string, dictionary]
        Worker rank, number of cpus, worker directory and worker data
    """
    # Initialize worker
    rank = 0
    comm = MPI.COMM_WORLD
    cpus = comm.Get_size()

    # Retrieve script arguments
    parser = ArgumentParser()
    parser.add_argument('worker_dir')
    args = parser.parse_args()
    worker_dir = args.worker_dir

    # Get worker rank if multiprocessing is used
    if cpus != 1:
        rank = comm.Get_rank()
        print(f'Worker {rank} starting...')

    with open(f'{worker_dir}worker_data.pickle', 'rb') as file:
        data = pickle.load(file)

    return rank, cpus, worker_dir, data


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
    xspec.AllModels.setEnergies('0.002 200 2000 log')
    xspec.Xset.abund = 'wilm'

    # Fit options
    xspec.Fit.statMethod = 'pgstat'
    xspec.Fit.query = 'no'
    xspec.Fit.nIterations = iterations

    return model
