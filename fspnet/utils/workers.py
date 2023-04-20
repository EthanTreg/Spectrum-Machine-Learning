"""
Utility function for workers that use MPI
"""
import pickle
from argparse import ArgumentParser

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
