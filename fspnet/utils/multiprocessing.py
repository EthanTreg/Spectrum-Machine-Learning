"""
Utility functions for multiprocessing
"""
import os
import re
import subprocess
from time import time

from fspnet.utils.utils import progress_bar


def mpi_multiprocessing(cpus: int, total: int, arg: str):
    """
    Creates and tracks multiple workers running a Python module using MPI,
    tracking is done through the worker print statement 'update'

    Can track failures through worker print statement 'fail_num='

    Supports single threading with debugging support (tested with PyCharm)

    Parameters
    ----------
    cpus : integer
        Number of threads to use
    total : integer
        Total number of
    arg : string
        Python module argument after python3 -m
    """
    failure_total = count = 0
    initial_time = time()
    text = ''

    # Start workers
    if cpus == 1:
        subprocess.run(f'python3 -m {arg}'.split(), check=True)
        return

    print(f'Starting {cpus} workers...')
    with subprocess.Popen(
            f'mpiexec -n {cpus} --use-hwthread-cpus '
            f'python3 -m {arg}'.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
    ) as proc:
        # Track progress of workers
        for line in iter(proc.stdout.readline, b''):
            line = line.decode('utf-8').strip()
            fail_num = re.search(r'fail_num=(\d+)', line)

            if fail_num and count != 0:
                failure_total += int(fail_num.group(1))
                success = count * 100 / (count + failure_total)
                text = f'\tSuccess: {success:.1f} % '
            elif 'update' in line:
                count += 1
                eta = (time() - initial_time) * (total / count - 1)
                text += f'ETA: {eta:.2f} s\tProgress: {count} / {total}'
                progress_bar(count, total + 1, text=text)
                text = ''
            elif 'error' in line.lower():
                print(line)


    print(f'\nWorkers finished\tTime: {time() - initial_time:.3e} s')


def check_cpus(cpus: int) -> int:
    """
    Checks if cpus is greater than the total number of threads, or if it is less than 1,
    if so, set the number of threads to the maximum number available

    Parameters
    ----------
    cpus : integer
        Number of threads to check

    Returns
    -------
    integer
        Number of threads
    """
    if cpus < 1 or cpus > os.cpu_count():
        return os.cpu_count()

    return cpus
