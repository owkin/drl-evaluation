"""Wrapper to launch jobs with Ray."""
import os
import subprocess

import ray

MEM_GPU = 15360
concurent = int(os.environ["concurent"])
cmd = os.environ["CMD"]


# Pancancer
CPU_TO_USE_PC_OS = 8
GPU_TO_USE_PC_OS = 0.5

# GE
CPU_TO_USE_GE = 12
GPU_TO_USE_GE = 0.5


@ray.remote(num_cpus=CPU_TO_USE_GE, num_gpus=GPU_TO_USE_GE)
def run_benchmark():
    """Wrapper to run jobs.

    Returns:
        subprocess
    """
    ret = subprocess.run(["python"] + cmd.split(), check=False)
    return ret


# Wait for all the results to complete
results = ray.get([run_benchmark.remote()])

ray.shutdown()
