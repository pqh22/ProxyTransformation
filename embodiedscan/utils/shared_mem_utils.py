import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing.shared_memory

def create_or_open_shared_memory(shm_name, size=10):
    try:
        # Try to create the shared memory; if it exists, an error is thrown
        shm = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=True, size=size)
        print("Shared memory created.")
    except FileExistsError:
        # If it exists, open the existing shared memory
        shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
        print("Shared memory already exists, connected to existing one.")
    return shm

def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size

def sa_create(name, var):
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x
