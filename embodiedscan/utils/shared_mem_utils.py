import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing.shared_memory
import pickle



# def _serialize_data(data):
#     def _serialize(data):
#         # Serialize the data using pickle with protocol 4 (or you could use protocol 5)
#         buffer = pickle.dumps(data, protocol=5)
#         return np.frombuffer(buffer, dtype=np.uint8)

#     data_list = [_serialize(x) for x in data]
#     address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
#     data_address: np.ndarray = np.cumsum(address_list)
    
#     data_bytes = np.concatenate(data_list)

#     shared_mem = shm.SharedMemory(create=True, size=len(data_bytes))
    
#     shared_mem.buf[:] = data_bytes
#     gc.collect()
#     return shared_mem, data_address

# def create_shared_memory_dict(data):
#     serialized_data = pickle.dumps(data, protocol=5)
#     shm = shared_memory.SharedMemory(name=shm_name, create=True, size=len(serialized_data))
#     shm.buf[:] = serialized_data
#     return shm

# def access_shared_memory_dict(shm_name):
#     shm = shared_memory.SharedMemory(name=shm_name)
#     serialized_data = bytes(shm.buf[:])
#     data = pickle.loads(serialized_data)
    
#     return data

# def create_or_open_shared_memory(shm_name):
#     try:
#         # Try to create the shared memory; if it exists, an error is thrown
#         shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size)
#         print("Shared memory created.")
#     except FileExistsError:
#         # If it exists, open the existing shared memory
#         shm = shared_memory.SharedMemory(name=shm_name)
#         print("Shared memory already exists, connected to existing one.")
#     return shm

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