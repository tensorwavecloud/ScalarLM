import torch
from gpu_aware_mpi import get_rank

import logging

logger = logging.getLogger(__name__)

def log_gpu_memory(prefix=""):
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        rank = get_rank()
        if rank == 0:
            logger.debug(f"{prefix} GPU {i}: Free={free/1e6:.2f}MB, Total={total/1e6:.2f}MB")

def get_model_memory_footprint(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    total_size = param_size + buffer_size
    return total_size  # in bytes