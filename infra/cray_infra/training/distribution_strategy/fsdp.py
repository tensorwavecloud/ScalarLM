import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import numpy as np
from mpi4py import MPI

import gc
import time

import logging

logger = logging.getLogger(__name__)


class FSDPLayer(nn.Module):
    def __init__(self, module, should_checkpoint=False):
        super().__init__()
        self.module = module
        self.shard_parameters()

        self.module.register_full_backward_hook(self._full_backward_hook)

        self.should_checkpoint = should_checkpoint

    def _full_backward_hook(self, module, grad_input, grad_output):
        self.free_params()

    def shard_parameters(self):
        self.sharded_parameter_metadata = {}

        logger.info(f"Rank {rank}: Sharding parameters for {self.module}")

        for name, param in list(self.module.named_parameters(recurse=False)):
            shard, metadata_dict = shard_tensor(param)

            self.sharded_parameter_metadata[name] = metadata_dict

            logger.info(
                f" Rank {rank}: Sharding parameter {name} with shape {param.shape} into {metadata_dict}, new shape {shard.shape}"
            )

            setattr(
                self.module,
                "shard_" + name,
                nn.Parameter(shard, requires_grad=param.requires_grad),
            )
            delattr(self.module, name)
            setattr(self.module, name, shard)

        logger.info(
            f" Rank {rank}: Sharded parameters are {[i[0] for i in self.module.named_parameters(recurse=False)]}"
        )

    def forward(self, *args, **kwargs):
        if self.should_checkpoint:
            return checkpoint(self.forward_op, *args, use_reentrant=True, **kwargs)
        else:
            return self.forward_op(*args, **kwargs)

    def forward_op(self, *args, **kwargs):
        self.gather_all_parameters()
        result = self.module(*args, **kwargs)

        self.free_params()

        return result

    def gather_all_parameters(self):
        logger.info(f"Rank {rank}: Gathering parameters for {self.module}")

        for name, param in self.module.named_parameters(recurse=False):

            if not name.startswith("shard_"):
                logger.info(f" Rank {rank}: Skipping parameter {name}")
                continue

            # Remove _shard_ prefix
            name = name[6:]

            # Get metadata for this parameter
            metadata_dict = self.sharded_parameter_metadata[name]

            # Gather all shards and reconstruct the full tensor
            full_tensor = all_gather_op(param, metadata_dict)

            logger.info(
                f" Rank {rank}: Gathered parameter {name} with shape {full_tensor.shape}"
            )

            # Copy the full tensor back to the original parameter
            setattr(self.module, name, full_tensor)

    def free_params(self):
        for name, param in self.module.named_parameters(recurse=False):

            if not name.startswith("shard_"):
                logger.info(f" Rank {rank}: Skipping parameter {name}")
                continue

            # Remove _shard_ prefix
            name = name[6:]

            if hasattr(self.module, name):
                if getattr(self.module, name).data_ptr() != param.data.data_ptr():
                    delattr(self.module, name)

            setattr(self.module, name, param.data)

        gc.collect()
        torch.cuda.empty_cache()


class SimpleFSDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._wrap_layers(model)

    def _wrap_layers(self, module):
        for name, child in module.named_children():
            params = list(child.parameters(recurse=False))

            grand_children = list(child.children())

            has_grand_children = False
            if len(grand_children) > 0:
                has_grand_children = True

            if len(params) > 0:
                wrapped = FSDPLayer(child, should_checkpoint=has_grand_children)
                setattr(module, name, wrapped)

            if has_grand_children:
                self._wrap_layers(child)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()


def shard_tensor(tensor):
    """Evenly shard tensor across ranks with padding if needed.
    Returns (shard, metadata_dict) where metadata_dict contains
    {rank: (original_numel, original_shape)} for all ranks."""
    original_shape = tensor.shape
    original_numel = tensor.numel()

    # Calculate padding for equal division
    padded_numel = ((original_numel + world_size - 1) // world_size) * world_size
    padding = padded_numel - original_numel

    # Pad tensor if needed
    if padding > 0:
        tensor_padded = torch.cat(
            [tensor.view(-1), torch.zeros(padding, device=tensor.device)]
        )
    else:
        tensor_padded = tensor.view(-1)

    # Split into equal shards
    shard_size = padded_numel // world_size
    start = rank * shard_size
    shard = tensor_padded[start : start + shard_size].clone()

    # Gather metadata from all ranks
    local_metadata = (original_numel, original_shape)
    all_metadata = comm.allgather(local_metadata)  # Gather metadata from all ranks

    # Create a dictionary of metadata keyed by rank
    metadata_dict = {rank: meta for rank, meta in enumerate(all_metadata)}

    return shard, metadata_dict


def collectives_all_gather(shard, metadata_dict):
    """Gather shards and reconstruct the full tensor using metadata."""
    # Prepare buffers
    shard_numpy = shard.detach().to(torch.float32).cpu().numpy().flatten()
    gathered = np.empty(shard_numpy.size * world_size, dtype=shard_numpy.dtype)

    # Collective operation
    comm.Allgather(shard_numpy, gathered)

    # Reconstruct the full tensor using metadata
    all_tensors = []
    offset = 0
    for rank in range(world_size):
        rank_original_numel, rank_original_shape = metadata_dict[rank]
        shard_size = gathered.size // world_size  # All shards are evenly sized
        rank_shard_flattened = gathered[
            offset : offset + rank_original_numel // world_size
        ]
        all_tensors.append(rank_shard_flattened)
        offset += shard_size

    concatenated_numpy = np.concatenate(all_tensors)

    original_shape = metadata_dict[rank][1]
    return (
        torch.from_numpy(concatenated_numpy)
        .to(shard.device)
        .to(shard.dtype)
        .reshape(original_shape)
    )


def collectives_reduce_scatter(tensor, metadata_dict):
    """Reduce-scatter with even sharding. Returns local shard trimmed to original size."""
    assert tensor.numel() % world_size == 0, "Input tensor must be padded first"

    tensor_numpy = tensor.detach().float().cpu().numpy()
    shard_size = tensor.numel() // world_size
    local_shard = np.zeros(shard_size, dtype=tensor_numpy.dtype)

    # Collective operation
    comm.Reduce_scatter(tensor_numpy, local_shard, op=MPI.SUM)

    # Trim padding on last rank using its original size from metadata_dict
    rank_original_numel, original_shape = metadata_dict[rank]

    if rank == world_size - 1:
        valid_elements = rank_original_numel - (world_size - 1) * shard_size
        local_shard = local_shard[:valid_elements]

    return torch.from_numpy(local_shard).to(tensor.device).to(tensor.dtype)


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, metadata_dict):
        ctx.metadata_dict = metadata_dict
        return collectives_all_gather(shard, metadata_dict)

    @staticmethod
    def backward(ctx, grad_output):
        metadata_dict = ctx.metadata_dict
        return collectives_reduce_scatter(grad_output, metadata_dict), None


all_gather_op = _AllGather.apply
