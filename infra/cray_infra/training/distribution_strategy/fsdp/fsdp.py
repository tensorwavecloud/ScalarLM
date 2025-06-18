import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from gpu_aware_mpi import get_size, get_rank, allgather, reduce_scatter
from collections import defaultdict
from cray_infra.training.metrics import get_model_memory_footprint

import time
import logging

logger = logging.getLogger(__name__)


class FSDPLayer(nn.Module):
    def __init__(self, module, should_checkpoint=False):
        super().__init__()
        self.module = module

        self.module.register_full_backward_hook(self._full_backward_hook)

        self.should_checkpoint = should_checkpoint

        self.perf_metrics = {
            "shard": {
                "time": 0.0,
            },
            "gather": {
                "time": 0.0,  # Placeholder for elapsed time in seconds
                "bytes": 0,  # Placeholder for bytes processed
            },
            "forward_pass": {"time": 0.0},  # Placeholder for elapsed time in seconds
            "free_params": {"time": 0.0},  # Placeholder for elapsed time in seconds
            "reduce_scatter": {"time": 0.0},  # Placeholder for elapsed time in seconds
        }

        start = time.time()
        self.shard_parameters()
        end = time.time()
        self.perf_metrics["shard"]["time"] += end - start

    def _full_backward_hook(self, module, grad_input, grad_output):
        self.free_params()

    def shard_parameters(self):
        rank = get_rank()
        self.sharded_parameter_metadata = {}

        logger.debug(f"Rank {rank}: Sharding parameters for {self.module}")

        for name, param in list(self.module.named_parameters(recurse=False)):
            shard, metadata_dict = shard_tensor(param)

            self.sharded_parameter_metadata[name] = metadata_dict

            logger.debug(
                f" Rank {rank}: Sharding parameter {name} with shape {param.shape} into {metadata_dict}, new shape {shard.shape}"
            )

            setattr(
                self.module,
                "shard_" + name,
                nn.Parameter(shard, requires_grad=param.requires_grad),
            )
            delattr(self.module, name)
            setattr(self.module, name, shard)

        logger.debug(
            f" Rank {rank}: Sharded parameters are {[i[0] for i in self.module.named_parameters(recurse=False)]}"
        )

    def forward(self, *args, **kwargs):
        if self.should_checkpoint:
            return checkpoint(self.forward_op, *args, use_reentrant=False, **kwargs)
        else:
            return self.forward_op(*args, **kwargs)

    def forward_op(self, *args, **kwargs):
        # gather all params
        start = time.time()
        self.gather_all_parameters()
        end = time.time()
        self.perf_metrics["gather"]["time"] += end - start

        # run forward pass
        start = time.time()
        result = self.module(*args, **kwargs)
        end = time.time()
        self.perf_metrics["forward_pass"]["time"] += end - start

        # free params
        start = time.time()
        self.free_params()
        end = time.time()
        self.perf_metrics["free_params"]["time"] += end - start

        return result

    def gather_all_parameters(self):
        rank = get_rank()
        logger.debug(f"Rank {rank}: Gathering parameters for {self.module}")

        for name, param in self.module.named_parameters(recurse=False):

            if not name.startswith("shard_"):
                logger.debug(f" Rank {rank}: Skipping parameter {name}")
                continue

            # Remove _shard_ prefix
            name = name[6:]

            # Get metadata for this parameter
            metadata_dict = self.sharded_parameter_metadata[name]

            # Gather all shards and reconstruct the full tensor
            full_tensor = all_gather_op(param, metadata_dict, self.perf_metrics)

            logger.debug(
                f" Rank {rank}: Gathered parameter {name} with shape {full_tensor.shape}"
            )

            # Copy the full tensor back to the original parameter
            setattr(self.module, name, full_tensor)

    def free_params(self):
        for name, param in self.module.named_parameters(recurse=False):

            if not name.startswith("shard_"):
                continue

            # Remove _shard_ prefix
            name = name[6:]

            if hasattr(self.module, name):
                if getattr(self.module, name).data_ptr() != param.data.data_ptr():
                    delattr(self.module, name)

            setattr(self.module, name, param.data)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class SimpleFSDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_memory_footprint = get_model_memory_footprint(self.model)
        self._wrap_layers(model)

    def _wrap_layers(self, module):
        for name, child in module.named_children():
            params = list(child.parameters(recurse=False))

            grand_children = list(child.children())

            has_grand_children = False
            if len(grand_children) > 0:
                has_grand_children = True

            all_params = list(child.parameters(recurse=False))
            any_requires_grad = any(param.requires_grad for param in all_params)
            # checkpoint if model memory footprint is > 32GB and has grand children and has any requires_grad
            should_checkpoint = (
                self.model_memory_footprint > (32 * 1024**3)
                and has_grand_children
                and any_requires_grad
            )

            if len(params) > 0:
                wrapped = FSDPLayer(child, should_checkpoint=should_checkpoint)
                setattr(module, name, wrapped)

            if has_grand_children:
                self._wrap_layers(child)

    def forward(self, *args, **kwargs):
        result = self.model(*args, **kwargs)

        rank = get_rank()
        if rank == 0:
            metrics_str = "\n"
            # Aggregate and print metrics
            aggregated = aggregate_perf_metrics(self.model)
            for op, metrics in aggregated.items():
                total_time = "{:.2f}".format(metrics["time"])
                metrics_str += f"{op}:\n  Total time: {total_time} s\n"

            logger.debug(metrics_str)

        return result

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def unwrap_model(self):
        unwrapped_state_dict = {}

        self.unwrap_layers(
            prefix="",
            module=self.model,
            unwrapped_state_dict=unwrapped_state_dict,
        )

        return unwrapped_state_dict

    def unwrap_layers(self, prefix, module, unwrapped_state_dict):
        rank = get_rank()
        for name, child in module.named_children():
            if isinstance(child, FSDPLayer):
                logger.debug(f" Rank {rank}: Unwrapping module {prefix}{name}")
                for param_name, param in child.module.named_parameters(recurse=False):
                    if param.requires_grad:
                        if param_name.startswith("shard_"):
                            # Remove _shard_ prefix
                            param_name = param_name[len("shard_") :]

                            # Get metadata for this parameter
                            metadata_dict = child.sharded_parameter_metadata[param_name]

                            # Gather all shards and reconstruct the full tensor
                            full_tensor = all_gather_op(param, metadata_dict)

                            unwrapped_state_dict[f"{prefix}{name}.{param_name}"] = (
                                full_tensor.to(torch.device("cpu"))
                            )

                        else:
                            unwrapped_state_dict[f"{prefix}{name}.{param_name}"] = (
                                param.to(torch.device("cpu"))
                            )

                        logger.debug(
                            f" Rank {rank}: Unwrapping parameter {prefix}{name}.{param_name}"
                        )
            else:
                logger.debug(f" Rank {rank}: Skipping module {prefix}{name}")
                self.unwrap_layers(
                    prefix=prefix + name + ".",
                    module=child,
                    unwrapped_state_dict=unwrapped_state_dict,
                )


def get_fsdp_layers(module):
    """Recursively collect all FSDPLayer instances"""
    fsdp_layers = []
    for child in module.children():
        if isinstance(child, FSDPLayer):
            fsdp_layers.append(child)
        fsdp_layers.extend(get_fsdp_layers(child))
    return fsdp_layers


def aggregate_perf_metrics(module):

    fsdp_layers = get_fsdp_layers(module)

    """Sum metrics across all FSDP layers"""
    aggregated = defaultdict(lambda: {"time": 0, "bytes": 0})
    for layer in fsdp_layers:
        for op, metrics in layer.perf_metrics.items():
            aggregated[op]["time"] += metrics.get("time", 0)
            if "bytes" in aggregated[op]:
                aggregated[op]["bytes"] += metrics.get("bytes", 0)
    return dict(aggregated)


def shard_tensor(tensor):
    """Evenly shard tensor across ranks with padding if needed.
    Returns (shard, metadata_dict) where metadata_dict contains
    {rank: (original_numel, original_shape)} for all ranks."""
    world_size = get_size()
    rank = get_rank()

    original_shape = tensor.shape
    original_numel = tensor.numel()

    # Calculate padding for equal division
    padded_numel = ((original_numel + world_size - 1) // world_size) * world_size
    padding = padded_numel - original_numel

    # Pad tensor if needed
    if padding > 0:
        tensor_padded = torch.cat(
            [
                tensor.view(-1),
                torch.zeros(padding, device=tensor.device, dtype=tensor.dtype),
            ]
        )
    else:
        tensor_padded = tensor.view(-1)

    # Split into equal shards
    shard_size = padded_numel // world_size
    start = rank * shard_size
    shard = tensor_padded[start : start + shard_size].clone()

    # Gather metadata from all ranks
    local_metadata = torch.tensor(
        [original_numel, *original_shape, shard_size, padding], dtype=torch.long
    )
    all_metadata = torch.zeros((world_size, local_metadata.numel()), dtype=torch.long)
    allgather(local_metadata, all_metadata.view(-1))

    # Reshape all_metadata back to 2D after allgather
    all_metadata = all_metadata.view(world_size, -1)

    # Create a dictionary of metadata keyed by rank
    metadata_dict = {rank: all_metadata[rank].tolist() for rank in range(world_size)}

    # Convert metadata back to original format
    metadata_dict = {
        rank: (meta[0], tuple(meta[1:-2]), meta[-2], meta[-1])
        for rank, meta in metadata_dict.items()
    }

    return shard, metadata_dict


def trim_padding(all_tensors, rank, world_size, metadata_dict):

    original_numel, _, shard_size, padding = metadata_dict[rank]

    if padding == 0:
        return all_tensors

    # all_tensors is a list of tensors that have been collected
    if padding > shard_size:
        # Calculate the number of fully padded tensors
        fully_padded_tensors = padding // shard_size

        # Remove fully padded tensors
        all_tensors = all_tensors[:-fully_padded_tensors]

        # Calculate the remaining padding in the last tensor
        remaining_padding = padding % shard_size

        # Trim the remaining padding from the last tensor
        if remaining_padding > 0 and len(all_tensors) > 0:
            last_tensor = all_tensors[-1]
            all_tensors[-1] = last_tensor[:-remaining_padding]
    else:
        # trim padding for last tensor
        valid_elements = original_numel - (world_size - 1) * shard_size
        all_tensors[-1] = all_tensors[-1][:valid_elements]

    return all_tensors


def collectives_all_gather(shard, metadata_dict):
    """Gather shards and reconstruct the full tensor using metadata."""
    world_size = get_size()
    rank = get_rank()

    # Prepare buffers
    orig_dtype = shard.dtype
    shard = shard.to(torch.float32)
    gathered = torch.zeros(
        shard.numel() * world_size, device=shard.device, dtype=torch.float32
    )

    # Collective operation in float32
    allgather(shard, gathered)

    # Convert gathered result back to original dtype
    gathered = gathered.to(orig_dtype)

    # Reconstruct the full tensor using metadata
    all_tensors = []
    offset = 0
    for r in range(world_size):
        shard_size = metadata_dict[r][2]
        rank_shard_flattened = gathered[offset : offset + shard_size]
        all_tensors.append(rank_shard_flattened)
        offset += shard_size

    all_tensors = trim_padding(all_tensors, rank, world_size, metadata_dict)
    concatenated = torch.cat(all_tensors)
    original_shape = metadata_dict[rank][1]

    return concatenated.reshape(original_shape)


def collectives_reduce_scatter(tensor, metadata_dict):
    """Reduce-scatter with even sharding. Returns local shard trimmed to original size."""
    world_size = get_size()
    rank = get_rank()

    # Save original dtype
    orig_dtype = tensor.dtype

    original_numel, _, shard_size, padding = metadata_dict[rank]

    # Pad tensor if needed
    tensor_padded = tensor.reshape(-1)
    if padding > 0:
        tensor_padded = torch.concatenate(
            [
                tensor_padded,
                torch.zeros(padding, device=tensor.device, dtype=tensor_padded.dtype),
            ]
        )

    # Convert to float32 for the collective
    tensor_padded = tensor_padded.to(torch.float32)
    local_shard = torch.zeros(shard_size, device=tensor.device, dtype=torch.float32)

    # Collective operation in float32
    reduce_scatter(tensor_padded, local_shard)

    # Convert result back to original dtype
    local_shard = local_shard.to(orig_dtype)

    # Trim padding on last rank using its original size from metadata_dict
    if rank == world_size - 1:
        valid_elements = original_numel - padding
        local_shard = local_shard[:valid_elements]

    return local_shard


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, metadata_dict, metrics={}):
        ctx.metadata_dict = metadata_dict
        ctx.metrics = metrics
        return collectives_all_gather(shard, metadata_dict)

    @staticmethod
    def backward(ctx, grad_output):
        metadata_dict = ctx.metadata_dict
        metrics = ctx.metrics

        start = time.time()
        result = collectives_reduce_scatter(grad_output, metadata_dict), None, None
        end = time.time()
        if "reduce_scatter" in metrics:
            metrics["reduce_scatter"]["time"] += end - start

        return result


all_gather_op = _AllGather.apply
