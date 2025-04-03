import torch
from cray_infra.training.distribution_strategy.fsdp import SimpleFSDP

from gpu_aware_mpi import get_size, get_rank

def load_distribution_strategy():
    device = get_device()

    strategy = {
        "device": device,
    }

    if get_size() > 1:
        strategy["strategy"] = SimpleFSDP

    return strategy


def get_device():
    if torch.cuda.is_available():
        rank = get_rank()

        gpu_count = torch.cuda.device_count()

        selected_gpu = rank % gpu_count

        if gpu_count > 1:
            return torch.device(f"cuda:{selected_gpu}")

        return torch.cuda.current_device()
    else:
        return torch.device("cpu")


def apply_distribution_strategy(model_info):
    distribution_strategy = load_distribution_strategy()
    model_info["distribution_strategy"] = distribution_strategy
    return model_info
