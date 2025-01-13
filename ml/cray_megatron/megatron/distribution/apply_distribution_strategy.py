import torch

def load_distribution_strategy():
    device = get_device()

    return {
        "device": device,
    }


def get_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return torch.device("cpu")


def apply_distribution_strategy(model_info):
    distribution_strategy = load_distribution_strategy()
    model_info["distribution_strategy"] = distribution_strategy
    return model_info
