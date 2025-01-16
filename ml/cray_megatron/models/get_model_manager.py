
from cray_infra.util.get_config import get_config

from cray_megatron.models.tokenformer.tokenformer_model_manager import TokenformerModelManager
from cray_megatron.models.diffusion_forcing.diffusion_forcing_model_manager import DiffusionForcingModelManager

def get_model_manager():
    config = get_config()

    if config["model"] == "diffusion_forcing":
        return DiffusionForcingModelManager()

    return TokenformerModelManager()


