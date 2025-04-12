
from cray_infra.util.get_config import get_config

from cray_megatron.models.tokenformer.tokenformer_model_manager import TokenformerModelManager

def get_model_manager():
    config = get_config()

    return TokenformerModelManager()


