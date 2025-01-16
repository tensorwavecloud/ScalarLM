
from cray_megatron.models.model_manager_base import ModelManagerBase

from cray_megatron.models.tokenformer.load_tokenformer_model import load_tokenformer_model

class TokenformerModelManager(ModelManagerBase):
    def load_model(self):
        return load_tokenformer_model()


