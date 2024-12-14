from ml.tokenformer.llama_tokenformer_layers import LlamaTokenformerDecoderLayer
from infra.cray_infra.vllm.tokenformer.tokenformer_model_manager import TransformersTokenformerModelManager
import logging

logger = logging.getLogger(__name__)

def replace_layers(model, custom_layer_class):
    # Replace layers with custom layers
    for i, layer in enumerate(model.model.layers):
        new_layer = custom_layer_class(model.config, i)
        new_layer.load_state_dict(layer.state_dict(), strict=False)
        model.model.layers[i] = new_layer
    return model

def create_llama_tokenformer_model(model):
    model = replace_layers(model, LlamaTokenformerDecoderLayer)
    tokenformer_model_manager = TransformersTokenformerModelManager(model)
    tokenformer_model = tokenformer_model_manager.model
    # Set requires_grad to False for all parameters in the model
    for param in tokenformer_model.parameters():
        param.requires_grad = False

    # Set requires_grad to True for tokenformer params and lm_head
    for name, param in tokenformer_model.named_parameters():
        for item in ["tokenformer", "lm_head"]:
            if item in name:
                param.requires_grad = True
        logger.debug(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    
    return tokenformer_model