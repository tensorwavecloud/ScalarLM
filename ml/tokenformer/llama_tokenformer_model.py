
import logging
from tokenformer.llama_tokenformer_layers import LlamaTokenformerDecoderLayer
from tokenformer.transformers_tokenformer import TransformersTokenformerSurgeon

def replace_layers(model, custom_layer_class):
    # Replace layers with custom layers
    for i, layer in enumerate(model.model.layers):
        new_layer = custom_layer_class(model.config, i)
        new_layer.load_state_dict(layer.state_dict(), strict=False)
        model.model.layers[i] = new_layer
    return model

def log_param_gradients(model, logger=logging.getLogger(__name__)):
    for name, param in model.named_parameters():
        logger.debug(f"Parameter: {name}, Requires Grad: {param.requires_grad}")


def create_llama_tokenformer_model(model, device, train_lm_head = None):
    model = replace_layers(model, LlamaTokenformerDecoderLayer)
    tokenformer_model = TransformersTokenformerSurgeon(model, device).insert_adapter_modules()

    if train_lm_head is None:
        # Big models with more than 100M parameters don't need to train the lm_head
        # and getting the gradient scale right can be tricky.
        # Finally, the lm_head can be big and slow down adaptor loading in inference.
        train_lm_head = count_parameters(tokenformer_model) < 100_000_000

    # Freeze all parameters
    for param in tokenformer_model.parameters():
        param.requires_grad = False

    # Unfreeze tokenformer and lm_head parameters
    for name, param in tokenformer_model.named_parameters():
        if any(module_name in name for module_name in ["tokenformer"]):
            param.requires_grad = True

    # If lm_head should be included in training, set it as well.
    # In some models, lm_head is tied to embeddings and not included as a param.
    # So it's best to access it directly.
    if train_lm_head:
        tokenformer_model.lm_head.weight.requires_grad = True

    # Log parameter gradients
    log_param_gradients(tokenformer_model)

    return tokenformer_model

# Define a function to count parameters
def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

