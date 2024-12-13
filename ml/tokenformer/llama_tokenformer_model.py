import torch
from torch import nn
from transformers import LlamaForCausalLM
from ml.tokenformer.llama_tokenformer_decoder_layer import LlamaTokenformerDecoderLayer
import logging

logger = logging.getLogger(__name__)

class TokenformerMLPAdapter(nn.Module):
    def __init__(self, layer, hidden_size):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
    
        self.tokenformer_k = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.tokenformer_v = nn.Parameter(torch.zeros(hidden_size, hidden_size))

    # Call layer with all inputs and kwargs
    def forward(
        self,
        query: torch.Tensor
    ):
        base_layer_results = self.layer(query)
        
        tokenformer_results = torch.nn.functional.scaled_dot_product_attention(
            query=query, key=self.tokenformer_k, value=self.tokenformer_v,
            attn_mask=None,
            is_causal=False # should be false for tokenformer
        )
        
        # sum the two outputs
        layer_and_adaptor_sum = base_layer_results + tokenformer_results
        return layer_and_adaptor_sum    

class TokenformerAttentionAdapter(nn.Module):
    def __init__(self, layer, hidden_size):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
    
        self.tokenformer_k = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.tokenformer_v = nn.Parameter(torch.zeros(hidden_size, hidden_size))

    # Call layer with all inputs and kwargs
    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal
    ):
        base_layer_results = self.layer(query, key, value, attn_mask, dropout_p, is_causal)
        
        tokenformer_results = torch.nn.functional.scaled_dot_product_attention(
            query=query, 
            key=self.tokenformer_k, 
            value=self.tokenformer_v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False # should be false for tokenformer
        )
        
        # sum the two outputs
        layer_and_adaptor_sum = base_layer_results + tokenformer_results
        return layer_and_adaptor_sum    

def is_attn_layer(layer_name):
    if layer_name.split('.')[-1] == "attn": 
        return True
    return False

def is_mlp_layer(layer_name):
    if "mlp" in layer_name.split('.')[-1]: 
        return True
    return False
    
    
def recursive_setattr(obj, attr, value):
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)    


def try_to_update_mlp(name, layer, model):
    """Try to wrap the layer with a TokenformerMLPAdaptor."""
    if not is_mlp_layer(name):
        return

    logger.info(f"Wrapping layer {name} with TokenformerMLPAdaptor")

    # Wrap the layer with a TokenformerMLPAdapter
    recursive_setattr(model, name, TokenformerMLPAdapter(layer, model.config.hidden_size))


def try_to_update_attn(name, layer, model):
    """Try to wrap the layer with a TokenformerAttentionAdaptor."""
    if not is_attn_layer(name):
        return

    logger.info(f"Wrapping layer {name} with TokenformerAttentionAdaptor")

    # Wrap the layer with a TokenformerAttentionAdapter
    recursive_setattr(model, name, TokenformerAttentionAdapter(layer, model.config.intermediate_size))

def add_tokenformer_adapters(model):
    # Add tokenformer adapters for mlp and attention
    for name, layer in model.named_modules():
        try_to_update_mlp(name, layer, model)
        try_to_update_attn(name, layer, model)
    return model

def replace_layers(model, custom_layer_class):
    # Replace layers with custom layers
    for i, layer in enumerate(model.model.layers):
        new_layer = custom_layer_class(model.config, i)
        new_layer.load_state_dict(layer.state_dict(), strict=False)
        model.model.layers[i] = new_layer
    return model

def create_llama_tokenformer_model(model):
    model = replace_layers(model, LlamaTokenformerDecoderLayer)
    model = add_tokenformer_adapters(model)
    # Set requires_grad to False for all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Set requires_grad to True for tokenformer params and lm_head
    for name, param in model.named_parameters():
        for item in ["tokenformer", "lm_head"]:
            if item in name:
                param.requires_grad = True
        logger.debug(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    
    return model

