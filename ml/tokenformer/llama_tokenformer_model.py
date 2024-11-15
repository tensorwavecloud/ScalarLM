from ml.tokenformer.llama_tokenformer_decoder_layer import LlamaTokenformerDecoderLayer
from transformers import LlamaForCausalLM


def replace_decoder_layers(model, custom_layer_class):
    # Replace decoder layers with custom layers
    for i, layer in enumerate(model.model.layers):
        new_layer = custom_layer_class(model.config, i)
        new_layer.load_state_dict(layer.state_dict(), strict=False)
        model.model.layers[i] = new_layer
    return model
        

def create_llama_tokenformer_model(model_id):
    model = LlamaForCausalLM.from_pretrained(model_id)
    model.config.architectures = ["LlamaTokenformerModel"]
    return replace_decoder_layers(model, LlamaTokenformerDecoderLayer)

