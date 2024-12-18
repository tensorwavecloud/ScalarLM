import torch
from torch import nn
from tokenformer.tokenformer_surgeon import TokenformerAttentionAdapter, TokenformerSurgeon

class TransformersTokenformerAttentionAdapter(TokenformerAttentionAdapter):
    def __init__(self, layer, hidden_size):
        super().__init__(layer, hidden_size)
        
    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    ) -> torch.Tensor:
        base_layer_results = self.layer(query=query, 
                                        key=key, 
                                        value=value, 
                                        attn_mask=attn_mask,
                                        dropout_p=dropout_p,
                                        is_causal=is_causal)
        
        return super().forward(query, base_layer_results)
    
class TransformersTokenformerSurgeon(TokenformerSurgeon):
    
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__(model)


    def update_attn(self, name, layer):
        """Try to wrap the layer with a TokenformerAttentionAdaptor."""
        if not self._is_attn_layer(name):
            return

        # Wrap the layer with a TokenformerAttentionAdapter
        self._recursive_setattr(self.model, name, TransformersTokenformerAttentionAdapter(layer, layer.head_dim))
        