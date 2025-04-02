import torch
from torch import nn
from typing import Optional, Tuple
from tokenformer.tokenformer_surgeon import TokenformerAttentionAdapter, TokenformerSurgeon

class TransformersTokenformerAttentionAdapter(TokenformerAttentionAdapter):
    def __init__(self, layer, hidden_size, device: torch.device):
        super().__init__(layer, hidden_size, device)

    def forward(self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        base_layer_results = self.layer(query=query,
                                        key=key,
                                        value=value,
                                        attn_mask=attn_mask,
                                        dropout_p=dropout_p,
                                        scale=scale,
                                        is_causal=is_causal)

        return super().forward(query, base_layer_results)

class TransformersTokenformerSurgeon(TokenformerSurgeon):

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        super().__init__(model, device)


    def update_attn(self, name, layer):
        """Try to wrap the layer with a TokenformerAttentionAdaptor."""
        if not self._is_attn_layer(name):
            return

        # logger.info(f"Wrapping layer {name} with TransformersTokenformerAttentionAdaptor")

        # Wrap the layer with a TokenformerAttentionAdapter
        # self._recursive_setattr(self.model, name, TransformersTokenformerAttentionAdapter(layer, layer.head_dim, self.device))

