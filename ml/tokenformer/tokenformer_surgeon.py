from abc import abstractmethod, ABC
import torch
from torch import nn

from cray_infra.util.get_config import get_config

import math

import logging

logger = logging.getLogger(__name__)


class TokenformerAdapter(nn.Module):
    def __init__(self, layer, hidden_size, device):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.num_heads = get_config()["tokenformer_num_heads"]
        self.head_dim = hidden_size // self.num_heads
        self.tokenformer_r = get_config()["tokenformer_r"]

        self.tokenformer_k = nn.Parameter(
            torch.zeros(self.num_heads, self.hidden_size, device=device)
        )
        self.tokenformer_v = nn.Parameter(
            torch.zeros(self.num_heads, self.hidden_size * self.tokenformer_r, device=device)
        )

        self.tokenformer_p = nn.Parameter(
            torch.zeros(self.tokenformer_r, self.hidden_size, device=device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        k_gain = 3.0 / math.sqrt(self.hidden_size / self.num_heads)
        v_gain = 3.0 / math.sqrt(self.hidden_size)

        nn.init.normal_(self.tokenformer_k, std=k_gain)
        nn.init.uniform_(self.tokenformer_v, a=-v_gain, b=v_gain)
        nn.init.zeros_(self.tokenformer_p)

    # Call layer with all inputs and kwargs
    def forward(self, query: torch.Tensor):
        all_base_layer_results = self.layer(query)

        tokenformer_results = self.tokenformer_op(query)

        if isinstance(all_base_layer_results, tuple):
            base_layer_results = all_base_layer_results[0]
        else:
            base_layer_results = all_base_layer_results

        # sum the two outputs
        layer_and_adaptor_sum = base_layer_results + tokenformer_results

        if isinstance(all_base_layer_results, tuple):
            results = (layer_and_adaptor_sum,) + all_base_layer_results[1:]
        else:
            results = layer_and_adaptor_sum

        return results

    def tokenformer_op(self, query):

        q = query.view(-1, self.num_heads, self.hidden_size // self.num_heads).transpose(0, 1)
        k = self.tokenformer_k.view(
            -1, self.num_heads, self.hidden_size // self.num_heads
        ).transpose(0, 1)
        v = self.tokenformer_v.view(
            -1, self.num_heads, self.hidden_size * self.tokenformer_r // self.num_heads
        ).transpose(0, 1)

        result = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,  # should be false for tokenformer
        )

        proj_down = (
            result.transpose(0, 1).contiguous().view([-1, self.hidden_size, self.tokenformer_r])
        )

        # tokenformer_p dims are [tokenformer_r, hidden_size]
        # query dims are [batch size, length, 1, hidden_size]
        # proj_down are [batch size, length, hidden_size, tokenformer_r]

        query_batch = query.view([-1, 1, self.hidden_size])

        # logger.info(f"query shape: {query.shape}")
        # logger.info(f"query batch shape: {query_batch.shape}")
        # logger.info(f"proj_down shape: {proj_down.shape}")
        # logger.info(f"tokenformer_p shape: {self.tokenformer_p.shape}")

        result = torch.bmm(query_batch, proj_down) @ self.tokenformer_p

        # logger.info(f"result shape: {result.shape}")

        return result.view(query.shape)

    # Visualize the size of the parameters
    def __repr__(self):
        return (
            f"TokenformerAdapter(\nhidden_size={self.hidden_size}\n(layer): "
            + self.layer.__repr__()
            + "\n)"
        )


class TokenformerSurgeon(ABC):

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def _is_mlp_layer(self, layer_name):
        return "mlp" in layer_name.split(".")[-1]

    def _recursive_setattr(self, obj, attr, value):
        attr = attr.split(".", 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            self._recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    def update_mlp(self, name, layer):
        """Try to wrap the layer with a TokenformerAdaptor."""
        if not self._is_mlp_layer(name):
            return

        logger.info(f"Wrapping layer {name} with TokenformerAdaptor")

        # Wrap the layer with a TokenformerAdapter
        self._recursive_setattr(
            self.model,
            name,
            TokenformerAdapter(layer, self.model.config.hidden_size, device=self.device),
        )

    def insert_adapter_modules(self):
        # Add tokenformer adapters for mlp and attention
        for name, layer in self.model.named_modules():
            self.update_mlp(name, layer)

        return self.model
