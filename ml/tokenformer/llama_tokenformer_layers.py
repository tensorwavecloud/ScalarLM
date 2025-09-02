import torch
from typing import Optional, Tuple
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    LlamaDecoderLayer,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache


class LlamaTokenformerAttention(LlamaAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def sdpa_attention_forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        if hasattr(module, "num_key_value_groups"):
            key = self.repeat_kv(key, module.num_key_value_groups)
            value = self.repeat_kv(value, module.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        if is_causal is None:
            is_causal = causal_mask is None and query.shape[2] > 1

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_output, attn_weights = self.sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaTokenformerDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaTokenformerAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        outputs = self.mlp(hidden_states)

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        hidden_states = residual + hidden_states

        return hidden_states
