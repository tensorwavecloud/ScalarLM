from transformers import PreTrainedModel

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

import torch


from torch import nn

from typing import Optional

from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)


class DiffusionForcingModelConfig(LlamaConfig):
    def __init__(
        self,
        num_hidden_layers=12,
        num_diffusion_iterations=2,
        diffusion_step_size=2,
        attention_dropout=0.3,
        **kwargs
    ):
        super().__init__(
            **kwargs,
            num_hidden_layers=num_hidden_layers,
            attention_dropout=attention_dropout
        )
        self.num_diffusion_iterations = num_diffusion_iterations
        self.diffusion_step_size = diffusion_step_size


class DiffusionForcingModel(PreTrainedModel):
    config_class = DiffusionForcingModelConfig
    base_model_prefix = "model"

    def __init__(self, config: DiffusionForcingModelConfig):
        super().__init__(config=config)

        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if noise is None:
            if self.training:
                hidden_states = self._add_noise(hidden_states)
        else:
            hidden_noise = torch.matmul(noise, self.embed_tokens.weight)
            hidden_states = hidden_noise + hidden_states

        position_ids = self._get_position_ids(input_ids)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        attention_mask = self._make_attention_mask(attention_mask, input_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
            )

        return DiffusionForcingLMOutput(logits=logits, loss=loss)

    def _add_noise(self, hidden_states):
        random_magnitude = float(
            torch.randint(0, self.config.num_diffusion_iterations, (1,)).item()
        )

        random_magnitude /= self.config.num_diffusion_iterations

        noise = (
            torch.randn_like(hidden_states)
            * random_magnitude
            * self.config.initializer_range
            * 10.0
        )
        hidden_states = hidden_states + noise
        return hidden_states

    def _get_position_ids(self, input_ids):
        position_ids = torch.arange(
            input_ids.size(1), dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _make_attention_mask(self, attention_mask, input_ids):
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)

        attention_mask = torch.ones(
            (batch_size, 1, seq_length, seq_length), device=input_ids.device
        )
        return attention_mask


@dataclass
class DiffusionForcingLMOutput:
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
