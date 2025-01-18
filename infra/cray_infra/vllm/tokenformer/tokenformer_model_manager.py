import torch
from torch import nn
from safetensors.torch import safe_open
from pathlib import Path
from typing import Optional, Any, Dict, List
import os
from tokenformer.tokenformer_surgeon import TokenformerSurgeon, TokenformerAttentionAdapter
from vllm.model_executor.models import SupportsLoRA
from vllm.lora.models import get_lora_id
from vllm.logger import init_logger

from cray_infra.vllm.adapter_commons.models import AdapterModel, AdapterModelManager
from cray_infra.vllm.attention import AttentionMetadata, AttentionType

logger = init_logger(__name__)

class vLLMTokenformerAttentionAdapter(TokenformerAttentionAdapter):
    def __init__(self, layer, hidden_size, device):
        super().__init__(layer, hidden_size, device)

    def forward(
        self,
        query,
        key,
        value,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:

        base_layer_results = self.layer(query=query,
                                        key=key,
                                        value=value,
                                        kv_cache=kv_cache,
                                        attn_metadata=attn_metadata,
                                        attn_type=attn_type)

        seq_len = query.shape[0]
        new_shape = [-1, self.layer.num_heads, seq_len, self.layer.head_dim]
        reshaped_query = torch.reshape(query, new_shape)
        reshaped_base_layer_results = torch.reshape(base_layer_results, new_shape)
        result = super().forward(reshaped_query, reshaped_base_layer_results)
        return torch.reshape(result, [-1, self.layer.num_heads * self.layer.head_dim])

class vLLMTokenformerSurgeon(TokenformerSurgeon):

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

        # Wrap the layer with a TokenformerAttentionAdapter
        self._recursive_setattr(self.model, name, vLLMTokenformerAttentionAdapter(layer, layer.head_dim, self.device))

class TokenformerModel(AdapterModel):
    """A tokenformer pre-trained model."""

    def __init__(self, tokenformers: Dict[str, torch.Tensor]) -> None:
        super().__init__(get_lora_id())
        self.tokenformers = tokenformers

    @classmethod
    def from_local_checkpoint(cls, model_dir: str, device: torch.device) -> "TokenformerModel":
        # Find all files that match the pattern Path(model_dir) / "model.*.safetensors"
        files = list(Path(model_dir).glob("*.safetensors"))

        if len(files) == 0:
            raise FileNotFoundError(f"Tokenformer tensor file not found: {model_dir}")


        for tokenformer_tensor_path in files:
            tokenformers = {}
            with safe_open(tokenformer_tensor_path, framework="pt") as f:
                for module in f.keys():
                    if any(key in module for key in ("tokenformer", "lm_head")):
                        tokenformers[module] = f.get_tensor(module).to(device)

        return cls(tokenformers)

class TokenformerModelManager(AdapterModelManager):
    """A manager that manages tokenformer models."""

    def __init__(
        self,
        model: SupportsLoRA,
        device: torch.device,
    ):
        self.model = vLLMTokenformerSurgeon(model, device).insert_adapter_modules()
        self._registered_adapters: Dict[int, Any] = {}
        self._active_adapters: List[int] = []
        self.tokenformer_model_cls = TokenformerModel

    @property
    def capacity(self) -> int:
        pass

    @property
    def adapter_slots(self) -> int:
        pass


    def activate_adapter(self, adapter_id: int) -> bool:
        if adapter_id in self._active_adapters:
            return False

        logger.info("Activating Tokenformer - adapter id: %d", adapter_id)

        model_state_dict = self.model.state_dict()
        for id, adapter in self._registered_adapters.items():
            tokenformers = adapter.tokenformers
            for key, value in tokenformers.items():
                model_state_dict[key] = value

        self.model.load_state_dict(model_state_dict)
        self._active_adapters.append(adapter_id)
        return True

    def deactivate_adapter(self, adapter_id: int) -> bool:
        if adapter_id not in self._active_adapters:
            return False

        self._active_adapters.remove(adapter_id)
        return True

    def add_adapter(self, adapter: TokenformerModel) -> bool:
        self._registered_adapters[adapter.id] = adapter

    def set_adapter_mapping(self, mapping: Any) -> None:
        pass

    def remove_adapter(self, adapter_id: int) -> bool:
        pass

    def remove_all_adapters(self) -> None:
        pass

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        pass

    def list_adapters(self) -> Dict[int, Any]:
        return self._registered_adapters

    def pin_adapter(self, adapter_id: int) -> bool:
        pass
