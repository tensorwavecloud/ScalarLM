import torch
from torch import nn
import safetensors.torch
from typing import Optional, Any, Dict, List
from infra.cray_infra.vllm.adapter_commons.models import AdapterModel, AdapterModelManager
import os
from ml.tokenformer.tokenformer_surgeon import TokenformerSurgeon, TokenformerAttentionAdapter
from vllm.model_executor.models import SupportsLoRA
from infra.cray_infra.vllm.attention import AttentionMetadata, AttentionType
from vllm.lora.models import get_lora_id
from vllm.logger import init_logger

logger = init_logger(__name__)

class vLLMTokenformerAttentionAdapter(TokenformerAttentionAdapter):
    def __init__(self, layer, hidden_size):
        super().__init__(layer, hidden_size)
        
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
        
        modify_query = torch.reshape(query, [1, self.layer.num_heads, -1, self.layer.head_dim])
        modify_base_layer_results = torch.reshape(base_layer_results, [1, self.layer.num_heads, -1, self.layer.head_dim])
        result = super().forward(modify_query, modify_base_layer_results)
        reshaped_result = torch.reshape(result, [-1, self.layer.num_heads * self.layer.head_dim])
        return reshaped_result
    
class vLLMTokenformerSurgeon(TokenformerSurgeon):
    
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__(model)


    def _try_to_update_attn(self, name, layer):
        """Try to wrap the layer with a TokenformerAttentionAdaptor."""
        if not self._is_attn_layer(name):
            return

        # Wrap the layer with a TokenformerAttentionAdapter
        self._recursive_setattr(self.model, name, vLLMTokenformerAttentionAdapter(layer, layer.head_dim))


class TokenformerModel(AdapterModel):
    """A tokenformer pre-trained model."""

    def __init__(
        self,
        tokenformers: Dict[str, torch.Tensor]
    ) -> None:
        super().__init__(get_lora_id())
        self.tokenformers = tokenformers

    @classmethod
    def from_local_checkpoint(cls, model_dir: str) -> "TokenformerModel":
        tokenformers: Dict[str, torch.Tensor] = {}
        tokenformer_tensor_path = os.path.join(model_dir, "model.safetensors")
        if os.path.isfile(tokenformer_tensor_path):
            with safetensors.safe_open(tokenformer_tensor_path, framework="pt") as f:
                for module in f.keys():
                    if "tokenformer" in module or "lm_head" in module:
                        tokenformers[module] = f.get_tensor(module)
        
        return cls(tokenformers)

class TokenformerModelManager(AdapterModelManager):
    """A manager that manages tokenformer models."""

    def __init__(
        self,
        model: SupportsLoRA,
    ):
        self.model = vLLMTokenformerSurgeon(model).insert_adapter_modules()
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
