import torch
from torch import nn
from safetensors.torch import safe_open
from pathlib import Path
from typing import Optional, Any, Dict, List
from collections import OrderedDict
import copy
from tokenformer.tokenformer_surgeon import (
    TokenformerSurgeon,
    TokenformerAttentionAdapter,
)
from vllm.model_executor.models import SupportsLoRA
from vllm.lora.models import get_lora_id
from vllm.logger import init_logger

from cray_infra.vllm.adapter_commons.models import AdapterModel, AdapterModelManager
from cray_infra.vllm.attention import AttentionMetadata, AttentionType

from vllm.adapter_commons.utils import (
    get_adapter,
    list_adapters,
    remove_adapter,
    deactivate_adapter,
)

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

        base_layer_results = self.layer(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            attn_type=attn_type,
        )

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

        # logger.info(f"Wrapping layer {name} with vLLMTokenformerAttentionAdaptor")

        # Wrap the layer with a TokenformerAttentionAdapter
        #self._recursive_setattr(
        #    self.model,
        #    name,
        #    vLLMTokenformerAttentionAdapter(layer, layer.head_dim, self.device),
        #)


class TokenformerModel(AdapterModel):
    """A tokenformer pre-trained model."""

    def __init__(self, tokenformers: Dict[str, torch.Tensor]) -> None:
        super().__init__(get_lora_id())
        self.tokenformers = tokenformers

    @classmethod
    def from_local_checkpoint(
        cls, model_dir: str, device: torch.device
    ) -> "TokenformerModel":
        # Find all files that match the pattern Path(model_dir) / "model.*.safetensors"
        files = list(Path(model_dir).glob("*.safetensors"))

        if len(files) == 0:
            raise FileNotFoundError(f"Tokenformer tensor file not found: {model_dir}")

        tokenformers = {}
        for tokenformer_tensor_path in files:
            with safe_open(tokenformer_tensor_path, framework="pt") as f:
                for module in f.keys():
                    if any(key in module for key in ("tokenformer", "lm_head")):
                        logger.info(f"Loading {module} from {tokenformer_tensor_path}")
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
        self._active_adapters: Dict[int, Any] = {}
        self.tokenformer_model_cls = TokenformerModel
        self.dtype = next(self.model.parameters()).dtype
        self.orig_lm_head = copy.deepcopy(
            {
                k: v.to(self.dtype)
                for k, v in self.model.state_dict().items()
                if "lm_head" in k
            }
        )

    def activate_adapter(self, adapter_id: int) -> bool:
        if (
            adapter_id not in self._registered_adapters
            or adapter_id in self._active_adapters
        ):
            return False

        logger.info(f"Activating Tokenformer - {adapter_id}")

        logger.info(f"Model is {self.model}")

        model_state_dict = self.model.state_dict()
        tokenformers = self._registered_adapters[adapter_id].tokenformers

        for key, value in self.orig_lm_head.items():
            logger.info(f"Loading original lm head {key} from adapter {adapter_id}")
            model_state_dict[key] = value

        for key, value in tokenformers.items():
            logger.info(f"Loading {key} from adapter {adapter_id}")
            model_state_dict[key] = value

        load_result = self.model.load_state_dict(model_state_dict, strict=False)

        if len(load_result.unexpected_keys) > 0:
            logger.warning(
                f"Unexpected keys in state dict: {load_result.unexpected_keys}"
            )

        self._active_adapters[adapter_id] = tokenformers
        return True

    def deactivate_adapter(self, adapter_id: int) -> bool:
        return deactivate_adapter(
            adapter_id, self._active_adapters, self._deactivate_adapter
        )

    def _deactivate_adapter(self, adapter_id: int):
        logger.info(f"Deactivating Tokenformer - {adapter_id}")
        model_state_dict = self.model.state_dict()
        tokenformers = self._registered_adapters[adapter_id].tokenformers

        for key in tokenformers:
            if "tokenformer_p" in key:
                nn.init.zeros_(model_state_dict[key])
            elif "lm_head" in key:
                model_state_dict[key] = self.orig_lm_head[key]

        self.model.load_state_dict(model_state_dict, strict=False)

    def add_adapter(self, adapter: TokenformerModel) -> bool:
        self._registered_adapters[adapter.id] = adapter

    def set_adapter_mapping(self, mapping: Any) -> None:
        pass

    def remove_adapter(self, adapter_id: int) -> bool:
        return remove_adapter(
            adapter_id, self._registered_adapters, self.deactivate_adapter
        )

    def remove_all_adapters(self) -> None:
        for id in self._registered_adapters:
            self.deactivate_adapter(id)
        self._registered_adapters.clear()
        self._active_adapters.clear()

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        get_adapter(adapter_id, self._registered_adapters)

    def list_adapters(self) -> Dict[int, Any]:
        return list_adapters(self._registered_adapters)

    def pin_adapter(self, adapter_id: int) -> bool:
        pass

    @property
    def capacity(self) -> int:
        pass

    @property
    def adapter_slots(self) -> int:
        pass
