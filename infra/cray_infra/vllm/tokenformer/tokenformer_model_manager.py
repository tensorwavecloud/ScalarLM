import torch
from torch import nn
from typing import Optional, Any, Dict
from infra.cray_infra.vllm.adapter_commons.models import AdapterModel, AdapterModelManager
from infra.cray_infra.vllm.attention import AttentionMetadata, AttentionType
import logging
import os

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

class TokenformerModel(AdapterModel):
    """A tokenformer pre-trained model."""

    def __init__(
        self,
        tokenformers: Dict[str, nn.Parameter],
    ) -> None:
        super().__init__()
        self.tokenformers = nn.ParameterDict(tokenformers)

    @classmethod
    def from_local_checkpoint(cls, model_dir: str) -> "TokenformerModel":
        checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No .pt files found in {model_dir}")
        checkpoint_file = checkpoint_files[0]

        checkpoint_path = os.path.join(model_dir, checkpoint_file)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        tensors = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        tokenformers = {}
        for key, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor) and "tokenformer" in key:
                tokenformers[key] = nn.Parameter(tensor)
        
        return cls(tokenformers)

class TokenformerModelManager(AdapterModelManager):
    """A manager that manages tokenformer models."""

    def __init__(
        self,
        model: nn.Module,
    ):
        self.model = model
        self.tokenformer_model_cls = TokenformerModel
        self._insert_tokenformer_adapter_modules()


    def _is_attn_layer(self, layer_name):
        return layer_name.split('.')[-1] == "attn"

    def _is_mlp_layer(self, layer_name):
        return "mlp" in layer_name.split('.')[-1]

    def _recursive_setattr(self, obj, attr, value):
        attr = attr.split(".", 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            self._recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    def _try_to_update_mlp(self, name, layer, model):
        """Try to wrap the layer with a TokenformerMLPAdaptor."""
        if not self._is_mlp_layer(name):
            return

        logger.info(f"Wrapping layer {name} with TokenformerMLPAdaptor")

        # Wrap the layer with a TokenformerMLPAdapter
        self._recursive_setattr(model, name, TokenformerMLPAdapter(layer, model.config.hidden_size))

    def _try_to_update_attn(self, name, layer, model):
        """Try to wrap the layer with a TokenformerAttentionAdaptor."""
        if not self._is_attn_layer(name):
            return

        logger.info(f"Wrapping layer {name} with TokenformerAttentionAdaptor")

        # Wrap the layer with a TokenformerAttentionAdapter
        self._recursive_setattr(model, name, TokenformerAttentionAdapter(layer, model.config.hidden_size))

    def _insert_tokenformer_adapter_modules(self): 
        # Add tokenformer adapters for mlp and attention
        for name, layer in self.model.named_modules():
            self._try_to_update_mlp(name, layer, self.model)
            self._try_to_update_attn(name, layer, self.model)

    @property
    def capacity(self) -> int:
        pass

    @property
    def adapter_slots(self) -> int:
        return self.lora_config.max_loras


    def activate_adapter(self, adapter_id: int) -> bool:
        pass

    def deactivate_adapter(self, adapter_id: int) -> bool:
        pass

    def add_adapter(self, adapter: TokenformerModel) -> bool:
        pass

    def set_adapter_mapping(self, mapping: Any) -> None:
        pass

    def remove_adapter(self, adapter_id: int) -> bool:
        pass

    def remove_all_adapters(self) -> None:
        pass

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        pass

    def list_adapters(self) -> Dict[int, Any]:
        pass

    def pin_adapter(self, adapter_id: int) -> bool:
        pass


