from abc import abstractmethod, ABC
import torch
from torch import nn
from typing import Optional, Any, Dict
from infra.cray_infra.vllm.adapter_commons.models import AdapterModel, AdapterModelManager
import os
from ml.tokenformer.tokenformer_surgeon import vLLMTokenformerSurgeon
from vllm.model_executor.models import SupportsLoRA

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
        model: SupportsLoRA,
    ):
        self.model = vLLMTokenformerSurgeon(model).insert_adapter_modules()
        self.tokenformer_model_cls = TokenformerModel
    
    @property
    def capacity(self) -> int:
        pass

    @property
    def adapter_slots(self) -> int:
        pass


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
