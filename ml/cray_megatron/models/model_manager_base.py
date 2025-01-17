
from cray_megatron.models.get_latest_checkpoint_path import get_latest_checkpoint_path
from cray_megatron.models.does_any_checkpoint_exist import does_any_checkpoint_exist

from abc import ABC, abstractmethod

class ModelManagerBase(ABC):
    @abstractmethod
    def load_model(self):
        pass

    def does_any_checkpoint_exist(self):
        return does_any_checkpoint_exist()

    def get_latest_checkpoint_path(self):
        return get_latest_checkpoint_path()

