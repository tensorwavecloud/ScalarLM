from cray_infra.training.get_latest_model import get_start_time

from sortedcontainers import SortedDict

class VLLMModelManager:
    def __init__(self):
        self._models = SortedDict()

    def set_registered_models(self, models):
        self._models = { get_start_time(model): model for model in models }

    def get_registered_models(self):
        return self._models.values()

    def register_model(self, model):
        start_time = get_start_time(model)
        self._models[start_time] = model


def get_vllm_model_manager():
    """
    Returns a singleton instance of VLLMModelManager.
    """
    if not hasattr(get_vllm_model_manager, "_instance"):
        get_vllm_model_manager._instance = VLLMModelManager()
    return get_vllm_model_manager._instance
