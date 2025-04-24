class VLLMModelManager:
    def __init__(self):
        self._models = []

    def set_registered_models(self, models):
        self._models = models

    def get_registered_models(self):
        return self._models

    def find_model(self, model_name):
        for model in self._models:
            if model_name in model:
                return model
        return None


def get_vllm_model_manager():
    """
    Returns a singleton instance of VLLMModelManager.
    """
    if not hasattr(get_vllm_model_manager, "_instance"):
        get_vllm_model_manager._instance = VLLMModelManager()
    return get_vllm_model_manager._instance
