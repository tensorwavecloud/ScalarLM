from cray_infra.util.get_job_config import get_job_config

from cray_megatron.megatron.distribution.apply_distribution_strategy import apply_distribution_strategy

from cray_megatron.models.model_manager_base import ModelManagerBase

from cray_megatron.models.diffusion_forcing.diffusion_forcing_model import (
    DiffusionForcingModel,
    DiffusionForcingModelConfig,
)

from transformers import AutoTokenizer


class DiffusionForcingModelManager(ModelManagerBase):
    def load_model(self):

        tokenizer = load_tokenizer()

        config = get_diffusion_forcing_config(tokenizer)

        model_info = {
            "model_name": "diffusion_forcing",
            "model": DiffusionForcingModel(config=config),
            "tokenizer": tokenizer,
        }

        model_info = apply_distribution_strategy(model_info)

        return model_info


def get_diffusion_forcing_config(tokenizer):
    job_config = get_job_config()

    model_config = DiffusionForcingModelConfig(
        vocab_size=tokenizer.vocab_size + 4,
        **dict(job_config["diffusion_forcing_config"])
    )

    return model_config


def load_tokenizer():
    return AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
