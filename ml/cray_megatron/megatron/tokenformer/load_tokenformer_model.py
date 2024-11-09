from cray_megatron.huggingface.download_model import download_model

from cray_infra.util.get_job_config import get_job_config

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model


def load_tokenformer_model():
    model_info = load_model_config()

    model_info = apply_tokenformer_adapter(model_info)

    distribution_strategy = load_distribution_strategy()

    model_info = apply_distribution_strategy(model_info, distribution_strategy)

    model_info = materialize_model(model_info)

    model_info = load_checkpoint_weights_if_exist(model_info)

    return model_info


def load_model_config():
    job_config = get_job_config()

    model_name = job_config["llm_name"]

    model_config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_info = {
        "model_name": model_name,
        "model_config": model_config,
        "tokenizer": tokenizer,
    }

    return model_info


def apply_tokenformer_adapter(model_info):
    return model_info


def load_distribution_strategy():
    return None


def apply_distribution_strategy(model_info, distribution_strategy):
    return model_info


def materialize_model(model_info):
    download_model(model_info["model_name"])

    model_info["model"] = AutoModelForCausalLM.from_pretrained(model_info["model_name"])

    job_config = get_job_config()
    lora_config = LoraConfig(**job_config["lora_config"])

    model_info["model"] = get_peft_model(model_info["model"], lora_config)

    print(model_info["model"])

    return model_info


def load_checkpoint_weights_if_exist(model_info):
    return model_info
