from cray_megatron.huggingface.download_model import download_model
from cray_megatron.megatron.distribution.apply_distribution_strategy import (
    apply_distribution_strategy,
)

from tokenformer.llama_tokenformer_model import create_llama_tokenformer_model

from cray_infra.util.get_job_config import get_job_config
from cray_infra.util.get_config import get_config

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch

import logging
import time

logger = logging.getLogger(__name__)


def load_tokenformer_model():
    start_time = time.time()
    model_info = load_model_config()

    model_info = apply_tokenformer_adapter(model_info)

    model_info = apply_distribution_strategy(model_info)

    model_info = materialize_model(model_info)

    model_info = load_checkpoint_weights_if_exist(model_info)

    total_time = time.time() - start_time
    logger.info(f"Total model loading time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
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


def materialize_model(model_info):
    download_model(model_info["model_name"])

    start_time = time.time()
    model_info["model"] = AutoModelForCausalLM.from_pretrained(
        model_info["model_name"],
        torch_dtype="auto",           # Use model's native dtype
        device_map="auto",            # Enable Big Model Inference
        low_cpu_mem_usage=True,       # Reduce CPU memory usage
        _fast_init=True               # Skip weight initialization (default True)
        )
    total_time = time.time() - start_time
    logger.info(f"from_pretrained latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    start_time = time.time()
    model_info["model"] = create_llama_tokenformer_model(
        model_info["model"], model_info["distribution_strategy"]["device"]
    )
    total_time = time.time() - start_time

    logger.info(f"create_llama_tokenformer_model latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    start_time = time.time()
    config = get_config()
    config_dtype = config["dtype"]
    dtype = (
        torch.float16
        if config_dtype == "float16"
        else torch.float32 if config_dtype == "float32" else torch.bfloat16
    )
    logger.info(f"Converting model to {dtype}...")

    model_info["model"] = model_info["model"].to(dtype=dtype)

    total_time = time.time() - start_time
    logger.info(f"model dtype conversion latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    if (
        "distribution_strategy" in model_info
        and "strategy" in model_info["distribution_strategy"]
    ):
        model_info["model"] = model_info["distribution_strategy"]["strategy"](
            model_info["model"]
        )

    logger.info(f"Model: {model_info['model']}")

    model_info["model"].to(model_info["distribution_strategy"]["device"])

    return model_info


def load_checkpoint_weights_if_exist(model_info):
    return model_info
