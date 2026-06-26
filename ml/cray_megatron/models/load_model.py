from cray_megatron.huggingface.download_model import download_model
from cray_megatron.megatron.distribution.apply_distribution_strategy import (
    apply_distribution_strategy,
)
from cray_megatron.collectives.main_rank_only import is_main_rank

from gpu_aware_mpi import get_size, get_rank, allgather

from adapters.add_adapters_to_model import add_adapters_to_model

from cray_infra.util.get_job_config import get_job_config
from cray_infra.util.get_config import get_config

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText

from cray_megatron.megatron.doc_mask import is_multimodal

import torch

import logging
import time

logger = logging.getLogger(__name__)


def load_model():
    start_time = time.time()
    model_info = load_model_config()

    model_info = apply_distribution_strategy(model_info)

    model_info = materialize_model(model_info)

    total_time = time.time() - start_time
    logger.info(
        f"Total model loading time: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )
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


def materialize_model(model_info):
    job_config = get_job_config()
    download_model(model_info["model_name"])

    # SDPA (PyTorch's built-in fused attention) is the default and only
    # auto-selected backend. Flash-attention support was removed: the
    # head_dim/per-layer gating and the Qwen3 family bugs made it more
    # trouble than it was worth for training, and it bloated the image
    # build. A job can still force another HF backend (e.g. "eager") via
    # attn_implementation; "auto" (the default) means SDPA.
    override = job_config.get("attn_implementation", "auto")
    attn_impl = override if (override and override != "auto") else "sdpa"

    # Multimodal wrappers (vision_config present) need the image-text-to-text
    # AutoModel: AutoModelForCausalLM rejects e.g. Qwen2VLConfig at load
    # ("Unrecognized configuration class ... for AutoModelForCausalLM"), which
    # is the Qwen2-VL TRAIN_FAILED in the cuda-spark sweep. AutoModelForImage-
    # TextToText loads both Qwen2-VL and Gemma3 conditional-generation models;
    # the text-only training forward works on them (LoRA is confined to the
    # language tower by resolve_target_modules).
    model_cls = (
        AutoModelForImageTextToText
        if is_multimodal(model_info["model_config"])
        else AutoModelForCausalLM
    )
    logger.info(
        "Loading model with %s, attn_implementation=%s",
        model_cls.__name__,
        attn_impl,
    )

    start_time = time.time()
    try:
        model_info["model"] = model_cls.from_pretrained(
            model_info["model_name"],
            torch_dtype="auto",  # Use model's native dtype
            attn_implementation=attn_impl,
            # device_map="auto",            # Enable Big Model Inference
            # low_cpu_mem_usage=True,       # Reduce CPU memory usage
            # _fast_init=True               # Skip weight initialization (default True)
        )
    except (ValueError, ImportError, RuntimeError) as e:
        # Some model configs (older architectures, custom attention heads)
        # reject SDPA at load time. eager is the universal pure-Python
        # reference path that every transformers config supports — the
        # last resort.
        if attn_impl != "eager":
            logger.warning(
                "from_pretrained refused attn_implementation=%s (%s); "
                "falling back to eager.",
                attn_impl,
                e,
            )
            attn_impl = "eager"
            model_info["model"] = model_cls.from_pretrained(
                model_info["model_name"],
                torch_dtype="auto",
                attn_implementation="eager",
            )
        else:
            raise

    total_time = time.time() - start_time
    logger.info(
        f"from_pretrained latency: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )

    if job_config.get("gradient_checkpointing", False):
        # Order matters: enable on the bare HF model before PEFT
        # wraps it. `enable_input_require_grads` is the PEFT-specific
        # incantation that lets gradients flow back through a frozen
        # base into LoRA adapters when the base is checkpointed —
        # without it the adapter params get zero grad. `use_cache`
        # has to be off because checkpointing recomputes activations
        # in backward and the kv cache assumes they were retained.
        logger.info("Enabling gradient checkpointing")
        model_info["model"].config.use_cache = False
        model_info["model"].gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model_info["model"], "enable_input_require_grads"):
            model_info["model"].enable_input_require_grads()

    start_time = time.time()
    model_info["model"] = add_adapters_to_model(
        model=model_info["model"], device=model_info["distribution_strategy"]["device"]
    )
    total_time = time.time() - start_time

    logger.info(
        f"create_tokenformer_model latency: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )
    start_time = time.time()
    # Per-job dtype (train_args["dtype"]) wins over global cray-config.yaml.
    # Both default to "auto", which means "use the model's native dtype".
    # The job-level knob is what the FAQ documents and is the right place
    # to put a per-run dtype override (e.g. fp32 on Apple Silicon CPU where
    # bf16 matmuls SIGILL).
    job_dtype = job_config.get("dtype", "auto")
    config_dtype = job_dtype if job_dtype != "auto" else get_config()["dtype"]

    if config_dtype != "auto":
        dtype = (
            torch.float16
            if config_dtype == "float16"
            else torch.float32 if config_dtype == "float32" else torch.bfloat16
        )
        logger.info(f"Converting model to {dtype}...")

        model_info["model"] = model_info["model"].to(dtype=dtype)
    else:
        logger.info("Using model's native dtype, no conversion needed.")

    total_time = time.time() - start_time
    logger.info(
        f"model dtype conversion latency: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )

    model_info["model"] = model_info["distribution_strategy"]["strategy"](
        model_info["model"]
    )

    if is_main_rank():
        logger.info(f"Model: {model_info['model']}")

    logger.info(
        f"Moving model to device: {model_info['distribution_strategy']['device']}..."
    )

    model_info["model"].to(model_info["distribution_strategy"]["device"])

    return model_info
