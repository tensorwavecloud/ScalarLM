from cray_infra.util.get_job_config import get_job_config

import time
import logging

from peft import get_peft_model, LoraConfig, TaskType

from adapters.resolve_target_modules import resolve_target_modules

logger = logging.getLogger(__name__)


def create_lora_model(model, device, train_lm_head=False):
    overall_start = time.time()

    # Step 1: Insert LoRA adapter modules
    logger.info("Starting LoRA adapter module insertion...")
    step2_start = time.time()
    job_config = get_job_config()
    lora_config = dict(job_config["lora_config"])

    # Resolve PEFT's "all-linear" shorthand ourselves: its expansion silently
    # fails for some architectures (Qwen3MoE under peft 0.19 + transformers 5.x)
    # and raises "Target modules {char-set} not found" — the qwen3-moe sweep
    # TRAIN_FAILED. For dense models this is identical to PEFT's expansion.
    lora_config["target_modules"] = resolve_target_modules(
        model, lora_config.get("target_modules", "all-linear")
    )

    logger.info(f"LoRA config: {lora_config}")

    lora_model = get_peft_model(model, LoraConfig(**lora_config))
    add_methods(lora_model)
    lora_model = lora_model.to(device)
    step2_time = time.time() - step2_start
    logger.info(
        f"LoRA adapter module insertion completed: {step2_time:.2f}s ({step2_time/60:.1f} minutes)"
    )

    # Step 2: Count parameters for train_lm_head decision
    if train_lm_head is None:
        # Big models with more than 100M parameters don't need to train the lm_head
        # and getting the gradient scale right can be tricky.
        # Finally, the lm_head can be big and slow down adaptor loading in inference.
        logger.info("Counting parameters...")
        step3_start = time.time()
        param_count = count_parameters(lora_model)
        step3_time = time.time() - step3_start
        train_lm_head = param_count < 100_000_000
        logger.info(
            f"Parameter counting completed: {step3_time:.2f}s, count={param_count:,}, train_lm_head={train_lm_head}"
        )
    else:
        step3_time = 0

    # Step 3: Freeze all parameters
    # NOTE: get_peft_model already freezes base model params, but we do this
    # explicitly here to match the original interface and ensure consistency.
    logger.info("Freezing all parameters...")
    step4_start = time.time()
    frozen_count = 0
    for param in lora_model.parameters():
        param.requires_grad = False
        frozen_count += 1
    step4_time = time.time() - step4_start
    logger.info(
        f"Parameter freezing completed: {step4_time:.2f}s, frozen {frozen_count:,} parameters"
    )

    # Step 4: Unfreeze LoRA parameters
    logger.info("Unfreezing LoRA parameters...")
    step5_start = time.time()
    unfrozen_count = 0
    for name, param in lora_model.named_parameters():
        if any(
            module_name in name
            for module_name in [
                "lora_",  # LoRA A/B weight matrices
            ]
        ):
            param.requires_grad = True
            unfrozen_count += 1
    step5_time = time.time() - step5_start
    logger.info(
        f"LoRA parameter unfreezing completed: {step5_time:.2f}s, unfrozen {unfrozen_count:,} parameters"
    )

    # Step 5: Handle lm_head training
    # If lm_head should be included in training, set it as well.
    # In some models, lm_head is tied to embeddings and not included as a param.
    # So it's best to access it directly.
    step6_start = time.time()
    if train_lm_head:
        logger.info("Setting lm_head for training...")
        base = (
            lora_model.base_model if hasattr(lora_model, "base_model") else lora_model
        )
        if hasattr(base, "lm_head") and hasattr(base.lm_head, "weight"):
            base.lm_head.weight.requires_grad = True
            logger.info("lm_head weight set to trainable")
        else:
            logger.warning("lm_head or lm_head.weight not found")
    step6_time = time.time() - step6_start
    logger.info(f"lm_head handling completed: {step6_time:.2f}s")

    # Step 6: Log parameter gradients
    logger.info("Logging parameter gradients...")
    step7_start = time.time()
    log_param_gradients(lora_model, logger)
    step7_time = time.time() - step7_start
    logger.info(f"Parameter gradient logging completed: {step7_time:.2f}s")

    total_time = time.time() - overall_start
    logger.info(
        f"create_lora_model total time: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )
    logger.info(
        f"Breakdown: adapter_insert={step2_time:.1f}s, param_count={step3_time:.1f}s, "
        f"freeze={step4_time:.1f}s, unfreeze={step5_time:.1f}s, "
        f"lm_head={step6_time:.1f}s, logging={step7_time:.1f}s"
    )

    return lora_model


def unwrap_model(self):
    model = self.model

    if hasattr(model, "unwrap_model"):
        return model.unwrap_model()
    else:
        return filter_checkpoint(model, model.state_dict())


def filter_checkpoint(model, state_dict):
    # Remove the layers without gradients
    saved_params = {}

    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            logger.info(f"Saving parameter {name}")
            saved_params[name] = state_dict[name]

    return saved_params


def add_methods(model):
    model.unwrap_model = unwrap_model.__get__(model)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def log_param_gradients(model, logger=logging.getLogger(__name__)):
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    logger.info(
        f"Parameter summary: {trainable_count:,} trainable out of {total_count:,} total"
    )
