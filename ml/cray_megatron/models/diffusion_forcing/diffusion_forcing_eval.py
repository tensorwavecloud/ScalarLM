from cray_megatron.models.diffusion_forcing.diffusion_forcing_model_manager import (
    DiffusionForcingModelManager,
)

from datasets import load_dataset

import torch
from safetensors import safe_open

from tqdm import tqdm

import os
import gc

import logging

logger = logging.getLogger(__name__)


def diffusion_forcing_eval(model_path: str):

    with torch.no_grad():
        dataset = load_eval_dataset()

        model_info = load_model_info(model_path)

        results = evaluate_model(model_info, dataset)

    print_results(results)


def load_eval_dataset():
    dataset = load_dataset("roneneldan/TinyStories")["train"]
    return dataset[:5]


def load_model_info(model_path: str):
    os.environ["CRAY_TRAINING_JOB_CONFIG_PATH"] = os.path.join(
        model_path, "config.yaml"
    )

    manager = DiffusionForcingModelManager()
    model_info = manager.load_model()

    saved_model_path = os.path.join(model_path, "saved_model", "model.safetensors")

    model_info["model"].load_state_dict(load_safetensors(saved_model_path))

    model_info["model"].eval()

    return model_info


def load_safetensors(saved_model_path):
    tensors = {}
    with safe_open(saved_model_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    return tensors


def evaluate_model(model_info, dataset):
    output_texts = []
    average_loss = 0.0
    item_count = 0

    max_length = 128

    for item in dataset["text"]:
        if len(item) > max_length:
            item = item[:max_length]

        logger.info(f"Evaluating item: {item}")
        input_ids = model_info["tokenizer"](item, return_tensors="pt").input_ids
        loss, text, prefix = evaluate_item(model_info, input_ids)

        output_texts.append((text, prefix, item))
        average_loss += loss
        item_count += 1

    average_loss /= item_count

    return output_texts, average_loss


def evaluate_item(model_info, input_ids):
    seq_len = input_ids.shape[1]
    diffusion_iterations = model_info["model"].config.num_diffusion_iterations
    diffusion_step_size = model_info["model"].config.diffusion_step_size

    logger.debug(f"seq_len: {seq_len}")

    prefix_len = seq_len // 2

    remaining_len = seq_len - prefix_len

    diagonal_length = remaining_len + diffusion_iterations - 1

    step_count = (diagonal_length + diffusion_step_size - 1) // diffusion_step_size

    output_ids = input_ids[:, :prefix_len]

    logits = None

    for step in tqdm(range(step_count)):
        prefix_end = min(prefix_len + diffusion_step_size * (step + 1) - 1, seq_len)
        current_sequence_len = output_ids.shape[1]
        logger.debug(
            f"Step: {step}, seq_len: {current_sequence_len}, prefix_end: {prefix_end}"
        )

        output_ids = pad_output_ids(output_ids, prefix_end, model_info)

        noise = generate_noise(prefix_end, logits, output_ids, model_info)

        next_step_input_ids = output_ids[:, :prefix_end]

        result = model_info["model"](
            input_ids=next_step_input_ids,
            noise=noise,
            labels=input_ids[:, :prefix_end],
        )

        loss = result.loss
        logits = result.logits

        output_ids = sample_next_tokens(logits, output_ids, current_sequence_len)

    return (
        loss,
        model_info["tokenizer"].decode(output_ids[0]),
        model_info["tokenizer"].decode(input_ids[0, :prefix_len]),
    )


def pad_output_ids(output_ids, prefix_end, model_info):
    pad_len = prefix_end - output_ids.shape[1]

    if pad_len > 0:
        pad_token_id = output_ids[0, -1].item()

        output_ids = torch.cat(
            [
                output_ids,
                torch.full(
                    (output_ids.shape[0], pad_len),
                    fill_value=pad_token_id,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )

    return output_ids


def generate_noise(prefix_end, logits, output_ids, model_info):
    batch_size = output_ids.shape[0]
    seq_len = output_ids.shape[1]
    vocab_size = model_info["model"].config.vocab_size

    noise = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32)

    # zeros up until prefix_end
    # logits from prefix_end to logits.shape[1]
    # random noise from logits.shape[1] to noise.shape[1]
    logits_len = prefix_end

    if logits is not None:
        logits_len = logits.shape[1]
        noise[:, prefix_end:logits_len, :] = logits[:, prefix_end:logits_len, :]

    noise[:, logits_len:, :] = torch.randn_like(noise[:, logits_len:, :]) * 300.0

    return noise


def sample_next_tokens(logits, output_ids, prefix_end):
    next_tokens = torch.argmax(logits[:, prefix_end:, :], dim=-1)

    logger.info(f"Adding next tokens to output ids: {next_tokens}")

    output_ids[:, prefix_end:] = next_tokens

    return output_ids


def print_results(results):
    output_texts, average_loss = results

    print(f"Average loss: {average_loss}")

    for text, prefix, item in output_texts:
        print(f"Prefix: {prefix}")
        print(f"Generated text: {text}")
        print(f"Original text: {item}")
