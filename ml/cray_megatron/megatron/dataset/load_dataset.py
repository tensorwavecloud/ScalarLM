from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.data_parallelism import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

import datasets
import jsonlines

import logging

logger = logging.getLogger(__name__)


def load_dataset(model, tokenizer, epoch):
    hf_dataset = datasets.IterableDataset.from_generator(
        make_dataset_generator(),
        features=datasets.Features(
            {
                "input": datasets.Value(dtype="string"),
                "output": datasets.Value(dtype="string"),
            }
        ),
    )
    shuffled_dataset = hf_dataset.shuffle(seed=42 + epoch, buffer_size=256)
    split_dataset = split_dataset_by_node(shuffled_dataset)

    tokenized_dataset = split_dataset.map(
        get_tokenize_function(model, tokenizer),
        batched=True,
        remove_columns=[
            "input",
            "output",
        ],
    )

    packed_dataset = tokenized_dataset.map(
        get_pack_function(model),
        batched=True,
    )

    torch_dataset = packed_dataset.with_format("torch")

    return torch_dataset


def make_dataset_generator():
    def read_dataset():
        dataset_path = get_dataset_path()
        with open(dataset_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            for obj in reader:
                yield obj

    return read_dataset


def get_dataset_path():
    job_config = get_job_config()
    return job_config["training_data_path"]


def split_dataset_by_node(dataset):
    data_parallel_rank = get_data_parallel_rank()
    data_parallel_world_size = get_data_parallel_world_size()

    num_shards = data_parallel_world_size
    index = data_parallel_rank

    filtered_dataset = dataset.filter(
        lambda example, idx: idx % data_parallel_world_size == data_parallel_rank,
        with_indices=True,
    )

    return filtered_dataset


def get_tokenize_function(model, tokenizer):

    def tokenize(dataset):
        text = [
            input_text + output_text
            for input_text, output_text in zip(dataset["input"], dataset["output"])
        ]

        tokens = tokenizer(text)

        tokens = add_eos_token(tokens, model, tokenizer)

        # Get the length of the input sequence in tokens
        input_text_lengths = [
            len(tokenizer(input_text)["input_ids"]) for input_text in dataset["input"]
        ]

        # labels are -100 for the input_text and input_ids for the output_text
        tokens["labels"] = [
            [-100] * input_text_length + input_ids[input_text_length:]
            for input_text_length, input_ids in zip(
                input_text_lengths, tokens["input_ids"]
            )
        ]

        return tokens

    return tokenize


def add_eos_token(tokens, model, tokenizer):
    # add stop token to the end of the sequence
    if model.generation_config is None:
        eos_token = tokenizer.eos_token_id
    elif model.generation_config.eos_token_id is None:
        eos_token = tokenizer.eos_token_id
    else:
        if isinstance(model.generation_config.eos_token_id, list):
            eos_token = model.generation_config.eos_token_id[-1]
        else:
            eos_token = model.generation_config.eos_token_id

    tokens["input_ids"] = [input_ids + [eos_token] for input_ids in tokens["input_ids"]]
    tokens["attention_mask"] = [
        attention_mask + [1] for attention_mask in tokens["attention_mask"]
    ]

    return tokens


def get_pack_function(model):
    job_config = get_job_config()

    block_size = min(
        model.config.max_position_embeddings, job_config["max_token_block_size"]
    )

    def pack(dataset):
        # Concatenate all texts.
        concatenated_dataset = {k: sum(dataset[k], []) for k in dataset.keys()}
        total_length = len(concatenated_dataset[list(dataset.keys())[0]])
        # We drop the small remainder, we could add padding instead, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_dataset.items()
        }

        return result

    return pack
