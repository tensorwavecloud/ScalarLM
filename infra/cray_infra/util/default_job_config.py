from pydantic import BaseModel

from typing import Optional


class LoraConfig(BaseModel):
    r: int = 32
    target_modules: str = "all-linear"
    use_rslora: bool = True
    modules_to_save: list = ["lm_head"]


class DiffusionForcingModelConfig(BaseModel):
    num_hidden_layers: int = 2
    num_diffusion_iterations: int = 3
    diffusion_step_size: int = 2
    hidden_size: int = 128
    num_attention_heads: int = 4
    attention_dropout: float = 0.1


class JobConfig(BaseModel):

    job_directory: str
    training_data_path: str
    dataset_hash: str

    #llm_name: str = "masint/tiny-random-llama"
    llm_name: str = "meta-llama/Llama-3.2-1B-Instruct"

    # Training
    max_steps: int = 100
    learning_rate: float = 3e-3
    batch_size: int = 1

    max_token_block_size: int = 16777216 # 16 mega tokens

    # Checkpointing
    steps_per_checkpoint: int = 100
    max_checkpoints_to_keep: int = 3

    gpus: int = 1
    nodes: int = 1

    lora_config: Optional[LoraConfig] = LoraConfig()
    diffusion_forcing_config: Optional[DiffusionForcingModelConfig] = (
        DiffusionForcingModelConfig()
    )

    # 4 hours in seconds
    timeout: int = 4 * 60 * 60

    training_history_length: int = 1024

