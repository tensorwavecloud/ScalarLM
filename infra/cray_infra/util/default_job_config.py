from pydantic import BaseModel


class LoraConfig(BaseModel):
    r: int = 32
    target_modules: str = "all-linear"
    use_rslora: bool = True
    modules_to_save: list = ["lm_head"]


class JobConfig(BaseModel):

    job_directory: str
    training_data_path: str

    llm_name: str = "masint/tiny-random-llama"

    max_steps: int = 100
    learning_rate: float = 1e-3
    steps_per_checkpoint: int = 100
    batch_size: int = 1

    gpus: int = 1
    nodes: int = 1

    lora_config: str = LoraConfig()

    # 4 hours in seconds
    timeout: int = 4 * 60 * 60

    training_history_length: int = 1024
