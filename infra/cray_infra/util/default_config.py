from pydantic import BaseModel


class Config(BaseModel):
    api_url: str = "http://localhost:8000"

    model: str = "masint/tiny-random-llama"

    # 10GB using 1024 for KB, 1024 for MB, 1024 for GB
    max_upload_file_size: int = 1024 * 1024 * 1024 * 10

    train_job_entrypoint: str = "/app/cray/scripts/train_job_entrypoint.sh"
    training_job_directory: str = "/app/cray/jobs"

    max_train_time: int = 15 * 60
    extra_training_seconds: int = 300  # 5 minutes buffer before slurm kills the job

    slurm_wait_time: int = 30 # seconds

    megatron_refresh_period: int = 30 # seconds

    vllm_api_url: str = "http://localhost:8001"

    generate_batch_size: int = 1024

    response_timeout: int = 60 # seconds
    inference_work_queue_timeout: int = 30 # seconds

    inference_work_queue_path: str = "/app/cray/inference_work_queue.sqlite"

    gpu_memory_utilization: float = 0.50
    max_model_length: int = 8192
    dtype: str = "bfloat16"

    max_log_length: int = 100

    server_list: str = "all"

    tokenformer_r: int = 32
    tokenformer_num_heads: int = 4

    tokenformer_cache_capacity: int = 2

