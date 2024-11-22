from pydantic import BaseModel

class Config(BaseModel):
    api_url: str = "http://localhost:8000"
    model: str = "masint/tiny-random-llama"
    #model: str = "meta-llama/Llama-3.2-1B"

    # 10GB using 1024 for KB, 1024 for MB, 1024 for GB
    max_upload_file_size: int = 1024 * 1024 * 1024 * 10

    data_directory: str = "/app/cray/datasets"

    train_job_entrypoint: str = "/app/cray/scripts/train_job_entrypoint.sh"
    training_job_directory: str = "/app/cray/jobs"

    max_train_time: int = 15 * 60
    extra_training_seconds: int = 300 # 5 minutes buffer before slurm kills the job

    megatron_refresh_period: int = 30 # seconds

    vllm_api_url: str = "http://localhost:8001"

    generate_batch_size: int = 4

