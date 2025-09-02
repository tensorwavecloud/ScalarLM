from pydantic import BaseModel
from typing import Optional


class Config(BaseModel):
    api_url: str = "http://localhost:8000"

    #model: str = "masint/tiny-random-qwen2-vl"
    model: str = "masint/tiny-random-llama"
    #model: str = "Snowflake/Arctic-Text2SQL-R1-7B"
    #model: str = "Qwen/Qwen2-7B-Instruct"
    #model: str = "Qwen/Qwen2-VL-7B-Instruct"


    # 10GB using 1024 for KB, 1024 for MB, 1024 for GB
    max_upload_file_size: int = 1024 * 1024 * 1024 * 10

    train_job_entrypoint: str = "/app/cray/scripts/train_job_entrypoint.sh"
    training_job_directory: str = "/app/cray/jobs"

    max_train_time: int = 15 * 60
    extra_training_seconds: int = 300  # 5 minutes buffer before slurm kills the job

    slurm_wait_time: int = 30 # seconds
    node_info_time_limit: int = 3600 # seconds

    megatron_refresh_period: int = 30 # seconds

    vllm_api_url: str = "http://localhost:8001"
    
    # vLLM Engine Configuration
    vllm_use_http: bool = True  # Use HTTP API (True) or direct engine calls (False)
    vllm_http_timeout: float = 30.0  # HTTP timeout in seconds
    
    # Direct engine configuration (when vllm_use_http=False)
    enable_lora: bool = True
    max_lora_rank: int = 16
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    trust_remote_code: bool = False
    enforce_eager: bool = False
    max_seq_len_to_capture: int = 8192
    max_logprobs: int = 20
    disable_sliding_window: bool = False

    generate_batch_size: int = 1024

    response_timeout: int = 60 # seconds
    inference_work_queue_timeout: int = 30 # seconds
    inference_work_queue_ack_timeout: int = 300 # seconds

    inference_work_queue_path: str = "/app/cray/inference_work_queue.sqlite"

    gpu_memory_utilization: float = 0.95
    max_model_length: int = 8192
    # 0.10.0 vllm issue: https://github.com/vllm-project/vllm/issues/21615
    limit_mm_per_prompt:str = '{"image":2}'
    
    dtype: str = "auto"

    max_log_length: int = 100

    server_list: str = "all"

    tokenformer_r: int = 32
    tokenformer_num_heads: int = 4

    tokenformer_cache_capacity: int = 2

    hf_token: str = ""

    hf_encrypted_token: bytes = b"gAAAAABoZ4CYsnzw-l4vEnBm_4zSfSinpxYoRmXmLYigjOP8q3e8-ZfWRViszKcSN_P5krZgur8NxwyYW_hNimIRqfeKgMNZThI8wB9zedsj9AJ0nmRfZbDeTISFnlgetSPcGs3-oBxQ"
    encryption_key: bytes = b"JAJOZunNSRFeXWXWVVVJfiKSzdzFMw0yFn8_JK50h60="

