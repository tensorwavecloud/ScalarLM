from pydantic import BaseModel


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str
    dataset_id: str
    job_directory: str
    model_name: str


class TrainJobStatusResponse(BaseModel):
    job_status: dict
    job_config: dict
    model_name: str
