from pydantic import BaseModel


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str
    dataset_id: str
    job_directory: str
    model_name: str
