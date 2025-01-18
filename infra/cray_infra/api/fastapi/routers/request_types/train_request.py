from pydantic import BaseModel


class TrainResponse(BaseModel):
    job_status: dict
    job_config: dict
