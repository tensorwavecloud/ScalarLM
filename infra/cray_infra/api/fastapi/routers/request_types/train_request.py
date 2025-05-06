from pydantic import BaseModel

from typing import Optional


class TrainResponse(BaseModel):
    job_status: dict
    job_config: dict
    deployed: Optional[bool] = False
