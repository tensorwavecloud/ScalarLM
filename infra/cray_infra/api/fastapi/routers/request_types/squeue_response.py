from pydantic import BaseModel

from typing import Optional


class SqueueResponse(BaseModel):
    squeue_output : Optional[str] = None
    error_message : Optional[str] = None
