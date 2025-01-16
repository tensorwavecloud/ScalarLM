from pydantic import BaseModel

from typing import Optional


class FinishWorkRequest(BaseModel):
    request_id: int
    response: Optional[str] = None
    error: Optional[str] = None


class FinishWorkRequests(BaseModel):
    requests: list[FinishWorkRequest]
