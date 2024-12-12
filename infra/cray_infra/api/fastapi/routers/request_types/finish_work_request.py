from pydantic import BaseModel

from typing import Optional


class FinishWorkRequest(BaseModel):
    request_id: int
    response: str


class FinishWorkRequests(BaseModel):
    requests: list[FinishWorkRequest]
