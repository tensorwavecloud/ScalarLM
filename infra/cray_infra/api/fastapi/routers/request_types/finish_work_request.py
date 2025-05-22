from pydantic import BaseModel

from typing import Optional, Union


class FinishWorkRequest(BaseModel):
    request_id: int
    response: Optional[Union[str, list[float]]] = None
    error: Optional[str] = None
    token_count: Optional[int] = None
    flop_count: Optional[int] = None


class FinishWorkRequests(BaseModel):
    requests: list[FinishWorkRequest]
