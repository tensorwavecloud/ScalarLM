from pydantic import BaseModel

from typing import Optional


class GetWorkResponse(BaseModel):
    prompt: str
    request_id: int
    request_type: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None


class GetWorkResponses(BaseModel):
    requests: list[GetWorkResponse]
