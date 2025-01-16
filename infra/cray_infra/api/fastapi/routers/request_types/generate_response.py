from pydantic import BaseModel

from typing import Optional

class Result(BaseModel):
    request_id: int
    response: Optional[str] = None
    error: Optional[str] = None

class GenerateResponse(BaseModel):
    results: list[Result]
