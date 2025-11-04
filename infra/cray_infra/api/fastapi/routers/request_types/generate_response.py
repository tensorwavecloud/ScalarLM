from pydantic import BaseModel

from typing import Optional, Union

class Result(BaseModel):
    request_id: str
    response: Optional[Union[str, list[float]]] = None
    error: Optional[str] = None

class GenerateResponse(BaseModel):
    results: list[Result]
