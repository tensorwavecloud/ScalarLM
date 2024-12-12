from pydantic import BaseModel

from typing import Optional

class GetResultsRequest(BaseModel):
    request_ids: list[int]

