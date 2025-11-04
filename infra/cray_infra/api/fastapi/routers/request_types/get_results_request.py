from pydantic import BaseModel

class GetResultsRequest(BaseModel):
    request_ids: list[str]

