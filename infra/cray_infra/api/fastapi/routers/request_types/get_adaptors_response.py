from pydantic import BaseModel

class GetAdaptorsResponse(BaseModel):
    new_adaptors: list[str]

