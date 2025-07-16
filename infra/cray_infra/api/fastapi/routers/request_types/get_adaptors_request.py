from pydantic import BaseModel

class GetAdaptorsRequest(BaseModel):
    loaded_adaptor_count: int

