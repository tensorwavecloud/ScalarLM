from pydantic import BaseModel

class GetNodeCountResponse(BaseModel):
    node_count: int


