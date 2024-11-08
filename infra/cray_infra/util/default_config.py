from pydantic import BaseModel

class Config(BaseModel):
    api_url: str = "http://localhost:8000"



