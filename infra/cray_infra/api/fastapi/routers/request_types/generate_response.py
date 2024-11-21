from pydantic import BaseModel


class GenerateResponse(BaseModel):
    responses: list[str] = []
