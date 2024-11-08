from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging

logger = logging.getLogger(__name__)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/train", response_model=TrainResponse)
def train(payload: TrainPayload = Body(...)):
    pass

app.add_exception_handler(500, internal_exception_handler)

