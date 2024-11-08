import os

import modal

app = modal.App()

openai_token_key = "MODAL_OPENAI_API_KEY"
hf_token_key = "HF_TOKEN"

@app.function(secrets=[modal.Secret.from_name(hf_token_key)])
def hf_token():
    print(os.environ.get(hf_token_key))

@app.function(secrets=[modal.Secret.from_name(openai_token_key)])
def openai_token():
    print(os.environ.get(openai_token_key))
