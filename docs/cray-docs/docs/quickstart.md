Let's start by submitting your first request to Craylm.

## Setup

Clone the [Craylm repository](https://github.com/cray-lm/cray-lm) and start the server.

```
git clone git@github.com:cray-lm/cray-lm.git
cd cray-lm
./cray up
```

This will bring up the Craylm development server on `localhost:8000`, which includes an OpenAI compatible API.

## Your first request

```console
curl http://localhost:8000/v1/openai/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "masint/tiny-random-llama",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

## Using the Python client

You can also use the Python client to interact with the local Craylm server.

```python

import masint

# Make sure to set the API URL to the local Craylm server
masint.api_url = "http://localhost:8000"

def get_dataset():
    dataset = []

    count = 4

    for i in range(count):
        dataset.append(f"What is {i} + {i}?")

    return dataset


llm = masint.SupermassiveIntelligence()

dataset = get_dataset()

results = llm.generate(prompts=dataset)

print(results)
```

# Loading a different model

Edit the file cray-lm/infra/cray_infra/util/default_config.py and change the `model` field to the desired model.

```python
model = "meta-llama/Llama-3.2-1B-Instruct"
```

Then restart the server.

```console
./cray up
```

# Submitting a request to the new model

```console
curl http://localhost:8000/v1/openai/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```


