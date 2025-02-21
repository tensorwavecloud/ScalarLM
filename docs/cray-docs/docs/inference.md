# Inference


## OpenAI Compatible Server

```console
curl https://meta-llama--llama-3-2-3b-instruct.cray-lm.com/v1/openai/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

## Using the Python client

You can also use the Python client to interact with the ScalarLM server.

```python

import masint

masint.api_url = "https://meta-llama--llama-3-2-3b-instruct.cray-lm.com"

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

