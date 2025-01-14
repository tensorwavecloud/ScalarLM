# Inference


## OpenAI Compatible Server

```console
$ curl http://localhost:8000/v1/chat/completions \
$     -H "Content-Type: application/json" \
$     -d '{
$         "model": "meta-llama/Llama-3.3-70B-Instruct",
$         "messages": [
$             {"role": "system", "content": "You are a helpful assistant."},
$             {"role": "user", "content": "Who won the world series in 2020?"}
$         ]
$     }'
```

## Using the Python client

You can also use the Python client to interact with the Cray server.

```python

import masint

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

