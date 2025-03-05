import masint


masint.api_url = "http://localhost:8000"
#masint.api_url = "https://meta-llama--llama-3-2-3b-instruct.cray-lm.com"
#masint.api_url = "https://greg1232--cray-cpu-llama-3-2-1b-instruct-fastapi-app.modal.run"
#masint.api_url = "https://greg1232--cray-nvidia-llama-3-2-3b-instruct-fastapi-app.modal.run"


def get_dataset(count):
    dataset = []

    for i in range(count):
        dataset.append(f"What is {i} + {i}?")

    return dataset


llm = masint.SupermassiveIntelligence()

dataset = get_dataset(count=3)

results = llm.embed(prompts=dataset,
# generate with default model
# model_name="c7c3ed39e0005e0e73145d49510c94d7b5e4f6552cd35c4a7a8b37d0b41f318e"
)

print(results)

