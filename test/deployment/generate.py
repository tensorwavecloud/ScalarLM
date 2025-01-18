import masint


masint.api_url = "https://llama8b.cray-lm.com"
#masint.api_url = "https://greg1232--cray-cpu-llama-3-2-1b-instruct-fastapi-app.modal.run"
#masint.api_url = "https://greg1232--cray-nvidia-llama-3-2-3b-instruct-fastapi-app.modal.run"


def get_dataset():
    dataset = []

    count = 1

    for i in range(count):
        dataset.append(f"What is {i} + {i}?")

    return dataset


llm = masint.SupermassiveIntelligence()

dataset = get_dataset()

results = llm.generate(prompts=dataset,
    # generate with default model
    model_name="64192c4967586d250f4bd852e035eae2e79a392db1a22cb2b09bfa04bd44721a"
)

print(results)
