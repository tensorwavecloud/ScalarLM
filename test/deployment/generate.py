import masint


#masint.api_url = "https://greg1232--cray-cpu-llama-3-2-1b-fastapi-app.modal.run"
#masint.api_url = "https://greg1232--cray-nvidia-llama-3-2-3b-instruct-fastapi-app.modal.run"


def get_dataset():
    dataset = []

    count = 1

    for i in range(count):
        dataset.append(f"What is {i} + {i}? ")

    return dataset


llm = masint.SupermassiveIntelligence()

dataset = get_dataset()

results = llm.generate(prompts=dataset, model_name="1321e27922cdc7112b1eb9320de8e1c4c76c561151edf8ec18ad346a4d065bbd")

print(results)
