import masint


#masint.api_url = "https://app.smasint.com"
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
    #model_name="c54e2b2f3ad24e8adc45b97648f8fbaf0b46813718f01feb44f0079f9ecc9b99"
)

print(results)
