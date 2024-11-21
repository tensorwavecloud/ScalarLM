import masint

masint.api_url = (
    "https://greg1232--deployment-modal-examples-cray-cray-py-fas-f8ed38-dev.modal.run"
)


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
