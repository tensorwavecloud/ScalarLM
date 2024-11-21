import masint

masint.api_url = "https://greg1232--deployment-modal-examples-cray-cray-py-fas-f8ed38-dev.modal.run"

def get_dataset():
    dataset = []

    count = 1

    for i in range(count):
        dataset.append(
            {"input": f"What is {i} + {i}", "output": "The answer is " + str(i + i)}
        )

    return dataset

llm = masint.SupermassiveIntelligence()

dataset = get_dataset()

status = llm.train(dataset, train_args={"max_steps": 100})

print(status)

