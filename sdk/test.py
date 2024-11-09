import masint
import asyncio

def get_dataset():
    dataset = []

    count = 1

    for i in range(count):
        dataset.append(
            {"input": f"What is {i} + {i}", "output": "The answer is " + str(i + i)}
        )

    return dataset

llm = masint.AsyncSupermassiveIntelligence()

dataset = get_dataset()

for i in range(10):
    status = asyncio.run(llm.train(dataset, train_args={"max_steps": 1000 + i}))

print(status)

