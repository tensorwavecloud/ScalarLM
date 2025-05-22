import scalarlm

scalarlm.api_url = "http://localhost:8000"

def get_dataset():
    dataset = []

    count = 1

    for i in range(count):
        dataset.append({"input": f"What is {i} + {i}?", "output": str(i + i)})

    return dataset


llm = scalarlm.SupermassiveIntelligence(api_url=scalarlm.api_url)

dataset = get_dataset()

status = llm.train(
    dataset,
    train_args={"max_steps": 10, "learning_rate": 1e-4, "gpus": 1,
            "max_token_block_size": 4096,
            "steps_per_checkpoint": 10000},
)

print(status)
