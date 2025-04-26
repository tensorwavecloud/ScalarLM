import scalarlm

# scalarlm.api_url = "http://localhost:8000"
# scalarlm.api_url = "https://tensorwave.cray-lm.com"
scalarlm.api_url = "https://llama8btensorwave.cray-lm.com"
# scalarlm.api_url = "https://vultr.sscalarlm.com"
# scalarlm.api_url = "https://meta-llama--llama-3-2-3b-instruct.cray-lm.com"
# scalarlm.api_url = "https://greg1232--cray-cpu-llama-3-2-1b-instruct-fastapi-app.modal.run"
# scalarlm.api_url = "https://greg1232--cray-nvidia-llama-3-2-3b-instruct-fastapi-app.modal.run"


def get_dataset():
    dataset = []

    count = 1

    for i in range(count):
        dataset.append({"input": f"What is {i} + {i}?", "output": str(i + i)})

    return dataset * 100


llm = scalarlm.SupermassiveIntelligence(api_url=scalarlm.api_url)

dataset = get_dataset()

status = llm.train(
    dataset,
    train_args={"max_steps": 100, "learning_rate": 1e-4, "gpus": 2,
            "max_token_block_size": 4096,
            "steps_per_checkpoint": 10000},
)

print(status)
