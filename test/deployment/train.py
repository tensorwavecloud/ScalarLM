import scalarlm

scalarlm.api_url = "http://localhost:8000"
# scalarlm.api_url = "https://tensorwave.cray-lm.com"
# scalarlm.api_url = "https://llama8btensorwave.cray-lm.com"
# scalarlm.api_url = "https://vultr.sscalarlm.com"
# scalarlm.api_url = "https://meta-llama--llama-3-2-3b-instruct.cray-lm.com"
# scalarlm.api_url = "https://greg1232--cray-cpu-llama-3-2-1b-instruct-fastapi-app.modal.run"
# scalarlm.api_url = "https://greg1232--cray-nvidia-llama-3-2-3b-instruct-fastapi-app.modal.run"


def get_dataset():
    dataset = []

    count = 1

    for i in range(count):
        dataset.append({"input": f"What is {i} + {i}?", "output": str(i + i)})

    return dataset


llm = scalarlm.SupermassiveIntelligence()

dataset = get_dataset()

status = llm.train(
    dataset,
    train_args={"max_steps": 5, "learning_rate": 1e-2, "gpus": 1, "max_gpus": 1},
)

print(status)
