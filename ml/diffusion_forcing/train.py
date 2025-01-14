import masint
from datasets import load_dataset

def get_dataset():
    tiny_stories = load_dataset("roneneldan/TinyStories")["train"][:5]

    dataset = []

    for item in tiny_stories["text"]:
        begin = item[:len(item) // 2]
        end = item[len(item) // 2:]
        dataset.append(
            {"input": begin, "output": end}
        )

    return dataset


llm = masint.SupermassiveIntelligence()

dataset = get_dataset()

status = llm.train(dataset, train_args={"max_steps": 100, "learning_rate": 3e-3})

print(status)

