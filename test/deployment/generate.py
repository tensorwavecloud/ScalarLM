import scalarlm

scalarlm.api_url = "http://localhost:8000"

def get_dataset(count):
    dataset = []

    for i in range(count):
        dataset.append(f"What is {i} + {i}?")

    return dataset


llm = scalarlm.SupermassiveIntelligence()

dataset = ["Hello, how are you?", "What is 10 + 15?", "Tell me a story"]

results = llm.generate(
    prompts=dataset,
    max_tokens=200,
)

print(results)
