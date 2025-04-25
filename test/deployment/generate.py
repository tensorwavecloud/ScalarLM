import scalarlm

# scalarlm.api_url = "http://localhost:8000"
# scalarlm.api_url = "https://tensorwave.cray-lm.com"
scalarlm.api_url = "https://llama8btensorwave.cray-lm.com"
# scalarlm.api_url = "https://vultr.sscalarlm.com"
# scalarlm.api_url = "https://meta-llama--llama-3-2-3b-instruct.cray-lm.com"


def get_dataset(count):
    dataset = []

    for i in range(count):
        dataset.append(f"What is {i} + {i}?")

    return dataset


llm = scalarlm.SupermassiveIntelligence()

dataset = get_dataset(count=1)

results = llm.generate(
    prompts=dataset,
    max_tokens=200,
    # generate with default model
    # model_name="2560805ddf87f7a1340be14e4b3dd38f7289a43de79410b3ac09944438842bfd"
)

print(results)
