import base64
import requests


import scalarlm

scalarlm.api_url = "http://localhost:8000"

def get_dataset(count):
    dataset = []

    for i in range(count):
        image_url = get_image(f"https://httpbin.org/image/png")
        dataset.append({"text": "What is in this image?",
                        "images": [image_url, image_url]})

    return dataset

def get_image(url):
    # Get the image and return a base64 encoded string
    response = requests.get(url)
    if response.status_code == 200:
        return "data:image/jpeg;base64," + base64.b64encode(response.content).decode('utf-8')
    else:
        raise Exception(f"Failed to fetch image from {url}")

llm = scalarlm.SupermassiveIntelligence()

dataset = get_dataset(count=1)

results = llm.generate(
    prompts=dataset,
    max_tokens=200,
)

print(results)

