import masint

masint.api_url = "http://localhost:8000"
# masint.api_url = "https://llama8b.cray-lm.com"
# masint.api_url = "https://greg1232--cray-cpu-llama-3-2-1b-instruct-fastapi-app.modal.run"
# masint.api_url = "https://greg1232--cray-nvidia-llama-3-2-3b-instruct-fastapi-app.modal.run"

llm = masint.SupermassiveIntelligence()

results = llm.health()

print(results)
