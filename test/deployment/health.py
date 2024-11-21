import masint

masint.api_url = (
    "https://greg1232--deployment-modal-examples-cray-cray-py-fas-f8ed38-dev.modal.run"
)

llm = masint.SupermassiveIntelligence()

results = llm.health()

print(results)

