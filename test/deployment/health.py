import scalarlm

scalarlm.api_url = "http://localhost:8000"

llm = scalarlm.SupermassiveIntelligence()

results = llm.health()

print(results)
