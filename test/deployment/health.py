import scalarlm

llm = scalarlm.SupermassiveIntelligence(api_url="http://localhost:8000")

results = llm.health()

print(results)
