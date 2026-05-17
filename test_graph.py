from src.agent.graph import app

result = app.invoke({
    "company": "Tesla"
})

print("\n===========RESULTS===========\n")

for item in result["sentiments"]:
    print(item)

print("\nOverall Sentiment Score:")
print(result["overall_sentiment"])