from src.agent.graph import app

result = app.invoke({
    "company": "Tesla"
})

print(result["report"])