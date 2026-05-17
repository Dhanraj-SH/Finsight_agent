import torch
import os
from tavily import TavilyClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading env variables
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key = tavily_api_key)


#Loading model
MODEL_PATH = "./models/finbert_pytorch_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

#Label Mapping
ib2label = {
    0: "negative",
    1: "neutral",
    2: "postive"
}

#Search node
def search_node(state):
    company = state["company"]
    query = f"""
    Latest news about {company} stock market earnings revenue losses profits lawsuits business expansion
    """
    response = client.search(
        query = query,
        search_depth = "advanced",
        max_results = 10
    )

    headlines = []

    for item in response["results"]:

        headlines.append({
            "headline": item["title"],
            "snippet": item["content"]
        })

    return {"headlines" : headlines}

#Sentiment node
def sentiment_node(state):
    headlines = state["headlines"]
    sentiment_results = []
    confidence_scores = []
    
    for item in headlines:
        text = item["headline"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            turncation = True,
            padding = "max_length",
            max_length = 128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)
            probabilties = F.softmax(outputs.logits, dim = 1)
            confidence, prediction = torch.max(probabilties, dim = 1)

        sentiment = ib2label[prediction.item()]
        confidence_score = confidence.item()

        if sentiment == "positive":
            confidence_scores.append(confidence_score)

        elif sentiment == "negative":
            confidence_scores.append(-confidence_score)

        else:
            confidence_scores.append(0)
        
        sentiment_results.append({
            "headline": text,
            "sentiment": sentiment,
            "confidence": round(confidence_score,4)
        })

    overall_sentiment = sum(confidence_scores)/len(confidence_scores)

    return{
        "sentiments": sentiment_results,
        "overall_sentiment": round(
            overall_sentiment,
            4
        )
    }