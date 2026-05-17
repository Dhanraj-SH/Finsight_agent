import torch
import os
from tavily import TavilyClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading env variables
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
client = TavilyClient(api_key = tavily_api_key)
llm = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = "llama-3.1-8b-instant"
    )

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
    2: "positive"
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
            "snippet": item["content"][:300]
        })

    return {"headlines" : headlines}

#Sentiment node
def sentiment_node(state):
    headlines = state["headlines"]
    sentiment_results = []
    confidence_scores = []
    
    for item in headlines:
        text = item["headline"] + ". " + item["snippet"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim = 1)
            confidence, prediction = torch.max(probabilities, dim = 1)

        sentiment = ib2label[prediction.item()]
        confidence_score = confidence.item()

        if sentiment == "positive":
            confidence_scores.append(confidence_score)

        elif sentiment == "negative":
            confidence_scores.append(-confidence_score)

        else:
            confidence_scores.append(0)
        
        sentiment_results.append({
            "headline": item["headline"],
            "snippet": item["snippet"],
            "sentiment": sentiment,
            "confidence": round(confidence_score,4)
        })

    overall_sentiment = sum(confidence_scores)/len(confidence_scores)

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for item in sentiment_results:

        if item["sentiment"] == "positive":
            positive_count += 1

        elif item["sentiment"] == "negative":
            negative_count += 1

        else:
            neutral_count += 1


    return{
        "sentiments": sentiment_results,
        "overall_sentiment": round(
            overall_sentiment,
            4
        ),
        "summary": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count
        }
    }

#Summarise_node
def summarise_node(state):
    company = state["company"]
    sentiments = state["sentiments"]
    overall_sentiment = state["overall_sentiment"]
    headlines_text = ""

    for item in sentiments:
        headlines_text += f"""
        Headline: {item['headline']}
        Sentiment: {item['sentiment']}
        Confidence: {item['confidence']}
        """

    prompt = f"""
    You are a financial analyst AI.

    Company: {company}

    Overall Sentiment Score:
    {overall_sentiment}

    Headlines and Sentiments:
    {headlines_text}

    Generate:

    1. Executive Summary (3 concise sentences)

    2. Key Risks (bullet points)

    3. Overall Outlook:
       Bullish / Bearish / Neutral
       with short reasoning.
    """

    response = llm.invoke([HumanMessage(content = prompt)])
    content = response.content

    return{
        "executive_summary": content
        }

#Report node
def report_node(state):
    company = state["company"]
    overall_sentiment = state["overall_sentiment"]
    summary = state["summary"]
    executive_summary = state["executive_summary"]
    report = f"""

    # Financial Sentiment Report

    ## Company
    {company}

    ## Overall Sentiment Score
    {overall_sentiment}

    
    ## News Summary
    Positive News: {summary['positive']}
    Negative News: {summary['negative']}
    Neutral News: {summary['neutral']}
    
    ## Executive Summary
    ---
    {executive_summary}
    ---

    ## Headlines Analysed    
    """
    for item in state["sentiments"]:
        report += f"""
        - {item['headline']}
        - Sentiment: {item['sentiment']}
        - Confidence: {item['confidence']}
        """

    return {
        "report": report    
    }