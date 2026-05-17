from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.models import CompanyRequest, AnalyzeResponse
from src.agent.graph import app as langgraph_app
import asyncio

#fastapi app
app = FastAPI(
    title = "Financail Senitment AI API",
    version = "1.0.0"
)

#cors
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

#Health check
@app.get("/health")

async def health_check():
    return {
        "status": "healthy",
        "model": "FinBERT + Langgraph + Groq"
    }


#Analyzing the endpoint
@app.post(
    "/analyze",
    response_model = AnalyzeResponse
)

async def analyze_company(request:CompanyRequest):
    try:
        result = await asyncio.to_thread(
            langgraph_app.invoke,
            {
                "company" : request.company
            }
        )

        return {
            "company" : request.company,
            "overall_sentiment": result["overall_sentiment"],
            "summary": result["summary"],
            "executive_summary": result["executive_summary"],
            "report": result["report"]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = str(e)
        )