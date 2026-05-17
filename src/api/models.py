from pydantic import BaseModel

class CompanyRequest(BaseModel):
    company: str

class AnalyzeResponse(BaseModel):
    company: str
    overall_sentiment: float
    summary: dict
    executive_summary: str
    report: str