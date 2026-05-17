from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    company: str
    headlines: List[Dict]
    sentiments: List[Dict]
    overall_sentiment: float
    summary: Dict
    executive_summary: str
    key_risks: str
    overall_outlook: str
    report: str