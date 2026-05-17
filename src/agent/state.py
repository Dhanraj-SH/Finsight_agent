from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    company: str
    headlines: List[Dict]
    sentiments: List[Dict]
    overall_sentiment: float