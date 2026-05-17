from langgraph.graph import StateGraph
from src.agent.state import AgentState
from src.agent.nodes import search_node,sentiment_node, report_node, summarise_node

#Creating a graph
graph = StateGraph(AgentState)

#Add nodes
graph.add_node("search_node", search_node)
graph.add_node("sentiment_node", sentiment_node)
graph.add_node("summarise_node", summarise_node)
graph.add_node("report_node", report_node)

#Defining edges of the graph
graph.set_entry_point("search_node")
graph.add_edge("search_node", "sentiment_node")
graph.add_edge("sentiment_node", "summarise_node")
graph.add_edge("summarise_node", "report_node")

app = graph.compile()