from agents.state import AgentState
from generation.rag import query_profile

def route_query(state: AgentState) -> str:
    query=state.get("query","")
    intent=state.get("intent","")
    profile=query_profile(query)
    if profile.get("comparative") is True:
        return "calculator"
    if intent in {"calculation","calculator","comparative"}:
        return "calculator"
    if intent in {"summary","summarization"}:
        return "summarizer"
    if intent in {"table","table_query"}:
        return "table_agent"
    return "retrieval"

def router_node(state: AgentState) -> dict:
    return {"route":route_query(state)}