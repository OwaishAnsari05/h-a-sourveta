from typing import Any
from agents.state import AgentState
from generation.rag import query_profile

def triage_node(state: AgentState) -> dict[str, Any]:
    query=state["query"]
    profile=query_profile(query)
    return {"intent": profile["type"]}