from typing import Any
from agents.state import AgentState
from generation.rag import format_citations

def citation_node(state: AgentState) -> dict[str,Any]:
    query=state["query"]
    answer=state.get("answer","")
    results=state.get("results",[])
    if not answer or not results:
        return {"citations":""}
    citations=format_citations(query,answer,results)
    return {"citations":citations}