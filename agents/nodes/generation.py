from typing import Any
from agents.state import AgentState
from generation.rag import generate_answer

def generation_node(state:AgentState)->dict[str,Any]:
    query=state["query"]
    context=state.get("context","")
    results=state.get("results",[])
    answer=generate_answer(query,(context,results))
    return {"answer":answer,"error":None}