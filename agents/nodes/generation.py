from typing import Any
from agents.state import AgentState
from generation.rag import generate_answer,REFUSAL

def generation_node(state:AgentState)->dict[str,Any]:
    query=state["query"]
    context=state.get("context","")
    results=state.get("results",[])
    if not results:
        return {"answer":REFUSAL,"error":"Missing retrieval evidence."}
    answer=generate_answer(query,(context,results))
    return {"answer":answer,"error":None if answer!=REFUSAL else "Generation refused due to insufficient grounding."}