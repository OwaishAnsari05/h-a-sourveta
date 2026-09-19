from typing import Any
from agents.state import AgentState
from generation.rag import extract_financial_answer

def calculator_node(state: AgentState) -> dict[str,Any]:
    query=state["query"]
    results=state.get("results",[])
    if not results:
        return {"answer":"","error":"Missing calculation evidence."}
    answer=extract_financial_answer(query,results)
    if answer is None:
        return {"answer":"","error":"Unable to calculate an answer from retrieved evidence."}
    return {"answer":answer,"error":None}