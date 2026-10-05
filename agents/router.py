from agents.state import AgentState
import re

SUPPORTED_ROUTES={"retrieval","calculator"}

def _is_calculation_query(query:str)->bool:
    normalized=query.lower().strip()
    if re.search(r"\b(calculate|compute|solve|evaluate)\b",normalized):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent)\b",normalized):
        return True
    if re.search(r"\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?",normalized):
        return True
    if re.search(r"\b(?:add|subtract|multiply|divide)\b.*\d",normalized):
        return True
    if re.search(r"\bdifference between\b.*\d.*\d",normalized):
        return True
    return False

def route_query(state:AgentState)->str:
    query=state.get("query","")
    intent=state.get("intent","")
    if intent in {"calculation","calculator"} or _is_calculation_query(query):
        return "calculator"
    return "retrieval"

def router_node(state:AgentState)->dict:
    return {"route":route_query(state)}