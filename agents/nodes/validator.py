from typing import Any
from agents.state import AgentState
from generation.rag import validate_answer_grounding,extract_answer_numbers,extract_numbers

def _normalize_number(value:str)->str:
    value=str(value or "").strip().lower()
    value=value.replace("₹","").replace("%","").replace(",","")
    for unit in ("lakhs","lakh","crores","crore","million","billion"):
        value=value.replace(unit,"")
    return value.strip()

def validator_node(state:AgentState)->dict[str,Any]:
    answer=state.get("answer","")
    results=state.get("results",[])
    route=state.get("route","retrieval")
    if not answer.strip():
        return {"validated":False,"validation_errors":["Missing answer."]}
    if not results:
        return {"validated":False,"validation_errors":["Missing grounding evidence."]}
    if route=="calculator":
        answer_numbers=extract_answer_numbers(answer)
        context_numbers=set()
        for result in results:
            context_numbers.update(extract_numbers(result.get("document","")))
        source_numbers=[n for n in answer_numbers if _normalize_number(n) in {_normalize_number(x) for x in context_numbers}]
        if len(source_numbers)<2:
            return {"validated":False,"validation_errors":["Calculator answer lacks sufficient source numeric evidence."]}
        return {"validated":True,"validation_errors":[]}
    valid=validate_answer_grounding(answer,results)
    if valid:
        return {"validated":True,"validation_errors":[]}
    answer_numbers=extract_answer_numbers(answer)
    context_numbers=set()
    for result in results:
        context_numbers.update(extract_numbers(result.get("document","")))
    normalized_context={_normalize_number(x) for x in context_numbers}
    if answer_numbers and all(_normalize_number(n) in normalized_context for n in answer_numbers):
        return {"validated":True,"validation_errors":[]}
    return {"validated":False,"validation_errors":["Answer grounding validation failed."]}