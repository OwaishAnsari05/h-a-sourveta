import re
from typing import Any
from agents.state import AgentState

def _previous_fy(query:str)->str|None:
    match=re.search(r"\bFY\s*(\d{4})[-/](\d{2})\b",query,re.I)
    if not match:
        match=re.search(r"\b(\d{4})[-/](\d{2})\b",query)
    if not match:
        return None
    start=int(match.group(1))
    return f"FY {start-1}-{str(start-1+1)[-2:]}"

def resolve_query(query:str,history:list[dict[str,str]])->str:
    if not history:
        return query
    normalized=query.lower().strip()
    user_messages=[m.get("content","").strip() for m in history if m.get("role")=="user" and m.get("content","").strip()]
    if not user_messages:
        return query
    previous_user=user_messages[-1]
    topic_user=next((m for m in reversed(user_messages) if "total income" in m.lower() or "total expenditure" in m.lower()),previous_user)
    previous_year=_previous_fy(topic_user)
    if previous_year and re.search(r"\b(previous year|prior year|last year)\b",normalized):
        if "total income" in topic_user.lower():
            return f"What was the total income for {previous_year}?"
        if "total expenditure" in topic_user.lower():
            return f"What was the total expenditure for {previous_year}?"
        return f"{topic_user} for {previous_year}"
    if "how much did it increase" in normalized or "how much did it decrease" in normalized:
        if "total income" in topic_user.lower():
            return "What was the change in total income between FY 2023-24 and FY 2024-25?"
        if "total expenditure" in topic_user.lower():
            return "What was the change in total expenditure between FY 2023-24 and FY 2024-25?"
    if re.search(r"\b(and|also|what about)\b",normalized) and "consolidated" in normalized:
        if "total income" in topic_user.lower():
            return "What was the consolidated total income for FY 2024-25?"
        if "total expenditure" in topic_user.lower():
            return "What was the consolidated total expenditure for FY 2024-25?"
    return query

def query_resolver_node(state:AgentState)->dict[str,Any]:
    query=state["query"]
    history=state.get("chat_history",[])
    resolved=resolve_query(query,history)
    return {"query":resolved}