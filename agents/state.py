from typing import Any,TypedDict

class AgentState(TypedDict,total=False):
    query: str
    document_id: str|None
    conversation_id: str|None
    chat_history: list[dict[str,str]]
    intent: str
    route: str
    context: str
    results: list[dict[str,Any]]
    answer: str
    citations: str
    validated: bool
    validation_errors: list[str]
    retry_count: int
    error: str|None