from typing import Any
from agents.state import AgentState
from generation.rag import ask_question

def retrieval_node(state: AgentState) -> dict[str, Any]:
    query=state["query"]
    document_id=state.get("document_id")
    if document_id:
        answer,results=ask_question(query,document_id=document_id)
    else:
        answer,results=ask_question(query)
    context="\n\n".join(str(result.get("document","")) for result in results)
    return {"answer":answer,"context":context,"results":results}