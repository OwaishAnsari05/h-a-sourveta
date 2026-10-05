from typing import Any
from agents.state import AgentState
from generation.rag import retrieve_evidence

def retrieval_node(state:AgentState)->dict[str,Any]:
    query=state["query"]
    document_id=state.get("document_id")
    print(" RETRIEVAL NODE START ",flush=True)
    print(f"QUERY: {query}",flush=True)
    print(f"DOCUMENT_ID: {document_id}",flush=True)
    try:
        context,results=retrieve_evidence(query,document_id=document_id)
        print(f"RESULT COUNT: {len(results)}",flush=True)
        return {"context":context,"results":results,"answer":"","error":None}
    except Exception as exc:
        print(f" RETRIEVAL NODE ERROR: {exc} ",flush=True)
        return {"context":"","results":[],"answer":"","error":str(exc)}