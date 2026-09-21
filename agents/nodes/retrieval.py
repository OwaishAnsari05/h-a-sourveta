from typing import Any
from agents.state import AgentState
from generation.rag import ask_question

def retrieval_node(state: AgentState) -> dict[str, Any]:
    query=state["query"]
    document_id=state.get("document_id")

    print(" RETRIEVAL NODE START ",flush=True)
    print(f"QUERY: {query}",flush=True)
    print(f"DOCUMENT_ID: {document_id}",flush=True)
    print(" CALLING ASK_QUESTION ",flush=True)

    try:
        if document_id:
            answer,results=ask_question(query,document_id=document_id)
        else:
            answer,results=ask_question(query)

        print(" ASK_QUESTION FINISHED ",flush=True)
        print(f"RESULT COUNT: {len(results)}",flush=True)

        context="\n\n".join(str(result.get("document","")) for result in results)

        print(" RETRIEVAL NODE FINISHED ",flush=True)

        return {
            "answer":answer,
            "context":context,
            "results":results,
        }

    except Exception as exc:
        print(f" RETRIEVAL NODE ERROR: {exc} ",flush=True)
        raise