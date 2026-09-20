from api.services.memory import get_history,add_exchange

def agent_rag_service(query:str,document_id:str|None=None,conversation_id:str|None=None):
    from generation.rag import serialize_sources
    from agents.graph import agent_graph

    history=get_history(conversation_id)

    state=agent_graph.invoke({
        "query":query,
        "document_id":document_id,
        "conversation_id":conversation_id,
        "chat_history":history,
    })

    results=state.get("results",[])
    answer=state.get("answer","")

    if conversation_id and answer:
        add_exchange(conversation_id,query,answer)

    return {
        "answer":answer,
        "sources":serialize_sources(results),
        "query":query.strip(),
        "document_id":document_id,
        "conversation_id":conversation_id,
        "source_count":len(results),
    }

def get_rag_service():
    return agent_rag_service