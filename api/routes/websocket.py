import asyncio
from fastapi import APIRouter,WebSocket,WebSocketDisconnect
from api.services.memory import add_exchange,get_history

router=APIRouter(tags=["WebSocket"])
GROUNDING_REFUSAL="I could not find sufficiently relevant information in the document to answer this question."

@router.websocket("/ws/chat")
async def websocket_chat(websocket:WebSocket):
    await websocket.accept()

    from agents.nodes.query_resolver import resolve_query
    from generation.rag import (
        build_context,
        format_citations,
        generate_answer_stream,
        hybrid_retrieve,
        protect_exact_matches,
        rerank_candidates,
        serialize_sources,
    )

    try:
        while True:
            data=await websocket.receive_json()
            query=str(data.get("query","")).strip()
            document_id=data.get("document_id")
            conversation_id=data.get("conversation_id")

            if not query:
                await websocket.send_json({"type":"error","message":"Query cannot be empty."})
                continue

            await websocket.send_json({"type":"status","message":"Processing query..."})

            try:
                history=get_history(conversation_id)
                resolved_query=resolve_query(query,history)

                await websocket.send_json({"type":"status","message":"Understanding conversation context..."})

                if resolved_query!=query:
                    await websocket.send_json({"type":"resolved_query","query":resolved_query})

                await websocket.send_json({"type":"status","message":"Retrieving relevant evidence..."})

                candidates=await asyncio.to_thread(
                    hybrid_retrieve,
                    resolved_query,
                    top_k=15,
                    document_id=document_id,
                )

                if not candidates:
                    answer=GROUNDING_REFUSAL
                    await websocket.send_json({"type":"answer","answer":answer})
                    await websocket.send_json({"type":"sources","sources":[]})
                    await websocket.send_json({
                        "type":"complete",
                        "query":query,
                        "resolved_query":resolved_query,
                        "document_id":document_id,
                        "conversation_id":conversation_id,
                        "source_count":0,
                    })

                    if conversation_id:
                        await asyncio.to_thread(add_exchange,conversation_id,query,answer)

                    continue

                await websocket.send_json({"type":"status","message":"Reranking evidence..."})

                reranked=await asyncio.to_thread(
                    rerank_candidates,
                    resolved_query,
                    candidates,
                    top_k=10,
                )

                protected=await asyncio.to_thread(
                    protect_exact_matches,
                    resolved_query,
                    reranked,
                    candidates=candidates,
                    top_k=10,
                )

                if document_id:
                    doc_id_str=str(document_id)
                    protected=[
                        r for r in protected
                        if r.get("metadata",{}).get("document_id")==doc_id_str
                        or r.get("document_id")==doc_id_str
                    ]

                context,results=await asyncio.to_thread(
                    build_context,
                    protected,
                    candidates,
                    resolved_query,
                    5,
                )

                if document_id:
                    doc_id_str=str(document_id)
                    results=[
                        r for r in results
                        if r.get("metadata",{}).get("document_id")==doc_id_str
                        or r.get("document_id")==doc_id_str
                    ]

                await websocket.send_json({"type":"status","message":"Generating answer..."})

                answer_parts=[]

                for chunk in generate_answer_stream(
                    resolved_query,
                    (context,results),
                ):
                    answer_parts.append(chunk)
                    await websocket.send_json({"type":"token","content":chunk})
                    await asyncio.sleep(0)

                answer="".join(answer_parts).strip()
                refused=answer==GROUNDING_REFUSAL

                if answer:
                    await websocket.send_json({"type":"answer","answer":answer})

                if refused:
                    results,citations=[],""
                else:
                    citations=await asyncio.to_thread(
                        format_citations,
                        resolved_query,
                        answer,
                        results,
                    ) if answer else ""

                if citations:
                    await websocket.send_json({"type":"citations","content":citations})

                if conversation_id and answer:
                    await asyncio.to_thread(
                        add_exchange,
                        conversation_id,
                        query,
                        answer,
                    )

                await websocket.send_json({
                    "type":"sources",
                    "sources":serialize_sources(results),
                })

                await websocket.send_json({
                    "type":"complete",
                    "query":query,
                    "resolved_query":resolved_query,
                    "document_id":document_id,
                    "conversation_id":conversation_id,
                    "source_count":len(results),
                })

            except Exception as exc:
                await websocket.send_json({
                    "type":"error",
                    "message":str(exc),
                })

    except WebSocketDisconnect:
        pass