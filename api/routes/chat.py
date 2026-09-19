from fastapi import APIRouter,Depends,HTTPException
from api.dependencies import get_rag_service
from api.schemas import ChatRequest,ChatResponse

router=APIRouter(tags=["Chat"])

@router.post("/chat",response_model=ChatResponse)
def chat(request:ChatRequest,rag_service=Depends(get_rag_service)):
    query=request.query.strip()
    if not query:
        raise HTTPException(status_code=400,detail="Query cannot be empty.")
    return rag_service(query,request.document_id,request.conversation_id)