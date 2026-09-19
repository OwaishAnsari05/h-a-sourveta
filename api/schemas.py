from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    document_id: str|None = None
    conversation_id: str|None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str
    document_id: str|None = None
    conversation_id: str|None = None
    source_count: int

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    page_count: int
    chunk_count: int

class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    count: int

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str