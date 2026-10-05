from pydantic import BaseModel

class ChatRequest(BaseModel):
    query:str
    document_id:str|None=None
    conversation_id:str|None=None

class ChatResponse(BaseModel):
    answer:str
    sources:list[dict]
    query:str
    document_id:str|None=None
    conversation_id:str|None=None
    source_count:int

class DocumentResponse(BaseModel):
    document_id:str
    filename:str
    status:str
    page_count:int
    chunk_count:int
    sha256:str|None=None
    size_bytes:int=0
    error:str|None=None
    created_at:str|None=None
    updated_at:str|None=None

class DocumentListResponse(BaseModel):
    documents:list[DocumentResponse]
    count:int

class DocumentUploadResponse(BaseModel):
    document_id:str
    filename:str
    status:str
    duplicate:bool=False
    message:str|None=None