import os
import uuid
from fastapi import APIRouter,BackgroundTasks,File,HTTPException,Request,UploadFile
from api.schemas import DocumentListResponse,DocumentResponse,DocumentUploadResponse
from api.services.documents import get_document,list_documents as get_documents,find_document_by_hash,delete_document_files
from api.services.hashing import sha256_bytes
from api.services.security import sanitize_filename,validate_pdf_bytes
from api.services.rate_limit import allow_request
from api.services.processing import process_document
from api.services.status import create_status,get_status,list_statuses

DOCUMENTS_DIR="data/documents"
router=APIRouter(prefix="/documents",tags=["Documents"])

@router.get("/",response_model=DocumentListResponse)
def list_documents():
    documents=get_documents()
    return {"documents":documents,"count":len(documents)}

@router.get("/status")
def document_statuses():
    statuses=list_statuses()
    return {"documents":statuses,"count":len(statuses)}

@router.get("/{document_id}",response_model=DocumentResponse)
def get_document_by_id(document_id:str):
    document=get_document(document_id)
    if document: return document
    status=get_status(document_id)
    if status: return status
    raise HTTPException(status_code=404,detail="Document not found.")

@router.post("/upload",response_model=DocumentUploadResponse)
async def upload_document(request:Request,background_tasks:BackgroundTasks,file:UploadFile=File(...)):
    if not allow_request(request.client.host if request.client else "unknown"):
        raise HTTPException(status_code=429,detail="Too many requests. Please try again later.")
    if not file.filename: raise HTTPException(status_code=400,detail="Filename is required.")
    filename=sanitize_filename(file.filename)
    if not filename.lower().endswith(".pdf"): raise HTTPException(status_code=400,detail="Only PDF files are supported.")
    contents=await file.read()
    try:
        validate_pdf_bytes(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400,detail=str(exc))
    file_hash=sha256_bytes(contents)
    existing=find_document_by_hash(file_hash)
    if existing:
        return {"document_id":existing["document_id"],"filename":existing.get("filename") or filename,"status":existing.get("status","processing"),"duplicate":True,"message":"This document is already in the workspace. Existing document reused; no duplicate ingestion started."}
    os.makedirs(DOCUMENTS_DIR,exist_ok=True)
    document_id=uuid.uuid4().hex
    saved_filename=f"{document_id}_{filename}"
    pdf_path=os.path.join(DOCUMENTS_DIR,saved_filename)
    try:
        with open(pdf_path,"wb") as f: f.write(contents)
        create_status(document_id,filename,sha256=file_hash,size_bytes=len(contents),storage_filename=saved_filename)
        background_tasks.add_task(process_document,pdf_path,document_id,filename)
        return {"document_id":document_id,"filename":filename,"status":"processing","duplicate":False,"message":"Document accepted for processing."}
    except Exception as exc:
        if os.path.exists(pdf_path): os.remove(pdf_path)
        raise HTTPException(status_code=500,detail=f"Upload failed: {exc}")

@router.post("/{document_id}/retry",response_model=DocumentResponse)
def retry_document(document_id:str,background_tasks:BackgroundTasks,request:Request):
    if not allow_request(request.client.host if request.client else "unknown"):
        raise HTTPException(status_code=429,detail="Too many requests. Please try again later.")
    status=get_status(document_id)
    if not status: raise HTTPException(status_code=404,detail="Document not found.")
    if status.get("status") not in {"failed"}: raise HTTPException(status_code=409,detail="Only failed documents can be retried.")
    storage_filename=status.get("storage_filename")
    pdf_path=os.path.join(DOCUMENTS_DIR,storage_filename or "")
    if not storage_filename or not os.path.exists(pdf_path): raise HTTPException(status_code=404,detail="Stored PDF not found.")
    create_status(document_id,status.get("filename",storage_filename),sha256=status.get("sha256"),size_bytes=status.get("size_bytes",0),storage_filename=storage_filename)
    background_tasks.add_task(process_document,pdf_path,document_id,status.get("filename",storage_filename))
    return get_status(document_id)

@router.delete("/{document_id}")
def delete_document(document_id:str,request:Request):
    if not allow_request(request.client.host if request.client else "unknown"):
        raise HTTPException(status_code=429,detail="Too many requests. Please try again later.")
    document=get_document(document_id)
    status=get_status(document_id)
    if not document and not status: raise HTTPException(status_code=404,detail="Document not found.")
    try:
        from vectorstore.chroma_store import get_chroma_client,CHROMA_PATH,COLLECTION_NAME
        client=get_chroma_client(CHROMA_PATH)
        try:
            collection=client.get_collection(name=COLLECTION_NAME)
            collection.delete(where={"document_id":str(document_id)})
        except Exception: pass
        if not delete_document_files(document_id): raise HTTPException(status_code=404,detail="Document not found.")
        return {"document_id":document_id,"status":"deleted"}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500,detail=f"Delete failed: {exc}")
