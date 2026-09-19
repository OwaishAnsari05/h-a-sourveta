import os
import uuid
from fastapi import APIRouter,BackgroundTasks,File,HTTPException,UploadFile
from api.schemas import DocumentListResponse,DocumentResponse,DocumentUploadResponse
from api.services.documents import get_document,list_documents as get_documents
from api.services.processing import process_document
from api.services.status import create_status,get_status,list_statuses

DOCUMENTS_DIR = "data/documents"
router = APIRouter(prefix="/documents",tags=["Documents"])

@router.get("/",response_model=DocumentListResponse)
def list_documents():
    documents = get_documents()
    return {"documents":documents,"count":len(documents)}

@router.get("/status")
def document_statuses():
    statuses = list_statuses()
    return {"documents":statuses,"count":len(statuses)}

@router.get("/{document_id}",response_model=DocumentResponse)
def get_document_by_id(document_id: str):
    document = get_document(document_id)
    if document:
        return document
    status = get_status(document_id)
    if status:
        return status
    raise HTTPException(status_code=404,detail="Document not found.")

@router.post("/upload",response_model=DocumentUploadResponse)
async def upload_document(background_tasks: BackgroundTasks,file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400,detail="Filename is required.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,detail="Only PDF files are supported.")

    os.makedirs(DOCUMENTS_DIR,exist_ok=True)

    document_id = uuid.uuid4().hex
    filename = os.path.basename(file.filename)
    saved_filename = f"{document_id}_{filename}"
    pdf_path = os.path.join(DOCUMENTS_DIR,saved_filename)

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400,detail="Uploaded file is empty.")

        with open(pdf_path,"wb") as f:
            f.write(contents)

        create_status(document_id,filename)

        background_tasks.add_task(
            process_document,
            pdf_path,
            document_id,
            filename,
        )

        return {
            "document_id":document_id,
            "filename":filename,
            "status":"processing",
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(status_code=500,detail=f"Upload failed: {str(e)}")