import json
import os
import re
import pymupdf
from api.services.status import get_status,find_by_sha256,delete_status

DOCUMENTS_DIR="data/documents"
CHUNKS_DIR="data/chunks"
LEGACY_CHUNKS_PATH="data/chunks.json"
UUID_PATTERN=re.compile(r"^[a-f0-9]{32}$")

def get_document_path(filename):
    return os.path.join(DOCUMENTS_DIR,filename)

def get_chunk_path(document_id):
    return os.path.join(CHUNKS_DIR,f"{document_id}.json")

def get_legacy_document_id(filename):
    if not os.path.exists(LEGACY_CHUNKS_PATH): return None
    try:
        with open(LEGACY_CHUNKS_PATH,"r",encoding="utf-8") as f: chunks=json.load(f)
        for chunk in chunks:
            if chunk.get("source")==filename and chunk.get("document_id"): return str(chunk["document_id"])
    except Exception: pass
    return None

def get_document_id_from_filename(filename):
    stem=os.path.splitext(filename)[0]
    first_part=stem.split("_",1)[0]
    if UUID_PATTERN.fullmatch(first_part): return first_part
    legacy_id=get_legacy_document_id(filename)
    return legacy_id or stem

def get_chunk_count(document_id):
    path=get_chunk_path(document_id)
    if os.path.exists(path):
        try:
            with open(path,"r",encoding="utf-8") as f: chunks=json.load(f)
            return len(chunks) if isinstance(chunks,list) else 0
        except Exception: return 0
    if os.path.exists(LEGACY_CHUNKS_PATH):
        try:
            with open(LEGACY_CHUNKS_PATH,"r",encoding="utf-8") as f: chunks=json.load(f)
            return sum(1 for chunk in chunks if str(chunk.get("document_id") or "")==str(document_id))
        except Exception: return 0
    return 0

def get_page_count(path):
    try:
        with pymupdf.open(path) as doc: return doc.page_count
    except Exception: return 0

def get_document_info(filename):
    path=get_document_path(filename)
    if not os.path.exists(path): return None
    document_id=get_document_id_from_filename(filename)
    status_record=get_status(document_id) or {}
    chunk_count=get_chunk_count(document_id)
    return {"document_id":document_id,"filename":status_record.get("filename") or filename,"status":status_record.get("status") or ("ready" if chunk_count>0 else "processing"),"page_count":status_record.get("page_count") or get_page_count(path),"chunk_count":status_record.get("chunk_count") or chunk_count,"sha256":status_record.get("sha256"),"size_bytes":status_record.get("size_bytes") or os.path.getsize(path),"error":status_record.get("error"),"created_at":status_record.get("created_at"),"updated_at":status_record.get("updated_at")}

def list_documents():
    if not os.path.exists(DOCUMENTS_DIR): return []
    documents=[]
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.lower().endswith(".pdf"):
            info=get_document_info(filename)
            if info: documents.append(info)
    return documents

def get_document(document_id):
    for document in list_documents():
        if document["document_id"]==document_id: return document
    return None

def find_document_by_hash(sha256):
    return find_by_sha256(sha256)

def delete_document_files(document_id):
    document=get_document(document_id)
    status=get_status(document_id)
    if not document and not status: return False
    storage_filename=(status or {}).get("storage_filename")
    if not storage_filename and document:
        storage_filename=next((f for f in os.listdir(DOCUMENTS_DIR) if get_document_id_from_filename(f)==document_id),None) if os.path.exists(DOCUMENTS_DIR) else None
    if storage_filename:
        path=get_document_path(storage_filename)
        if os.path.exists(path): os.remove(path)
    chunk_path=get_chunk_path(document_id)
    if os.path.exists(chunk_path): os.remove(chunk_path)
    delete_status(document_id)
    return True