import json
import os
import re
import pymupdf

DOCUMENTS_DIR = "data/documents"
CHUNKS_DIR = "data/chunks"
LEGACY_CHUNKS_PATH = "data/chunks.json"
UUID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

def get_document_path(filename):
    return os.path.join(DOCUMENTS_DIR,filename)

def get_chunk_path(document_id):
    return os.path.join(CHUNKS_DIR,f"{document_id}.json")

def get_legacy_document_id(filename):
    if not os.path.exists(LEGACY_CHUNKS_PATH):
        return None
    try:
        with open(LEGACY_CHUNKS_PATH,"r",encoding="utf-8") as f:
            chunks = json.load(f)
        for chunk in chunks:
            if chunk.get("source") == filename:
                document_id = chunk.get("document_id")
                if document_id:
                    return str(document_id)
    except Exception:
        pass
    return None

def get_document_id_from_filename(filename):
    stem = os.path.splitext(filename)[0]
    first_part = stem.split("_",1)[0]
    if UUID_PATTERN.fullmatch(first_part):
        return first_part
    legacy_id = get_legacy_document_id(filename)
    return legacy_id or stem

def get_chunk_count(document_id):
    path = get_chunk_path(document_id)
    if os.path.exists(path):
        try:
            with open(path,"r",encoding="utf-8") as f:
                chunks = json.load(f)
            return len(chunks) if isinstance(chunks,list) else 0
        except Exception:
            return 0
    if document_id == "tata_annual_report_2024_25" and os.path.exists(LEGACY_CHUNKS_PATH):
        try:
            with open(LEGACY_CHUNKS_PATH,"r",encoding="utf-8") as f:
                chunks = json.load(f)
            return len(chunks) if isinstance(chunks,list) else 0
        except Exception:
            return 0
    return 0

def get_page_count(path):
    try:
        doc = pymupdf.open(path)
        page_count = doc.page_count
        doc.close()
        return page_count
    except Exception:
        return 0

def get_document_info(filename):
    path = get_document_path(filename)
    if not os.path.exists(path):
        return None
    document_id = get_document_id_from_filename(filename)
    chunk_count = get_chunk_count(document_id)
    return {
        "document_id":document_id,
        "filename":filename,
        "status":"ready" if chunk_count > 0 else "processing",
        "page_count":get_page_count(path),
        "chunk_count":chunk_count,
    }

def list_documents():
    if not os.path.exists(DOCUMENTS_DIR):
        return []
    documents = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        info = get_document_info(filename)
        if info:
            documents.append(info)
    return documents

def get_document(document_id):
    for document in list_documents():
        if document["document_id"] == document_id:
            return document
    return None