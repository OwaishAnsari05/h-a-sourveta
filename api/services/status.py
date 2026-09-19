import json
import os
import threading
from datetime import datetime,timezone

STATUS_PATH="data/document_status.json"
_lock=threading.Lock()

def _load():
    if not os.path.exists(STATUS_PATH):
        return {}
    try:
        with open(STATUS_PATH,"r",encoding="utf-8") as f:
            data=json.load(f)
        return data if isinstance(data,dict) else {}
    except Exception:
        return {}

def _save(data):
    directory=os.path.dirname(STATUS_PATH)
    if directory:
        os.makedirs(directory,exist_ok=True)
    temp_path=f"{STATUS_PATH}.tmp"
    with open(temp_path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    os.replace(temp_path,STATUS_PATH)

def create_status(document_id,filename):
    with _lock:
        data=_load()
        now=datetime.now(timezone.utc).isoformat()
        data[document_id]={
            "document_id":document_id,
            "filename":filename,
            "status":"processing",
            "page_count":0,
            "chunk_count":0,
            "error":None,
            "created_at":now,
            "updated_at":now,
        }
        _save(data)
        return data[document_id]

def update_status(document_id,status,page_count=None,chunk_count=None,error=None):
    with _lock:
        data=_load()
        item=data.get(document_id)
        if item is None:
            item={
                "document_id":document_id,
                "filename":"",
                "status":"processing",
                "page_count":0,
                "chunk_count":0,
                "error":None,
            }
        item["status"]=status
        if page_count is not None:
            item["page_count"]=page_count
        if chunk_count is not None:
            item["chunk_count"]=chunk_count
        item["error"]=error
        item["updated_at"]=datetime.now(timezone.utc).isoformat()
        data[document_id]=item
        _save(data)
        return item

def get_status(document_id):
    with _lock:
        return _load().get(document_id)

def list_statuses():
    with _lock:
        return list(_load().values())