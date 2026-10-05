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

def create_status(document_id,filename,sha256=None,size_bytes=0,storage_filename=None):
    with _lock:
        data=_load()
        now=datetime.now(timezone.utc).isoformat()
        data[document_id]={"document_id":document_id,"filename":filename,"storage_filename":storage_filename or "","sha256":sha256,"size_bytes":int(size_bytes or 0),"status":"processing","page_count":0,"chunk_count":0,"error":None,"created_at":now,"updated_at":now}
        _save(data)
        return data[document_id]

def find_by_sha256(sha256):
    if not sha256:
        return None
    with _lock:
        for item in _load().values():
            if item.get("sha256")==sha256:
                return item
    return None

def update_status(document_id,status,page_count=None,chunk_count=None,error=None):
    with _lock:
        data=_load()
        item=data.get(document_id)
        if item is None:
            item={"document_id":document_id,"filename":"","storage_filename":"","sha256":None,"size_bytes":0,"status":"processing","page_count":0,"chunk_count":0,"error":None,"created_at":datetime.now(timezone.utc).isoformat()}
        item["status"]=status
        if page_count is not None: item["page_count"]=page_count
        if chunk_count is not None: item["chunk_count"]=chunk_count
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

def delete_status(document_id):
    with _lock:
        data=_load()
        removed=data.pop(document_id,None)
        _save(data)
        return removed