import hmac
import re
from api.config import API_KEY,MAX_UPLOAD_BYTES

_FILENAME_RE=re.compile(r"[^A-Za-z0-9._()\- ]+")

def sanitize_filename(filename:str)->str:
    name=(filename or "").replace("\\","/").split("/")[-1].strip()
    name=_FILENAME_RE.sub("_",name)
    name=re.sub(r"\s+"," ",name)
    if not name: name="document.pdf"
    if not name.lower().endswith(".pdf"): name += ".pdf"
    return name[:180]

def validate_pdf_bytes(data:bytes)->None:
    if not data: raise ValueError("Uploaded file is empty.")
    if len(data)>MAX_UPLOAD_BYTES: raise ValueError(f"PDF exceeds the {MAX_UPLOAD_BYTES//(1024*1024)} MB upload limit.")
    if not data.startswith(b"%PDF-"): raise ValueError("Uploaded file is not a valid PDF.")

def api_key_valid(value:str|None)->bool:
    if not API_KEY: return True
    return bool(value) and hmac.compare_digest(value,API_KEY)