from datetime import datetime,timezone
from fastapi import APIRouter
from api.config import MAX_UPLOAD_MB,RATE_LIMIT_PER_MINUTE

router=APIRouter(tags=["Health"])

@router.get("/health")
def health():
    return {"status":"healthy","service":"sourveta-document-intelligence","version":"1.0.0","timestamp":datetime.now(timezone.utc).isoformat(),"upload_limit_mb":MAX_UPLOAD_MB,"rate_limit_per_minute":RATE_LIMIT_PER_MINUTE}