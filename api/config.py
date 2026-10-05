import os

MAX_UPLOAD_MB=int(os.getenv("SOURVETA_MAX_UPLOAD_MB","25"))
MAX_UPLOAD_BYTES=MAX_UPLOAD_MB*1024*1024
RATE_LIMIT_PER_MINUTE=int(os.getenv("SOURVETA_RATE_LIMIT_PER_MINUTE","60"))
ALLOWED_ORIGINS=[item.strip() for item in os.getenv("SOURVETA_ALLOWED_ORIGINS","").split(",") if item.strip()]
API_KEY=os.getenv("SOURVETA_API_KEY","").strip()

def get_allowed_origins()->list[str]:
    return ALLOWED_ORIGINS or ["*"]