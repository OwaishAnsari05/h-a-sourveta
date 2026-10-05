import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self,request:Request,call_next):
        response=await call_next(request)
        response.headers.setdefault("X-Content-Type-Options","nosniff")
        response.headers.setdefault("X-Frame-Options","DENY")
        response.headers.setdefault("Referrer-Policy","strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy","camera=(), microphone=(), geolocation=()")
        if request.url.path.startswith("/documents"):
            response.headers.setdefault("Cache-Control","no-store")
        response.headers.setdefault("X-Request-ID",request.headers.get("X-Request-ID",uuid.uuid4().hex))
        return response
