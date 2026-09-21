from fastapi import FastAPI,Request
from api.routes import chat,documents,health,websocket

app=FastAPI(title="Tata Annual Report RAG API",version="1.0.0")

@app.get("/debug")
def debug():
    return {"status":"ok","message":"Render API is receiving requests"}

@app.middleware("http")
async def request_logger(request:Request,call_next):
    print(f"========== REQUEST START: {request.method} {request.url.path} ==========",flush=True)
    try:
        response=await call_next(request)
        print(f"========== REQUEST END: {request.method} {request.url.path} STATUS={response.status_code} ==========",flush=True)
        return response
    except Exception as exc:
        print(f"========== REQUEST ERROR: {request.method} {request.url.path} ERROR={exc} ==========",flush=True)
        raise

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(websocket.router)