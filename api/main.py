from fastapi import FastAPI
from api.routes import chat,documents,health,websocket

app=FastAPI(
    title="Tata Annual Report RAG API",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(websocket.router)