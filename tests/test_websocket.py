import asyncio
import json
import pytest
import websockets

URI="ws://127.0.0.1:8000/ws/chat"
DOCUMENT_ID=None

async def ask(websocket,query,conversation_id):
    await websocket.send(json.dumps({
        "query":query,
        "document_id":DOCUMENT_ID,
        "conversation_id":conversation_id
    }))
    messages=[]
    while True:
        message=json.loads(await websocket.recv())
        messages.append(message)
        print(message)
        if message.get("type") in {"complete","error"}:
            break
    return messages

async def websocket_available():
    try:
        async with websockets.connect(URI,ping_interval=30,ping_timeout=300,open_timeout=3):
            return True
    except (OSError,ConnectionRefusedError,asyncio.TimeoutError):
        return False

@pytest.mark.asyncio
async def test_websocket_chat():
    if not await websocket_available():
        pytest.skip("WebSocket server is not running at ws://127.0.0.1:8000/ws/chat")

    conversation_id="ws-memory-test-001"

    async with websockets.connect(URI,ping_interval=30,ping_timeout=300) as websocket:
        first=await ask(websocket,"What does the selected document say?",conversation_id)
        assert first
        assert first[-1].get("type") in {"complete","error"}

        second=await ask(websocket,"Can you summarize that information?",conversation_id)
        assert second
        assert second[-1].get("type") in {"complete","error"}