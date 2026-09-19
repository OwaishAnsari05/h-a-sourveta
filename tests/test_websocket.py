import asyncio
import json
import websockets

async def ask(websocket,query,conversation_id):
    await websocket.send(json.dumps({
        "query":query,
        "document_id":"tata_annual_report_2024_25",
        "conversation_id":conversation_id
    }))

    while True:
        message=json.loads(await websocket.recv())
        print(message)

        if message.get("type")=="complete":
            break

        if message.get("type")=="error":
            break

async def main():
    uri="ws://127.0.0.1:8000/ws/chat"

    async with websockets.connect(uri,ping_interval=30,ping_timeout=300) as websocket:
        conversation_id="ws-memory-test-001"

        await ask(
            websocket,
            "What was the total income for FY 2024-25?",
            conversation_id
        )

        print("\n--- FOLLOW-UP QUESTION ---\n")

        await ask(
            websocket,
            "How much did it increase?",
            conversation_id
        )

asyncio.run(main())