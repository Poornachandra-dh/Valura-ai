import asyncio
import json
import os
from typing import Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from src.safety import check as check_safety
from src.classifier import classify, get_openai_client
from src.router import route

app = FastAPI(title="Valura AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session persistence: { session_id: [history] }
SESSIONS: Dict[str, list] = {}

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str

def load_user_data(user_id: str) -> Dict[str, Any]:
    """Loads user data from the fixtures."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures", "users")
    user_file = os.path.join(fixtures_dir, f"{user_id}.json")
    if os.path.exists(user_file):
        with open(user_file, "r") as f:
            return json.load(f)
    return {"user_id": user_id, "name": "Unknown", "positions": []}

async def process_chat(request: ChatRequest) -> AsyncGenerator[ServerSentEvent, None]:
    """
    Executes the full pipeline and yields SSE events.
    Pipeline: Safety Guard -> Classifier -> Router.
    """
    try:
        # 1. Safety Guard
        safety_verdict = check_safety(request.query)
        if safety_verdict.blocked:
            yield ServerSentEvent(
                event="error",
                data=json.dumps({"error": "safety_violation", "message": safety_verdict.message})
            )
            return

        # Load context
        history = SESSIONS.get(request.session_id, [])
        user_data = load_user_data(request.user_id)

        # 2. Intent Classifier
        # Using a timeout to prevent hanging, simulated via asyncio.to_thread
        classification = await asyncio.to_thread(classify, request.query, history)
        
        yield ServerSentEvent(
            event="classification",
            data=classification.model_dump_json()
        )

        # 3. Router & Agent Execution
        # Execute the agent synchronously in a thread
        agent_response = await asyncio.to_thread(route, classification, user_data)
        
        # In a real streaming scenario we'd stream tokens from the LLM, but since the
        # assignment returns a structured JSON (e.g. Portfolio Health), we stream the 
        # final object as a completed event.
        yield ServerSentEvent(
            event="agent_response",
            data=json.dumps(agent_response)
        )
        
        # 4. Update memory
        history.append({
            "user": request.query,
            "agent_response": json.dumps(agent_response)
        })
        SESSIONS[request.session_id] = history
        
        yield ServerSentEvent(event="done", data="[DONE]")

    except Exception as e:
        yield ServerSentEvent(
            event="error",
            data=json.dumps({"error": "internal_error", "message": str(e)})
        )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main endpoint for chatting with Valura AI.
    Streams SSE responses.
    """
    # Enforce a sane timeout (e.g., 15 seconds) for the entire stream generation
    async def timeout_wrapper():
        try:
            async with asyncio.timeout(15.0):
                async for event in process_chat(request):
                    yield event
        except asyncio.TimeoutError:
            yield ServerSentEvent(
                event="error", 
                data=json.dumps({"error": "timeout", "message": "The request timed out."})
            )
            
    return EventSourceResponse(timeout_wrapper())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
