from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from router import ChainRouter
from uuid import uuid4
import time

class QueryRequest(BaseModel):
    query: str 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
active_sessions = {}
router = ChainRouter()

def event_generator(session_id: str):
    global active_sessions
    if session_id not in active_sessions: yield "[DONE]"
    full_response = router.call_chain(active_sessions[session_id])
    for resp in full_response.split(" "):
        time.sleep(0.01)
        yield f"data: {resp}\n\n"
    active_sessions.pop(session_id, None)

@app.get("/sse/{session_id}")
async def sse_endpoint(session_id: str):
    if session_id in active_sessions:
        return StreamingResponse(event_generator(session_id), media_type="text/event-stream")

@app.post("/query")
async def get_query(request: QueryRequest):
    global active_sessions
    session_id = str(uuid4())
    active_sessions[session_id] = request.query
    return {"uuid": f"{session_id}"}