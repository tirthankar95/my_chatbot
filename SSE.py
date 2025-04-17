from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
global_response = """The universe is a vast and awe-inspiring expanse, encompassing everything that exists—from the tiniest subatomic particles to the largest galaxies. It is believed to have originated around 13.8 billion years ago with the Big Bang, an event that marked the beginning of space, time, and matter. The universe is constantly expanding, with billions of galaxies housing countless stars, planets, and other celestial bodies. It operates under fundamental forces and laws, such as gravity, which govern its behavior. Despite humanity's advancements in science and technology, much of the universe remains a mystery, with phenomena like dark matter and dark energy continuing to intrigue scientists and spark exploration. Its sheer scale and complexity make it one of the greatest marvels of existence."""
active_sessions = {}

def event_generator(session_id: str):
    global global_response, active_sessions
    if session_id not in active_sessions: yield ""
    for resp in global_response.split(" "):
        time.sleep(0.1)
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
    active_sessions[session_id] = True
    return {"uuid": f"{session_id}"}