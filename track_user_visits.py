from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

app = FastAPI()

# Allow all origins for testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to track session activity.
active_sessions = {}

# Session inactivity threshold (in seconds)
INACTIVE_THRESHOLD = 300  # 5 minutes

class SessionUpdate(BaseModel):
    session_id: str

@app.post("/update_session")
def update_session(session: SessionUpdate):
    global active_sessions
    active_sessions[session.session_id] = time.time()
    print(f"Received update for session: {session.session_id} at {active_sessions[session.session_id]}")
    return {"message": "Session updated successfully."}

@app.get("/active_sessions")
def get_active_sessions():
    global active_sessions
    current_time = time.time()
    # Remove sessions inactive for more than 5 minutes in-place.
    for sid in list(active_sessions.keys()):
        if current_time - active_sessions[sid] >= INACTIVE_THRESHOLD:
            del active_sessions[sid]
    print(f"Active sessions count: {len(active_sessions)}")
    return {"active_user_count": len(active_sessions)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
