"""
FILE: api.py
-------------
FastAPI backend for Aristostat.
Wraps the LangGraph pipeline from main.py and exposes
SSE-based streaming endpoints for the UI.

Endpoints:
  POST /session/start     — upload CSV + query, start pipeline
  POST /session/resume    — send user response to interrupt
  GET  /session/{id}/stream — SSE stream of pipeline events
  GET  /report/{id}       — get final report data
  GET  /download/{id}     — download generated docx
"""

import os
import uuid
import json
import asyncio
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Import Aristostat graph functions ──
from main import run_aristostat, resume_aristostat, _get_interrupt

app = FastAPI(title="Aristostat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──
# Each session: { state, interrupt, queue, thread_id, csv_path, status }
sessions: dict[str, dict] = {}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class ResumeRequest(BaseModel):
    session_id: str
    response:   str


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _extract_interrupt(state: dict) -> dict | None:
    return _get_interrupt(state)


def _build_event(event_type: str, data: Any) -> str:
    """Format a Server-Sent Event."""
    payload = json.dumps({"type": event_type, "data": data})
    return f"data: {payload}\n\n"


def _push(session_id: str, event_type: str, data: Any):
    """Push an event into the session queue."""
    q = sessions[session_id]["queue"]
    q.put_nowait(_build_event(event_type, data))


def _run_pipeline_step(session_id: str, state: dict):
    """
    Processes one pipeline step synchronously.
    Pushes events to the session queue.
    Called in a thread to avoid blocking the event loop.
    """
    session = sessions[session_id]
    thread_id = session["thread_id"]

    interrupt_val = _extract_interrupt(state)

    if interrupt_val is None:
        # Pipeline finished
        report = state.get("report_output", {})
        fatal  = state.get("fatal_error")

        if fatal:
            _push(session_id, "fatal_error", {"message": fatal})
        else:
           # WITH
            docx_path = report.get("docx_path", "")
            _push(session_id, "complete", {
                "report":      report,
                "docx_path":   docx_path,
                "docx_ready":  bool(docx_path) and report.get("docx_generated", False),
            })
        session["status"] = "complete"
        session["state"]  = state
        _push(session_id, "done", {})
        return

    # There is an interrupt — send it to the UI
    interrupt_type = interrupt_val.get("type", "confirm")
    options = interrupt_val.get("options", [])

    # Normalise options for JSON
    clean_options = []
    for opt in options:
        if isinstance(opt, dict):
            clean_options.append({
                "solution_id":  opt.get("solution_id", ""),
                "description":  opt.get("description", str(opt)),
                "action_type":  opt.get("action_type", ""),
            })
        else:
            clean_options.append({"solution_id": str(opt), "description": str(opt)})

    _push(session_id, "interrupt", {
        "interrupt_type": interrupt_type,
        "message":        interrupt_val.get("message", ""),
        "prompt":         interrupt_val.get("prompt", "Your response:"),
        "options":        clean_options,
        "auto_proceed":   interrupt_type == "info",
    })

    session["status"]    = "waiting"
    session["state"]     = state
    session["interrupt"] = interrupt_val


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/session/start")
async def start_session(
    file:  UploadFile = File(...),
    query: str        = Form(...),
):
    """Upload CSV and start the pipeline."""
    session_id = str(uuid.uuid4())
    thread_id  = f"thread_{session_id}"

    # Save uploaded file
    csv_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
    content  = await file.read()
    csv_path.write_bytes(content)

    # Init session
    queue = asyncio.Queue()
    sessions[session_id] = {
        "thread_id": thread_id,
        "csv_path":  str(csv_path),
        "query":     query,
        "queue":     queue,
        "state":     None,
        "interrupt": None,
        "status":    "running",
    }

    # Run first pipeline step in background
    loop = asyncio.get_event_loop()
    def _start():
        try:
            state = run_aristostat(str(csv_path), query, thread_id)
            _run_pipeline_step(session_id, state)
        except Exception as e:
            _push(session_id, "error", {"message": str(e), "trace": traceback.format_exc()})
            _push(session_id, "done", {})

    loop.run_in_executor(None, _start)

    return {"session_id": session_id}


@app.post("/session/resume")
async def resume_session(req: ResumeRequest):
    """Send user response to the current interrupt."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "waiting":
        raise HTTPException(status_code=400, detail="Session not waiting for input")

    session["status"] = "running"
    thread_id = session["thread_id"]

    loop = asyncio.get_event_loop()
    def _resume():
        try:
            state = resume_aristostat(req.response, thread_id)
            _run_pipeline_step(session_id=req.session_id, state=state)
        except Exception as e:
            _push(req.session_id, "error", {"message": str(e), "trace": traceback.format_exc()})
            _push(req.session_id, "done", {})

    loop.run_in_executor(None, _resume)

    return {"ok": True}


@app.get("/session/{session_id}/stream")
async def stream_session(session_id: str):
    """SSE stream — client connects once and receives all events."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    queue = session["queue"]

    async def event_generator():
        yield _build_event("connected", {"session_id": session_id})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
                yield event
                if '"type": "done"' in event:
                    break
            except asyncio.TimeoutError:
                yield _build_event("ping", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/download/{session_id}")
async def download_report(session_id: str):
    """Download the generated docx report."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    state = session.get("state", {})
    if not state:
        raise HTTPException(status_code=404, detail="No state found")

    report   = state.get("report_output", {})
    docx_path = report.get("docx_path", "")

    if not docx_path or not Path(docx_path).exists():
        raise HTTPException(status_code=404, detail="Report not generated yet")

    return FileResponse(
        path=docx_path,
        filename="aristostat_report.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Serve frontend ──
app.mount("/", StaticFiles(directory=".", html=True), name="static")