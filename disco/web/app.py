"""FastAPI application for real-time ASR web UI."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from disco.web.service import ConnectionManager, PipelineService

STATIC_DIR = Path(__file__).parent / "static"

connections = ConnectionManager()
pipeline = PipelineService(connections)


def set_config(
    device: int | None = None,
    language: str = "English",
    translate_korean: bool = False,
    silence_duration: float = 0.5,
    min_utterance_duration: float = 0.5,
    asr_backend: str = "voxtral",
    model_name: str | None = None,
):
    pipeline.set_config(
        device=device,
        language=language,
        translate_korean=translate_korean,
        silence_duration=silence_duration,
        min_utterance_duration=min_utterance_duration,
        asr_backend=asr_backend,
        model_name=model_name,
    )


app = FastAPI(title="Disco - Real-time ASR")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup() -> None:
    connections.attach_loop()


@app.on_event("shutdown")
async def shutdown() -> None:
    pipeline.stop()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/config")
async def get_config():
    return pipeline.config_payload()


@app.post("/api/start")
async def start_recording():
    try:
        status = pipeline.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": status}


@app.post("/api/stop")
async def stop_recording():
    return {"status": pipeline.stop()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connections.connect(websocket)

    await websocket.send_json({"type": "config", **pipeline.config_payload()})

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        connections.disconnect(websocket)
