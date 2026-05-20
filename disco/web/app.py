"""FastAPI application for real-time ASR web UI."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from disco.asr import Transcriber
from disco.audio.source import AudioSource
from disco.diar import Diarizer
from disco.runtime.events import EnrichedFinal, EventBus, Interim
from disco.runtime.runtime import Runtime
from disco.translation import KoreanTranslator

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class AppConfig:
    device: int | None = None
    language: str = "English"
    translate_korean: bool = False
    silence_duration: float = 0.5
    sample_rate: int = 16000
    min_utterance_duration: float = 0.5
    speaker_change_hold: float = 0.4


@dataclass
class AppState:
    config: AppConfig = field(default_factory=AppConfig)
    clients: set[WebSocket] = field(default_factory=set)
    runtime: Runtime | None = None
    source: AudioSource | None = None
    bus: EventBus | None = None
    loop: asyncio.AbstractEventLoop | None = None


_state = AppState()


def set_config(
    device: int | None = None,
    language: str = "English",
    translate_korean: bool = False,
    silence_duration: float = 0.5,
    min_utterance_duration: float = 0.5,
):
    _state.config = AppConfig(
        device=device,
        language=language,
        translate_korean=translate_korean,
        silence_duration=silence_duration,
        min_utterance_duration=min_utterance_duration,
    )


async def _broadcast(message: dict) -> None:
    disconnected: set[WebSocket] = set()
    for client in _state.clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.add(client)
    _state.clients -= disconnected


def _schedule_broadcast(message: dict) -> None:
    """Hand a message to the asyncio loop from a worker thread."""
    if _state.loop is None or not _state.clients:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(message), _state.loop)


def _on_interim(event: Interim) -> None:
    msg: dict = {"type": "interim", "text": event.text}
    if event.speaker is not None:
        msg["speaker"] = event.speaker
    _schedule_broadcast(msg)


def _on_final(event: EnrichedFinal) -> None:
    msg: dict = {"type": "final", "text": event.text}
    if event.speaker is not None:
        msg["speaker"] = event.speaker
    if event.translation is not None:
        msg["translation"] = event.translation
    _schedule_broadcast(msg)


def _build_runtime() -> Runtime:
    cfg = _state.config
    transcriber = Transcriber(sample_rate=cfg.sample_rate)
    diarizer = Diarizer(sample_rate=cfg.sample_rate)
    translator = KoreanTranslator() if cfg.translate_korean else None
    if translator is not None:
        translator.load()

    bus = EventBus()
    bus.subscribe(Interim, _on_interim)
    bus.subscribe(EnrichedFinal, _on_final)

    _state.bus = bus
    return Runtime(
        bus=bus,
        transcriber=transcriber,
        diarizer=diarizer,
        translator=translator,
        language=cfg.language,
        sample_rate=cfg.sample_rate,
        silence_duration=cfg.silence_duration,
        min_utterance_duration=cfg.min_utterance_duration,
        speaker_change_hold=cfg.speaker_change_hold,
    )


def _start_runtime() -> None:
    if _state.runtime is not None:
        return
    _state.loop = asyncio.get_event_loop()
    _state.runtime = _build_runtime()
    _state.source = AudioSource(
        sample_rate=_state.config.sample_rate,
        channels=1,
        device=_state.config.device,
    )
    _state.runtime.start(_state.source)


def _stop_runtime() -> None:
    if _state.runtime is None:
        return
    _state.runtime.stop()
    _state.runtime = None
    _state.source = None


app = FastAPI(title="Disco - Real-time ASR")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup() -> None:
    _start_runtime()


@app.on_event("shutdown")
async def shutdown() -> None:
    _stop_runtime()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/config")
async def get_config():
    return {
        "language": _state.config.language,
        "translate_korean": _state.config.translate_korean,
        "is_recording": _state.runtime is not None,
    }


@app.post("/api/start")
async def start_recording():
    if _state.runtime is not None:
        return {"status": "already_recording"}
    _start_runtime()
    return {"status": "started"}


@app.post("/api/stop")
async def stop_recording():
    if _state.runtime is None:
        return {"status": "not_recording"}
    _stop_runtime()
    return {"status": "stopped"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _state.clients.add(websocket)

    await websocket.send_json({
        "type": "config",
        "language": _state.config.language,
        "translate_korean": _state.config.translate_korean,
        "is_recording": _state.runtime is not None,
    })

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        _state.clients.discard(websocket)
