"""FastAPI application for real-time ASR web UI."""

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from disco.asr import Transcriber
from disco.asr.transcriber import StreamingTranscription, is_hallucination
from disco.audio.capture import AudioCapture
from disco.config import LANG_CODE_MAP
from disco.diar import Diarizer
from disco.translation import KoreanTranslator
from disco.vad import SileroVAD

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class AppConfig:
    """Application configuration."""

    device: int | None = None
    language: str = "English"
    translate_korean: bool = False
    silence_duration: float = 0.5
    sample_rate: int = 16000
    # Minimum audio fed into a session before we'll honor a VAD-triggered finalize.
    min_utterance_duration: float = 0.5
    # Seconds the diarizer must see a different speaker before we close the session.
    speaker_change_hold: float = 0.4


@dataclass
class AppState:
    """Application state."""

    config: AppConfig = field(default_factory=AppConfig)
    is_recording: bool = False
    clients: set[WebSocket] = field(default_factory=set)
    vad: SileroVAD | None = None
    transcriber: Transcriber | None = None
    translator: KoreanTranslator | None = None
    diarizer: Diarizer | None = None
    audio_thread: threading.Thread | None = None


_state = AppState()


def set_config(
    device: int | None = None,
    language: str = "English",
    translate_korean: bool = False,
    silence_duration: float = 0.5,
    min_utterance_duration: float = 0.5,
):
    """Set application configuration from CLI."""
    _state.config = AppConfig(
        device=device,
        language=language,
        translate_korean=translate_korean,
        silence_duration=silence_duration,
        min_utterance_duration=min_utterance_duration,
    )


def _init_components():
    """Initialize ASR components."""
    if _state.vad is None:
        _state.vad = SileroVAD(sample_rate=_state.config.sample_rate)
        _state.vad.load()

    if _state.transcriber is None:
        _state.transcriber = Transcriber(sample_rate=_state.config.sample_rate)
        _state.transcriber.load()

    if _state.diarizer is None:
        _state.diarizer = Diarizer(sample_rate=_state.config.sample_rate)
        _state.diarizer.load()

    if _state.config.translate_korean and _state.translator is None:
        _state.translator = KoreanTranslator()
        _state.translator.load()


async def _broadcast(message: dict):
    """Broadcast message to all connected clients."""
    disconnected = set()
    for client in _state.clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.add(client)
    _state.clients -= disconnected


def _finalize(
    session: StreamingTranscription,
    diar_start: float,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Close the session, drain, and broadcast the final text with speaker label."""
    session.close()
    session.drain()
    text = session.text.strip()
    if not text or is_hallucination(text):
        return
    if not _state.clients:
        return

    result: dict = {"type": "final", "text": text}
    speaker = _state.diarizer.dominant_speaker_in(
        diar_start, _state.diarizer.elapsed_seconds()
    )
    if speaker is not None:
        result["speaker"] = int(speaker)

    if _state.translator:
        source_lang = LANG_CODE_MAP.get(_state.config.language.lower(), "en")
        result["translation"] = _state.translator.translate(text, source_lang)

    asyncio.run_coroutine_threadsafe(_broadcast(result), loop)


def _audio_processing_loop(loop: asyncio.AbstractEventLoop):
    """Continuously feed audio into a streaming session and emit deltas."""
    config = _state.config
    diarizer = _state.diarizer
    audio_capture = AudioCapture(
        sample_rate=config.sample_rate,
        channels=1,
        device=config.device,
    )

    required_silence_chunks = max(1, int(config.silence_duration / 0.1))
    min_utterance_samples = int(config.min_utterance_duration * config.sample_rate)

    session: StreamingTranscription | None = None
    samples_fed = 0
    silence_chunks = 0
    last_emitted_text = ""

    diar_start: float = 0.0
    bound_speaker: int | None = None
    speaker_change_start: float | None = None

    diarizer.start()

    def _reset_session_state():
        nonlocal session, samples_fed, silence_chunks, last_emitted_text
        nonlocal bound_speaker, speaker_change_start
        session = None
        samples_fed = 0
        silence_chunks = 0
        last_emitted_text = ""
        bound_speaker = None
        speaker_change_start = None

    try:
        with audio_capture:
            while _state.is_recording:
                chunk = audio_capture.get_chunk(timeout=0.05)

                if chunk is not None:
                    samples = chunk.reshape(-1)
                    has_speech = _state.vad.is_speech_chunk(chunk)
                    diarizer.feed(samples)

                    if has_speech:
                        if session is None:
                            session = _state.transcriber.start_session()
                            diar_start = diarizer.elapsed_seconds()
                            samples_fed = 0
                            last_emitted_text = ""
                            bound_speaker = None
                            speaker_change_start = None
                        silence_chunks = 0
                    elif session is not None:
                        silence_chunks += 1

                    if session is not None:
                        session.feed(samples)
                        samples_fed += len(samples)

                if session is None:
                    continue

                # Bind primary speaker once we have enough diarizer output.
                cur_t = diarizer.elapsed_seconds()
                if bound_speaker is None:
                    bound_speaker = diarizer.dominant_speaker_in(diar_start, cur_t)
                else:
                    # Detect sustained speaker change to force an early finalize.
                    latest = diarizer.speaker_at(cur_t - 0.2)
                    if latest is not None and latest != bound_speaker:
                        if speaker_change_start is None:
                            speaker_change_start = cur_t
                        elif cur_t - speaker_change_start >= config.speaker_change_hold:
                            _finalize(session, diar_start, loop)
                            _reset_session_state()
                            continue
                    else:
                        speaker_change_start = None

                # Drain a bounded amount of decode work per iteration.
                if session.step() and session.text != last_emitted_text:
                    last_emitted_text = session.text
                    if _state.clients:
                        msg: dict = {"type": "interim", "text": last_emitted_text}
                        if bound_speaker is not None:
                            msg["speaker"] = int(bound_speaker)
                        asyncio.run_coroutine_threadsafe(_broadcast(msg), loop)

                # End of utterance: long enough silence after some real audio.
                if (
                    silence_chunks >= required_silence_chunks
                    and samples_fed >= min_utterance_samples
                ):
                    _finalize(session, diar_start, loop)
                    _reset_session_state()

        # Recording stopped — finalize any in-flight utterance.
        if session is not None:
            _finalize(session, diar_start, loop)
    finally:
        diarizer.stop()


app = FastAPI(title="Disco - Real-time ASR")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup():
    """Auto-start recording on server startup."""
    _init_components()
    _state.is_recording = True

    loop = asyncio.get_event_loop()
    _state.audio_thread = threading.Thread(
        target=_audio_processing_loop,
        args=(loop,),
        daemon=True,
    )
    _state.audio_thread.start()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return {
        "language": _state.config.language,
        "translate_korean": _state.config.translate_korean,
        "is_recording": _state.is_recording,
    }


@app.post("/api/start")
async def start_recording():
    """Start audio capture and transcription."""
    if _state.is_recording:
        return {"status": "already_recording"}

    _init_components()
    _state.is_recording = True

    loop = asyncio.get_event_loop()
    _state.audio_thread = threading.Thread(
        target=_audio_processing_loop,
        args=(loop,),
        daemon=True,
    )
    _state.audio_thread.start()

    return {"status": "started"}


@app.post("/api/stop")
async def stop_recording():
    """Stop audio capture."""
    if not _state.is_recording:
        return {"status": "not_recording"}

    _state.is_recording = False
    if _state.audio_thread:
        _state.audio_thread.join(timeout=2.0)
        _state.audio_thread = None

    return {"status": "stopped"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for receiving transcription updates."""
    await websocket.accept()
    _state.clients.add(websocket)

    await websocket.send_json({
        "type": "config",
        "language": _state.config.language,
        "translate_korean": _state.config.translate_korean,
        "is_recording": _state.is_recording,
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


@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown."""
    _state.is_recording = False
    if _state.audio_thread:
        _state.audio_thread.join(timeout=2.0)
