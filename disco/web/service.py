"""Web-facing pipeline and connection services."""

import asyncio
import threading
from dataclasses import dataclass, field

from fastapi import WebSocket

from disco.asr import make_transcriber
from disco.audio.source import AudioSource
from disco.diar import Diarizer
from disco.runtime.events import EnrichedFinal, EnrichedInterim, EventBus, Interim
from disco.runtime.runtime import Runtime
from disco.translation import KoreanTranslator


@dataclass
class AppConfig:
    device: int | None = None
    language: str = "English"
    translate_korean: bool = False
    silence_duration: float = 0.5
    sample_rate: int = 16000
    min_utterance_duration: float = 0.5
    speaker_change_hold: float = 0.4
    asr_backend: str = "voxtral"
    model_name: str | None = None


@dataclass
class ConnectionManager:
    clients: set[WebSocket] = field(default_factory=set)
    loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self) -> None:
        self.loop = asyncio.get_running_loop()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    async def _broadcast(self, message: dict) -> None:
        disconnected: set[WebSocket] = set()
        for client in list(self.clients):
            try:
                await client.send_json(message)
            except Exception:
                disconnected.add(client)
        self.clients -= disconnected

    def schedule(self, message: dict) -> None:
        if self.loop is None or not self.clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)


class PipelineService:
    """Owns the realtime pipeline lifecycle behind the web API."""

    def __init__(self, connections: ConnectionManager):
        self.config = AppConfig()
        self.connections = connections
        self.runtime: Runtime | None = None
        self.source: AudioSource | None = None
        self.bus: EventBus | None = None
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self.runtime is not None

    def set_config(
        self,
        *,
        device: int | None = None,
        language: str = "English",
        translate_korean: bool = False,
        silence_duration: float = 0.5,
        min_utterance_duration: float = 0.5,
        asr_backend: str = "voxtral",
        model_name: str | None = None,
    ) -> None:
        self.config = AppConfig(
            device=device,
            language=language,
            translate_korean=translate_korean,
            silence_duration=silence_duration,
            min_utterance_duration=min_utterance_duration,
            asr_backend=asr_backend,
            model_name=model_name,
        )

    def config_payload(self) -> dict:
        return {
            "language": self.config.language,
            "translate_korean": self.config.translate_korean,
            "is_recording": self.is_recording,
        }

    def start(self) -> str:
        with self._lock:
            if self.runtime is not None:
                return "already_recording"
            self.runtime = self._build_runtime()
            self.source = AudioSource(
                sample_rate=self.config.sample_rate,
                channels=1,
                device=self.config.device,
            )
            try:
                self.runtime.start(self.source)
            except Exception:
                self.runtime = None
                self.source = None
                raise
            return "started"

    def stop(self) -> str:
        with self._lock:
            if self.runtime is None:
                return "not_recording"
            self.runtime.stop()
            self.runtime = None
            self.source = None
            return "stopped"

    def _build_runtime(self) -> Runtime:
        cfg = self.config
        transcriber = make_transcriber(
            cfg.asr_backend,
            model_name=cfg.model_name,
            sample_rate=cfg.sample_rate,
            language=cfg.language,
        )
        diarizer = Diarizer(sample_rate=cfg.sample_rate)
        translator = KoreanTranslator() if cfg.translate_korean else None
        if translator is not None:
            translator.load()

        bus = EventBus()
        bus.subscribe(Interim, self._on_interim)
        bus.subscribe(EnrichedInterim, self._on_enriched_interim)
        bus.subscribe(EnrichedFinal, self._on_final)

        self.bus = bus
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

    def _on_interim(self, event: Interim) -> None:
        msg: dict = {"type": "interim", "text": event.text, "span": event.span}
        if event.speaker is not None:
            msg["speaker"] = event.speaker
        self.connections.schedule(msg)

    def _on_enriched_interim(self, event: EnrichedInterim) -> None:
        msg: dict = {"type": "interim", "text": event.text, "span": event.span}
        if event.speaker is not None:
            msg["speaker"] = event.speaker
        if event.translation is not None:
            msg["translation"] = event.translation
        self.connections.schedule(msg)

    def _on_final(self, event: EnrichedFinal) -> None:
        msg: dict = {"type": "final", "text": event.text}
        if event.speaker is not None:
            msg["speaker"] = event.speaker
        if event.translation is not None:
            msg["translation"] = event.translation
        self.connections.schedule(msg)
