"""Web-facing pipeline and connection services."""

import asyncio
import threading
from dataclasses import dataclass, field

from fastapi import WebSocket

from disco.asr import make_transcriber
from disco.audio.source import AudioSource
from disco.diar import Diarizer
from disco.runtime.events import (
    EnrichedFinal,
    EnrichedInterim,
    EventBus,
    FinalDiscarded,
    Interim,
)
from disco.runtime.runtime import Runtime
from disco.translation import KoreanTranslator
from disco.vad import SmartTurnEndpoint


@dataclass
class AppConfig:
    device: int | None = None
    language: str = "English"
    translate_korean: bool = False
    silence_duration: float = 0.5
    sample_rate: int = 16000
    min_utterance_duration: float = 0.5
    speaker_change_hold: float = 0.4
    max_utterance_duration: float = 10.0
    asr_backend: str = "voxtral"
    model_name: str | None = None
    translation_model: str | None = None
    smart_turn: bool = False
    smart_turn_model: str = "mlx-community/smart-turn-v3"
    smart_turn_threshold: float = 0.5


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
        translation_model: str | None = None,
        smart_turn: bool = False,
        smart_turn_model: str = "mlx-community/smart-turn-v3",
        smart_turn_threshold: float = 0.5,
    ) -> None:
        self.config = AppConfig(
            device=device,
            language=language,
            translate_korean=translate_korean,
            silence_duration=silence_duration,
            min_utterance_duration=min_utterance_duration,
            asr_backend=asr_backend,
            model_name=model_name,
            translation_model=translation_model,
            smart_turn=smart_turn,
            smart_turn_model=smart_turn_model,
            smart_turn_threshold=smart_turn_threshold,
        )

    def config_payload(self) -> dict:
        return {
            "language": self.config.language,
            "translate_korean": self.config.translate_korean,
            "is_recording": self.is_recording,
            "smart_turn": self.config.smart_turn,
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
        translator = None
        if cfg.translate_korean:
            model_name = cfg.translation_model
            translator = (
                KoreanTranslator(model_name=model_name)
                if model_name is not None
                else KoreanTranslator()
            )
        if translator is not None:
            translator.load()
        smart_turn = None
        if cfg.smart_turn:
            smart_turn = SmartTurnEndpoint(
                model_name=cfg.smart_turn_model,
                sample_rate=cfg.sample_rate,
                threshold=cfg.smart_turn_threshold,
            )

        bus = EventBus()
        bus.subscribe(Interim, self._on_interim)
        bus.subscribe(EnrichedInterim, self._on_enriched_interim)
        bus.subscribe(FinalDiscarded, self._on_final_discarded)
        bus.subscribe(EnrichedFinal, self._on_final)

        self.bus = bus
        return Runtime(
            bus=bus,
            transcriber=transcriber,
            diarizer=diarizer,
            translator=translator,
            smart_turn=smart_turn,
            language=cfg.language,
            sample_rate=cfg.sample_rate,
            silence_duration=cfg.silence_duration,
            min_utterance_duration=cfg.min_utterance_duration,
            speaker_change_hold=cfg.speaker_change_hold,
            max_utterance_duration=cfg.max_utterance_duration,
        )

    def _on_interim(self, event: Interim) -> None:
        msg: dict = {
            "type": "interim",
            "text": event.text,
            "span": event.span,
            "utterance_id": event.utterance_id,
        }
        if event.speaker is not None:
            msg["speaker"] = event.speaker
        self.connections.schedule(msg)

    def _on_enriched_interim(self, event: EnrichedInterim) -> None:
        msg: dict = {
            "type": "interim",
            "text": event.text,
            "span": event.span,
            "utterance_id": event.utterance_id,
        }
        if event.speaker is not None:
            msg["speaker"] = event.speaker
        if event.translation is not None:
            msg["translation"] = event.translation
        self.connections.schedule(msg)

    def _on_final(self, event: EnrichedFinal) -> None:
        msg: dict = {
            "type": "final",
            "text": event.text,
            "span": event.span,
            "utterance_id": event.utterance_id,
            "utterance_ids": event.utterance_ids,
        }
        if event.speaker is not None:
            msg["speaker"] = event.speaker
        if event.translation is not None:
            msg["translation"] = event.translation
        self.connections.schedule(msg)

    def _on_final_discarded(self, event: FinalDiscarded) -> None:
        msg: dict = {
            "type": "final_discarded",
            "span": event.span,
            "utterance_id": event.utterance_id,
            "reason": event.reason,
        }
        if event.speaker is not None:
            msg["speaker"] = event.speaker
        self.connections.schedule(msg)
