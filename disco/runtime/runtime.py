"""Wire AudioSource → workers → EventBus + enrich Final into EnrichedFinal."""

import os
import threading
import time

from disco.asr.transcriber import Transcriber
from disco.audio.source import AudioSource
from disco.config import LANG_CODE_MAP
from disco.diar.sortformer import Diarizer
from disco.runtime.events import (
    EnrichedFinal,
    EventBus,
    Final,
    QueueOverflow,
    SpeakerChange,
    SpeechEnd,
    SpeechStart,
)
from disco.runtime.transcriber_worker import TranscriberWorker
from disco.runtime.turn_detector import TurnDetector
from disco.translation.korean import KoreanTranslator
from disco.vad.silero import SileroVAD


class Runtime:
    """Owns the worker lifecycle and Final → EnrichedFinal enrichment.

    Callers construct a Runtime with the loaded models and a bus, attach
    their own subscribers (Console, WebSocket, etc.), then call ``start``
    with an ``AudioSource``. The runtime registers the three workers as
    audio consumers, wires the turn events to the transcriber worker,
    and hooks ``Final`` for speaker/translation enrichment.
    """

    # Wait for sortformer to catch up before resolving the final speaker.
    ENRICHMENT_GRACE_S = 0.3
    METRICS_INTERVAL_S = 10.0

    def __init__(
        self,
        *,
        bus: EventBus,
        vad: SileroVAD,
        transcriber: Transcriber,
        diarizer: Diarizer,
        translator: KoreanTranslator | None = None,
        language: str = "English",
        sample_rate: int = 16000,
        silence_duration: float = 0.5,
        min_utterance_duration: float = 0.5,
        speaker_change_hold: float = 0.4,
    ):
        self.bus = bus
        self.vad = vad
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.translator = translator
        self.language = language
        self.sample_rate = sample_rate

        # Route diarizer overflows through the bus so callers see them
        # alongside the worker-emitted ones.
        diarizer.on_overflow = lambda depth: bus.publish(
            QueueOverflow(component="diarizer", depth=depth)
        )

        self.turn_detector = TurnDetector(
            vad=vad,
            diarizer=diarizer,
            bus=bus,
            sample_rate=sample_rate,
            silence_duration=silence_duration,
            min_utterance_duration=min_utterance_duration,
            speaker_change_hold=speaker_change_hold,
        )
        self.transcriber_worker = TranscriberWorker(
            transcriber=transcriber,
            bus=bus,
            sample_rate=sample_rate,
        )

        self._source: AudioSource | None = None
        self._wired = False
        self._metrics_thread: threading.Thread | None = None
        self._metrics_stop = threading.Event()
        self._metrics_enabled = os.environ.get("DISCO_METRICS") == "1"

    def start(self, source: AudioSource) -> None:
        self._source = source

        self.diarizer.start()
        self.transcriber_worker.start()
        self.turn_detector.start()

        source.subscribe(self.diarizer)
        source.subscribe(self.turn_detector)
        source.subscribe(self.transcriber_worker)

        if not self._wired:
            self.bus.subscribe(SpeechStart, self._on_speech_start)
            self.bus.subscribe(SpeechEnd, self._on_speech_end)
            self.bus.subscribe(SpeakerChange, self._on_speaker_change)
            self.bus.subscribe(Final, self._on_final)
            self.bus.subscribe(QueueOverflow, self._on_overflow)
            self._wired = True

        source.start()

        if self._metrics_enabled and self._metrics_thread is None:
            self._metrics_stop.clear()
            self._metrics_thread = threading.Thread(
                target=self._metrics_loop, daemon=True
            )
            self._metrics_thread.start()

    def stop(self) -> None:
        if self._metrics_thread is not None:
            self._metrics_stop.set()
            self._metrics_thread.join(timeout=2.0)
            self._metrics_thread = None
        if self._source is not None:
            self._source.stop()
            self._source = None
        self.turn_detector.stop()
        self.transcriber_worker.stop()
        self.diarizer.stop()

    # ---- bus handlers ----

    def _on_speech_start(self, event: SpeechStart) -> None:
        self.transcriber_worker.open_session(event.t)

    def _on_speech_end(self, event: SpeechEnd) -> None:
        self.transcriber_worker.close_session(event.t)

    def _on_speaker_change(self, event: SpeakerChange) -> None:
        # Force-finalize the current session; the next SpeechStart will
        # open a new one. The drain race in TranscriberWorker handles any
        # audio that arrives between close and the next open.
        self.transcriber_worker.close_session(event.t)

    def _on_final(self, event: Final) -> None:
        # Sortformer emits segments with some lag; wait briefly so the
        # last words of the utterance are covered by the speaker query.
        threading.Timer(self.ENRICHMENT_GRACE_S, self._enrich, args=(event,)).start()

    def _enrich(self, event: Final) -> None:
        t_start, t_end = event.span
        speaker = self.diarizer.dominant_speaker_in(t_start, t_end)

        translation: str | None = None
        if self.translator is not None:
            source_lang = LANG_CODE_MAP.get(self.language.lower(), "en")
            try:
                translation = self.translator.translate(event.text, source_lang)
            except Exception as exc:
                print(f"Translation error: {exc}")

        self.bus.publish(
            EnrichedFinal(
                text=event.text,
                span=event.span,
                speaker=int(speaker) if speaker is not None else None,
                translation=translation,
            )
        )

    def _on_overflow(self, event: QueueOverflow) -> None:
        print(f"[backpressure] {event.component} queue dropped chunk (depth={event.depth})")

    def _metrics_loop(self) -> None:
        while not self._metrics_stop.wait(self.METRICS_INTERVAL_S):
            td_depth = self.turn_detector._queue.qsize()
            tw_depth = self.transcriber_worker._queue.qsize()
            diar_depth = self.diarizer._queue.qsize()
            print(
                f"[metrics] queues: turn={td_depth} transcriber={tw_depth} "
                f"diarizer={diar_depth} t={time.strftime('%H:%M:%S')}"
            )
