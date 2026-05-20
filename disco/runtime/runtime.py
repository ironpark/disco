"""Wire AudioSource → workers → EventBus + enrich Final into EnrichedFinal."""

import os
import threading
import time

from disco.asr.transcriber import Transcriber
from disco.audio.source import AudioSource
from disco.config import LANG_CODE_MAP
from disco.diar.sortformer import Diarizer
from disco.runtime.coordinator import Coordinator
from disco.runtime.debug import log as debug_log
from disco.runtime.events import (
    EnrichedFinal,
    EventBus,
    Final,
    QueueOverflow,
    SpeakerActivity,
    SpeakerChange,
    SpeechEnd,
    SpeechStart,
)
from disco.runtime.transcriber_worker import TranscriberWorker
from disco.translation.korean import KoreanTranslator


class Runtime:
    """Owns the worker lifecycle and Final → EnrichedFinal enrichment.

    Callers construct a Runtime with the loaded models and a bus, attach
    their own subscribers (Console, WebSocket, etc.), then call ``start``
    with an ``AudioSource``. The runtime wires the diarizer as the sole
    audio consumer that drives turn detection: each processed chunk emits
    SpeakerActivity, the Coordinator turns that into SpeechStart/End/
    SpeakerChange, and the TranscriberWorker opens/closes its session
    accordingly. The diarizer also receives the audio for its own
    cumulative segment store used at finalize time.
    """

    # Wait for sortformer to catch up before resolving the final speaker.
    ENRICHMENT_GRACE_S = 0.3
    METRICS_INTERVAL_S = 10.0

    def __init__(
        self,
        *,
        bus: EventBus,
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
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.translator = translator
        self.language = language
        self.sample_rate = sample_rate

        # Plumb diarizer callbacks through the bus.
        diarizer.on_overflow = lambda depth: bus.publish(
            QueueOverflow(component="diarizer", depth=depth)
        )
        diarizer.on_activity = lambda t_start, t_end, primary, all_spks: bus.publish(
            SpeakerActivity(
                t_start=t_start,
                t_end=t_end,
                primary_speaker=primary,
                all_speakers=all_spks,
            )
        )

        # Translate the configured durations into chunk counts at 100 ms /
        # chunk — matches AudioSource's default block size.
        block_s = 0.1
        self.coordinator = Coordinator(
            bus=bus,
            silence_chunks_for_end=max(1, int(silence_duration / block_s)),
            min_utterance_chunks=max(1, int(min_utterance_duration / block_s)),
            speaker_change_chunks=max(1, int(speaker_change_hold / block_s)),
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

        source.subscribe(self.diarizer)
        source.subscribe(self.transcriber_worker)

        if not self._wired:
            self.bus.subscribe(SpeakerActivity, self.coordinator.on_activity)
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
        self.transcriber_worker.stop()
        self.diarizer.stop()

    # ---- bus handlers ----

    def _on_speech_start(self, event: SpeechStart) -> None:
        self.transcriber_worker.open_session(event.t)

    def _on_speech_end(self, event: SpeechEnd) -> None:
        self.transcriber_worker.close_session(event.t)

    def _on_speaker_change(self, event: SpeakerChange) -> None:
        # Close the previous speaker's session and open a fresh one at the
        # same instant. TranscriberWorker's deferred-open slot handles the
        # case where the new _Open arrives while the previous session is
        # still draining.
        self.transcriber_worker.close_session(event.t)
        self.transcriber_worker.open_session(event.t)

    def _on_final(self, event: Final) -> None:
        # Sortformer emits segments with some lag; wait briefly so the
        # last words of the utterance are covered by the speaker query.
        threading.Timer(self.ENRICHMENT_GRACE_S, self._enrich, args=(event,)).start()

    def _enrich(self, event: Final) -> None:
        t_start, t_end = event.span
        speaker = self.diarizer.dominant_speaker_in(t_start, t_end)
        diar_now = self.diarizer.elapsed_seconds()
        debug_log(
            "enrich",
            f"Final span=({t_start:.2f},{t_end:.2f})",
            f"speaker={'S' + str(speaker) if speaker is not None else '?'}",
            f"diar_now={diar_now:.2f}s",
            f"text={event.text[:40]!r}",
        )

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
            tw_depth = self.transcriber_worker._queue.qsize()
            diar_depth = self.diarizer._queue.qsize()
            print(
                f"[metrics] queues: transcriber={tw_depth} "
                f"diarizer={diar_depth} t={time.strftime('%H:%M:%S')}"
            )
