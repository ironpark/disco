"""Final transcript enrichment stage."""

import queue
import threading

from disco.config import LANG_CODE_MAP
from disco.diar.sortformer import Diarizer
from disco.runtime.debug import log as debug_log
from disco.runtime.events import EnrichedFinal, EventBus, Final
from disco.translation.korean import KoreanTranslator


_STOP = object()


class FinalEnricher:
    """Serially attaches speaker labels and optional translation to finals."""

    def __init__(
        self,
        *,
        bus: EventBus,
        diarizer: Diarizer,
        translator: KoreanTranslator | None = None,
        language: str = "English",
        grace_s: float = 0.3,
    ):
        self.bus = bus
        self.diarizer = diarizer
        self.translator = translator
        self.language = language
        self.grace_s = grace_s
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._queue.put(_STOP)
        self._thread.join(timeout=5.0)
        self._running = False
        self._thread = None

    def submit(self, event: Final) -> None:
        if self._running:
            self._queue.put(event)

    def _worker(self) -> None:
        while True:
            event = self._queue.get()
            if event is _STOP:
                break
            if self.grace_s > 0 and not self._stop.is_set():
                self._stop.wait(self.grace_s)
            self._enrich(event)

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
