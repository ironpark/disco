"""Coalesced interim translation stage."""

import threading
import time

from disco.config import LANG_CODE_MAP
from disco.runtime.debug import log as debug_log
from disco.runtime.events import EnrichedInterim, EventBus, Interim
from disco.translation.korean import KoreanTranslator


class InterimTranslator:
    """Translate the latest interim text at a bounded cadence."""

    def __init__(
        self,
        *,
        bus: EventBus,
        translator: KoreanTranslator | None,
        language: str = "English",
        interval_s: float = 1.0,
        min_chars: int = 8,
    ):
        self.bus = bus
        self.translator = translator
        self.language = language
        self.interval_s = interval_s
        self.min_chars = min_chars

        self._lock = threading.Lock()
        self._updated = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: Interim | None = None
        self._last_text = ""
        self._last_emit_at = 0.0

    def start(self) -> None:
        if self.translator is None or self._thread is not None:
            return
        self._stop.clear()
        self._updated.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._updated.set()
        self._thread.join(timeout=3.0)
        self._thread = None

    def submit(self, event: Interim) -> None:
        if self.translator is None:
            return
        text = event.text.strip()
        if len(text) < self.min_chars:
            return
        with self._lock:
            self._latest = event
        self._updated.set()

    def _worker(self) -> None:
        while not self._stop.is_set():
            self._updated.wait()
            self._updated.clear()
            if self._stop.is_set():
                break

            elapsed = time.monotonic() - self._last_emit_at
            if elapsed < self.interval_s and self._stop.wait(self.interval_s - elapsed):
                break

            with self._lock:
                event = self._latest
                self._latest = None
            if event is None:
                continue

            text = event.text.strip()
            if text == self._last_text:
                continue

            source_lang = LANG_CODE_MAP.get(self.language.lower(), "en")
            translation = self.translator.translate(text, source_lang)
            self._last_text = text
            self._last_emit_at = time.monotonic()
            debug_log(
                "interim",
                f"translated span=({event.span[0]:.2f},{event.span[1]:.2f})",
                f"text={text[:40]!r}",
            )
            self.bus.publish(
                EnrichedInterim(
                    text=event.text,
                    span=event.span,
                    speaker=event.speaker,
                    translation=translation,
                )
            )
