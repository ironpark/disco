"""Prioritized translation stage for final and interim transcript events."""

import queue
import threading
import time
from dataclasses import dataclass
from itertools import count

from disco.config import LANG_CODE_MAP
from disco.runtime.debug import log as debug_log
from disco.runtime.events import (
    EnrichedFinal,
    EnrichedInterim,
    EventBus,
    Interim,
    LabeledFinal,
)
from disco.translation.korean import KoreanTranslator


_STOP = object()


@dataclass(frozen=True)
class _InterimTranslationJob:
    event: Interim

    @property
    def utterance_id(self) -> int:
        return self.event.utterance_id

    @property
    def text(self) -> str:
        return self.event.text.strip()

    @property
    def key(self) -> tuple[int, str]:
        return (self.utterance_id, self.text)


@dataclass(frozen=True)
class _FinalTranslationJob:
    event: LabeledFinal


class TranslationService:
    """Translate finals before coalesced interim text.

    Finals are user-visible durable records, so they always take priority.
    Interim work keeps only the latest candidate and is discarded if the
    utterance/text changed while translation was running.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        translator: KoreanTranslator | None,
        language: str = "English",
        interim_interval_s: float = 1.0,
        interim_min_chars: int = 8,
    ):
        self.bus = bus
        self.translator = translator
        self.language = language
        self.interim_interval_s = interim_interval_s
        self.interim_min_chars = interim_min_chars

        self._finals: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = count()
        self._lock = threading.Lock()
        self._updated = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_interims: dict[int, _InterimTranslationJob] = {}
        self._finalized_utterance_ids: set[int] = set()
        self._last_interim_key: tuple[int, str] | None = None
        self._last_interim_emit_at = 0.0

    def start(self) -> None:
        if self._thread is not None:
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
        self._finals.put((0, next(self._seq), _STOP))
        self._thread.join(timeout=5.0)
        self._thread = None

    def submit_interim(self, event: Interim) -> None:
        if self.translator is None:
            return
        text = event.text.strip()
        if len(text) < self.interim_min_chars:
            return
        job = _InterimTranslationJob(event=event)
        with self._lock:
            if event.utterance_id in self._finalized_utterance_ids:
                debug_log("translate", f"drop finalized interim utt={event.utterance_id}")
                return
            self._pending_interims[job.utterance_id] = job
        self._updated.set()

    def submit_final(self, event: LabeledFinal) -> None:
        if self.translator is None:
            self._publish_final(event, translation=None)
            return
        with self._lock:
            for utterance_id in event.utterance_ids:
                self._finalized_utterance_ids.add(utterance_id)
                self._pending_interims.pop(utterance_id, None)
        self._finals.put((0, next(self._seq), _FinalTranslationJob(event=event)))
        self._updated.set()

    def _worker(self) -> None:
        while not self._stop.is_set():
            if self._drain_one_final():
                continue

            self._updated.wait()
            self._updated.clear()
            if self._stop.is_set():
                break

            elapsed = time.monotonic() - self._last_interim_emit_at
            if elapsed < self.interim_interval_s:
                if self._stop.wait(self.interim_interval_s - elapsed):
                    break
                if self._drain_one_final():
                    continue

            event = self._take_latest_interim()
            if event is None:
                continue
            if self._drain_one_final():
                with self._lock:
                    if (
                        event.utterance_id not in self._finalized_utterance_ids
                        and event.utterance_id not in self._pending_interims
                    ):
                        job = _InterimTranslationJob(event=event)
                        self._pending_interims[job.utterance_id] = job
                        self._updated.set()
                continue
            self._translate_interim(event)
            with self._lock:
                if self._pending_interims:
                    self._updated.set()

        while self._drain_one_final(block=False):
            pass

    def _drain_one_final(self, *, block: bool = False) -> bool:
        try:
            _, _, item = self._finals.get(block=block, timeout=0.05 if block else 0)
        except queue.Empty:
            return False
        if item is _STOP:
            return False

        event = item.event
        source_lang = LANG_CODE_MAP.get(self.language.lower(), "en")
        translation: str | None = None
        try:
            translation = self.translator.translate(event.text, source_lang)
        except Exception as exc:
            print(f"Translation error: {exc}")
        debug_log(
            "translate",
            f"final utt={event.utterance_id}",
            f"span=({event.span[0]:.2f},{event.span[1]:.2f})",
            f"text={event.text[:40]!r}",
        )
        self._publish_final(event, translation=translation)
        return True

    def _take_latest_interim(self) -> Interim | None:
        with self._lock:
            while self._pending_interims:
                job = min(
                    self._pending_interims.values(),
                    key=lambda item: item.event.span[1],
                )
                self._pending_interims.pop(job.utterance_id, None)
                if job.utterance_id in self._finalized_utterance_ids:
                    continue
                if job.key != self._last_interim_key:
                    return job.event
            return None

    def _translate_interim(self, event: Interim) -> None:
        text = event.text.strip()
        key = (event.utterance_id, text)
        source_lang = LANG_CODE_MAP.get(self.language.lower(), "en")
        with self._lock:
            if event.utterance_id in self._finalized_utterance_ids:
                debug_log("translate", f"drop finalized interim utt={event.utterance_id}")
                return
        translation = self.translator.translate(text, source_lang)
        with self._lock:
            if event.utterance_id in self._finalized_utterance_ids:
                debug_log("translate", f"drop finalized interim utt={event.utterance_id}")
                return
            current = self._pending_interims.get(event.utterance_id)
        if current is not None:
            if current.key != key:
                debug_log("translate", f"drop stale interim utt={event.utterance_id}")
                return

        self._last_interim_key = key
        self._last_interim_emit_at = time.monotonic()
        debug_log(
            "translate",
            f"interim utt={event.utterance_id}",
            f"span=({event.span[0]:.2f},{event.span[1]:.2f})",
            f"text={text[:40]!r}",
        )
        self.bus.publish(
            EnrichedInterim(
                text=event.text,
                span=event.span,
                utterance_id=event.utterance_id,
                speaker=event.speaker,
                translation=translation,
            )
        )

    def _publish_final(
        self, event: LabeledFinal, *, translation: str | None
    ) -> None:
        self.bus.publish(
            EnrichedFinal(
                text=event.text,
                ref=event.ref,
                translation=translation,
            )
        )
