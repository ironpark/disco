"""Final transcript enrichment stage."""

import queue
import threading

from disco.diar.sortformer import Diarizer
from disco.runtime.debug import log as debug_log
from disco.runtime.events import EventBus, Final, LabeledFinal, TurnRef


_STOP = object()


class FinalEnricher:
    """Serially attaches speaker labels and merges adjacent finals."""

    def __init__(
        self,
        *,
        bus: EventBus,
        diarizer: Diarizer,
        language: str = "English",
        grace_s: float = 0.3,
        merge_grace_s: float = 1.0,
    ):
        self.bus = bus
        self.diarizer = diarizer
        self.language = language
        self.grace_s = grace_s
        self.merge_grace_s = merge_grace_s
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._stop = threading.Event()
        self._pending: LabeledFinal | None = None

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
            timeout = self._pending_timeout()
            try:
                event = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._flush_pending(reason="merge_timeout")
                continue
            if event is _STOP:
                self._flush_pending(reason="stop")
                break
            if self.grace_s > 0 and not self._stop.is_set():
                self._stop.wait(self.grace_s)
            enriched = self._enrich(event)
            self._handle_enriched(enriched)

    def _pending_timeout(self) -> float | None:
        if self._pending is None:
            return None
        remaining = self.merge_grace_s - (
            self.diarizer.elapsed_seconds() - self._pending.span[1]
        )
        return max(0.0, remaining)

    def _enrich(self, event: Final) -> LabeledFinal:
        t_start, t_end = event.span
        speaker = self.diarizer.dominant_speaker_in(t_start, t_end)
        diar_now = self.diarizer.elapsed_seconds()
        debug_log(
            "enrich",
            f"Final span=({t_start:.2f},{t_end:.2f})",
            f"utt={event.utterance_id}",
            f"speaker={'S' + str(speaker) if speaker is not None else '?'}",
            f"diar_now={diar_now:.2f}s",
            f"text={event.text[:40]!r}",
        )

        return LabeledFinal(
            text=event.text,
            ref=TurnRef.single(
                utterance_id=event.utterance_id,
                span=event.span,
                speaker=int(speaker) if speaker is not None else None,
            ),
        )

    def _handle_enriched(self, event: LabeledFinal) -> None:
        if self._pending is None:
            self._pending = event
            debug_log(
                "enrich",
                f"Final pending span=({event.span[0]:.2f},{event.span[1]:.2f})",
                f"utt={event.utterance_id}",
                f"speaker={'S' + str(event.speaker) if event.speaker is not None else '?'}",
            )
            return

        gap = event.span[0] - self._pending.span[1]
        if (
            self._pending.speaker is not None
            and event.speaker == self._pending.speaker
            and 0 <= gap <= self.merge_grace_s
        ):
            debug_log(
                "enrich",
                f"Final merged gap={gap:.2f}s",
                f"speaker=S{event.speaker}",
                f"prev=({self._pending.span[0]:.2f},{self._pending.span[1]:.2f})",
                f"next=({event.span[0]:.2f},{event.span[1]:.2f})",
            )
            self._pending = LabeledFinal(
                text=self._join_text(self._pending.text, event.text),
                ref=self._pending.ref.merged_with(event.ref, speaker=event.speaker),
            )
            return

        self._flush_pending(reason=f"next_gap={gap:.2f}s")
        self._pending = event

    def _flush_pending(self, *, reason: str) -> None:
        if self._pending is None:
            return
        event = self._pending
        self._pending = None
        debug_log(
            "enrich",
            f"Final publish span=({event.span[0]:.2f},{event.span[1]:.2f})",
            f"utt={event.utterance_id}",
            f"speaker={'S' + str(event.speaker) if event.speaker is not None else '?'}",
            f"reason={reason}",
            f"text={event.text[:40]!r}",
        )
        self.bus.publish(event)

    def _join_text(self, left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        if self.language.lower() in {"japanese", "chinese", "korean"}:
            return f"{left}{right}"
        return f"{left} {right}"
