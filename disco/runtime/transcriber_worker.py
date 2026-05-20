"""Voxtral streaming session owned by a dedicated worker thread."""

import queue
import threading
from dataclasses import dataclass

import numpy as np

from disco.asr.transcriber import StreamingTranscription, Transcriber, is_hallucination
from disco.runtime.events import EventBus, Final, Interim, QueueOverflow


@dataclass(frozen=True)
class _Open:
    t: float


@dataclass(frozen=True)
class _Close:
    t: float


_STOP = object()


class TranscriberWorker:
    """Owns a Voxtral streaming session on its own thread.

    External interface (all thread-safe):
        ``feed(chunk)``         AudioConsumer; enqueue raw audio.
        ``open_session(t)``     Begin a new utterance at diarizer-time ``t``.
        ``close_session(t)``    Finalize the current utterance at ``t``.
        ``start()`` / ``stop()`` Lifecycle.

    State machine (worker-thread only):
        idle       — no session; audio chunks are dropped.
        recording  — feed chunks to the session and run step() between gets.
        draining   — session.close() done; pump step() until done. Any audio
                     arriving here is parked in ``pending`` and replayed into
                     the next session so we don't drop the first words of the
                     following utterance when SpeechEnd and the next
                     SpeechStart fire close together.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        bus: EventBus,
        sample_rate: int = 16000,
        max_queue: int = 200,
    ):
        self.transcriber = transcriber
        self.bus = bus
        self.sample_rate = sample_rate
        self.max_queue = max_queue

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False
        self._ready = threading.Event()

    # ---- producer-side API (any thread) ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready.wait()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._running = False
        self._queue.put(_STOP)
        self._thread.join(timeout=2.0)
        self._thread = None

    def feed(self, chunk: np.ndarray) -> None:
        self._enqueue(chunk, kind="audio")

    def open_session(self, t: float) -> None:
        self._enqueue(_Open(t=t), kind="open")

    def close_session(self, t: float) -> None:
        self._enqueue(_Close(t=t), kind="close")

    def _enqueue(self, item, kind: str) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop the oldest item to make room; warn via event bus.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass
            self.bus.publish(
                QueueOverflow(component="transcriber", depth=self._queue.qsize())
            )

    # ---- worker thread ----

    def _worker(self) -> None:
        # Voxtral and its session live entirely on this thread so MLX
        # thread-local streams stay consistent across feed / step / close.
        self.transcriber.load()
        self._ready.set()

        state: str = "idle"
        session: StreamingTranscription | None = None
        session_start_t: float = 0.0
        session_end_t: float = 0.0
        last_emit_text: str = ""
        pending: list[np.ndarray] = []  # audio parked while draining

        while self._running:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                item = None

            if item is _STOP:
                break

            if isinstance(item, _Open):
                if state != "idle":
                    # Already recording: ignore stray open. Caller is expected
                    # to close before opening a new session.
                    pass
                else:
                    session = self.transcriber.start_session()
                    session_start_t = item.t
                    session_end_t = item.t
                    last_emit_text = ""
                    state = "recording"
                    # Replay any audio that arrived while we were idle/draining.
                    for chunk in pending:
                        session.feed(chunk)
                        session_end_t += len(chunk) / self.sample_rate
                    pending.clear()

            elif isinstance(item, _Close):
                if state == "recording" and session is not None:
                    session.close()
                    session_end_t = max(session_end_t, item.t)
                    state = "draining"
                # if idle: ignore; if draining: already closing
            elif isinstance(item, np.ndarray):
                chunk = item
                if state == "recording" and session is not None:
                    session.feed(chunk)
                    session_end_t += len(chunk) / self.sample_rate
                elif state == "draining":
                    # Park for the next session.
                    pending.append(chunk)
                # idle: drop

            # Progress decoding regardless of whether we just consumed an item.
            if state == "recording" and session is not None:
                if session.step() and session.text != last_emit_text:
                    last_emit_text = session.text
                    self.bus.publish(
                        Interim(
                            text=session.text,
                            span=(session_start_t, session_end_t),
                        )
                    )

            elif state == "draining" and session is not None:
                # Pump until session reports done.
                session.step()
                if session.done:
                    text = session.text.strip()
                    if text and not is_hallucination(text):
                        self.bus.publish(
                            Final(
                                text=text,
                                span=(session_start_t, session_end_t),
                            )
                        )
                    session = None
                    last_emit_text = ""
                    state = "idle"

        # Drain on shutdown.
        if session is not None:
            session.close()
            session.drain()
            text = session.text.strip()
            if text and not is_hallucination(text):
                self.bus.publish(
                    Final(text=text, span=(session_start_t, session_end_t))
                )
