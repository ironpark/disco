"""Voxtral streaming session owned by a dedicated worker thread."""

import queue
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from disco.asr.transcriber import StreamingTranscription, Transcriber, is_hallucination
from disco.runtime.debug import log as debug_log
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
        idle       — no session yet. Chunks arriving here are parked in the
                     bounded ``pending`` buffer. _Open replays them so the
                     first chunks of each utterance (which always land in the
                     TW queue *before* the _Open emitted by TurnDetector for
                     the same chunk) aren't lost.
        recording  — feed chunks to the session and run step() between gets.
        draining   — session.close() done; pump step() until done. Any audio
                     arriving here is also parked in ``pending`` and replayed
                     into the next session for the SpeechEnd/SpeechStart race.

    ``pending`` is a bounded deque so long silences don't grow it forever.
    """

    # ~1 s of pre-/post-utterance audio is enough to cover normal TurnDetector lag.
    PENDING_MAXLEN = 10

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
        # Sliding buffer of recent audio. Captures chunks that arrive in
        # this worker's queue before the corresponding _Open from the
        # coordinator (the speech-trigger chunk almost always lands here
        # first) and chunks that arrive while we're still draining the
        # previous session.
        pending: deque[np.ndarray] = deque(maxlen=self.PENDING_MAXLEN)
        # _Open received while not idle. Applied once we transition to idle
        # so a tight close/open pair (typical for SpeakerChange) doesn't
        # silently drop the new session.
        deferred_open: _Open | None = None

        def open_now(open_evt: _Open) -> None:
            nonlocal session, session_start_t, session_end_t
            nonlocal last_emit_text, state
            session = self.transcriber.start_session()
            session_start_t = open_evt.t
            session_end_t = open_evt.t
            last_emit_text = ""
            state = "recording"
            replayed = len(pending)
            for chunk in pending:
                session.feed(chunk)
            pending.clear()
            debug_log(
                "tw",
                f"_Open t={open_evt.t:.2f}",
                f"replayed={replayed} chunks",
            )

        while self._running:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                item = None

            if item is _STOP:
                break

            if isinstance(item, _Open):
                if state == "idle":
                    open_now(item)
                else:
                    # Recording or draining — keep the most recent open and
                    # apply it once draining completes.
                    deferred_open = item
                    debug_log(
                        "tw",
                        f"_Open deferred t={item.t:.2f}",
                        f"state={state}",
                    )

            elif isinstance(item, _Close):
                if state == "recording" and session is not None:
                    session.close()
                    session_end_t = max(session_end_t, item.t)
                    state = "draining"
                    debug_log(
                        "tw",
                        f"_Close t={item.t:.2f}",
                        f"span=({session_start_t:.2f},{session_end_t:.2f})",
                    )
                # if idle: ignore; if draining: already closing
            elif isinstance(item, np.ndarray):
                chunk = item
                if state == "recording" and session is not None:
                    session.feed(chunk)
                    session_end_t += len(chunk) / self.sample_rate
                else:
                    # idle or draining: park for the (next) session.
                    pending.append(chunk)

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
                    debug_log(
                        "tw",
                        f"drained text={text[:40]!r}",
                        f"span=({session_start_t:.2f},{session_end_t:.2f})",
                        f"buffered={len(pending)}",
                    )
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
                    if deferred_open is not None:
                        pending_open = deferred_open
                        deferred_open = None
                        open_now(pending_open)

        # Drain on shutdown.
        if session is not None:
            session.close()
            session.drain()
            text = session.text.strip()
            if text and not is_hallucination(text):
                self.bus.publish(
                    Final(text=text, span=(session_start_t, session_end_t))
                )
