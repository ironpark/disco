"""Voxtral streaming session owned by a dedicated worker thread."""

import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from disco.asr.hallucination import is_hallucination
from disco.audio.frame import AudioFrame
from disco.runtime.debug import log as debug_log
from disco.runtime.events import (
    EventBus,
    Final,
    Interim,
    QueueOverflow,
    WorkerBackpressure,
)

if TYPE_CHECKING:
    from disco.asr.transcriber import StreamingTranscription, Transcriber


@dataclass(frozen=True)
class _Open:
    t: float
    utterance_id: int


@dataclass(frozen=True)
class _Close:
    t: float
    utterance_id: int


@dataclass
class _SessionState:
    session: Any
    start_t: float
    end_t: float
    utterance_id: int
    last_emit_text: str = ""


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
                     TW queue *before* the _Open emitted by the coordinator
                     for the same chunk) aren't lost.
        recording  — feed chunks to the session and run step() between gets.
        draining   — session.close() done; pump step() until done. Any audio
                     arriving here is also parked in ``pending`` and replayed
                     into the next session for the SpeechEnd/SpeechStart race.

    ``pending`` is a bounded deque so long silences don't grow it forever.
    """

    # ~1 s of pre-/post-utterance audio is enough to cover normal Coordinator lag.
    PENDING_MAXLEN = 10

    def __init__(
        self,
        transcriber: "Transcriber",
        bus: EventBus,
        sample_rate: int = 16000,
        max_queue: int = 200,
        max_draining_sessions: int = 3,
    ):
        self.transcriber = transcriber
        self.bus = bus
        self.sample_rate = sample_rate
        self.max_queue = max_queue
        self.max_draining_sessions = max_draining_sessions

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False
        self._ready = threading.Event()
        self._load_error: Exception | None = None
        self._state_lock = threading.Lock()
        self._draining_count = 0
        self._oldest_drain_age = 0.0

    # ---- producer-side API (any thread) ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=120.0):
            self._running = False
            raise TimeoutError("Timed out loading ASR model")
        if self._load_error is not None:
            raise RuntimeError("ASR model failed to load") from self._load_error

    def stop(self) -> None:
        if self._thread is None:
            return
        self._running = False
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(_STOP)
        self._thread.join(timeout=2.0)
        self._thread = None

    def feed(self, frame: AudioFrame | np.ndarray) -> None:
        self._enqueue(frame, kind="audio")

    def open_session(self, t: float, utterance_id: int) -> None:
        self._enqueue(_Open(t=t, utterance_id=utterance_id), kind="open")

    def close_session(self, t: float, utterance_id: int) -> None:
        self._enqueue(_Close(t=t, utterance_id=utterance_id), kind="close")

    def snapshot(self) -> dict[str, float | int]:
        with self._state_lock:
            return {
                "queue_depth": self._queue.qsize(),
                "draining_count": self._draining_count,
                "oldest_drain_age": self._oldest_drain_age,
            }

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
        try:
            self.transcriber.load()
        except Exception as exc:
            self._load_error = exc
            self._ready.set()
            return
        self._ready.set()

        recording: _SessionState | None = None
        draining: list[_SessionState] = []
        # Sliding buffer of recent audio. Captures chunks that arrive in
        # this worker's queue before the corresponding _Open from the
        # coordinator (the speech-trigger chunk almost always lands here
        # first) and chunks that arrive while we're still draining the
        # previous session.
        pending: deque[AudioFrame | np.ndarray] = deque(maxlen=self.PENDING_MAXLEN)

        def update_state() -> None:
            now_end = recording.end_t if recording is not None else 0.0
            if draining:
                if now_end == 0.0:
                    now_end = max(state.end_t for state in draining)
                oldest = max(0.0, now_end - min(state.end_t for state in draining))
            else:
                oldest = 0.0
            with self._state_lock:
                self._draining_count = len(draining)
                self._oldest_drain_age = oldest

        def open_now(open_evt: _Open) -> None:
            nonlocal recording
            recording = _SessionState(
                session=self.transcriber.start_session(),
                start_t=open_evt.t,
                end_t=open_evt.t,
                utterance_id=open_evt.utterance_id,
            )
            replayed = len(pending)
            for chunk in pending:
                recording.session.feed(
                    chunk.samples if isinstance(chunk, AudioFrame) else chunk
                )
            pending.clear()
            debug_log(
                "tw",
                f"_Open t={open_evt.t:.2f}",
                f"utt={open_evt.utterance_id}",
                f"replayed={replayed} chunks",
                f"draining={len(draining)}",
            )
            update_state()

        def publish_interim(state: _SessionState) -> None:
            if state.session.step() and state.session.text != state.last_emit_text:
                state.last_emit_text = state.session.text
                self.bus.publish(
                    Interim(
                        text=state.session.text,
                        span=(state.start_t, state.end_t),
                        utterance_id=state.utterance_id,
                    )
                )

        def finish_drained(state: _SessionState) -> None:
            text = state.session.text.strip()
            debug_log(
                "tw",
                f"drained text={text[:40]!r}",
                f"span=({state.start_t:.2f},{state.end_t:.2f})",
                f"utt={state.utterance_id}",
                f"buffered={len(pending)}",
                f"remaining_drains={len(draining)}",
            )
            if text and not is_hallucination(text):
                self.bus.publish(
                    Final(
                        text=text,
                        span=(state.start_t, state.end_t),
                        utterance_id=state.utterance_id,
                    )
                )

        def add_draining(state: _SessionState) -> None:
            draining.append(state)
            if len(draining) > self.max_draining_sessions:
                dropped = draining.pop(0)
                debug_log(
                    "tw",
                    "draining cap reached",
                    f"cap={self.max_draining_sessions}",
                    f"dropped_utt={dropped.utterance_id}",
                )
                self.bus.publish(
                    WorkerBackpressure(
                        component="transcriber",
                        reason="draining_cap",
                        depth=len(draining) + 1,
                    )
                )
                try:
                    dropped.session.drain()
                except Exception as exc:
                    debug_log("tw", f"dropped drain failed: {exc}")
                finish_drained(dropped)
            update_state()

        while self._running:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                item = None

            if item is _STOP:
                break

            if isinstance(item, _Open):
                if recording is None:
                    open_now(item)
                else:
                    debug_log(
                        "tw",
                        f"_Open ignored t={item.t:.2f}",
                        f"utt={item.utterance_id}",
                        "state=recording",
                    )

            elif isinstance(item, _Close):
                if recording is not None and recording.utterance_id == item.utterance_id:
                    recording.session.close()
                    recording.end_t = max(recording.end_t, item.t)
                    add_draining(recording)
                    debug_log(
                        "tw",
                        f"_Close t={item.t:.2f}",
                        f"span=({recording.start_t:.2f},{recording.end_t:.2f})",
                        f"utt={recording.utterance_id}",
                        f"draining={len(draining)}",
                    )
                    recording = None
                    update_state()
                elif recording is not None:
                    debug_log(
                        "tw",
                        f"_Close ignored t={item.t:.2f}",
                        f"utt={item.utterance_id}",
                        f"recording_utt={recording.utterance_id}",
                    )
                # if idle: ignore
            elif isinstance(item, (AudioFrame, np.ndarray)):
                chunk = item.samples if isinstance(item, AudioFrame) else item
                if recording is not None:
                    recording.session.feed(chunk)
                    if isinstance(item, AudioFrame):
                        recording.end_t = max(recording.end_t, item.t_end)
                    else:
                        recording.end_t += len(chunk) / self.sample_rate
                else:
                    # No open recording session yet: park for the next session.
                    pending.append(item)

            if recording is not None:
                publish_interim(recording)

            for state in list(draining):
                # For backends that don't produce text during recording
                # (e.g. Qwen3-ASR), this is where text first appears.
                publish_interim(state)
                if state.session.done:
                    draining.remove(state)
                    finish_drained(state)
                    update_state()

        # Drain on shutdown.
        if recording is not None:
            recording.session.close()
            add_draining(recording)
        for state in draining:
            state.session.drain()
            finish_drained(state)
        with self._state_lock:
            self._draining_count = 0
            self._oldest_drain_age = 0.0
