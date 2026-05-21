"""Voxtral streaming session owned by a dedicated worker thread."""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from disco.asr.hallucination import is_hallucination
from disco.audio.frame import AudioFrame, AudioRingBuffer
from disco.runtime.debug import log as debug_log
from disco.runtime.events import (
    EventBus,
    Final,
    FinalDiscarded,
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
    speaker: int | None = None


@dataclass(frozen=True)
class _Close:
    t: float
    utterance_id: int


@dataclass(frozen=True)
class _SpeakerBind:
    utterance_id: int
    speaker: int


class _SessionPhase(Enum):
    RECORDING = "recording"
    FINALIZING = "finalizing"

    @property
    def emits_interim(self) -> bool:
        return self is _SessionPhase.RECORDING


@dataclass
class _SessionState:
    session: Any
    start_t: float
    end_t: float
    utterance_id: int
    phase: _SessionPhase
    speaker: int | None = None
    last_emit_text: str = ""
    finalizing_started_at: float | None = None


class _CompletedTranscription:
    def __init__(self, text: str):
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    @property
    def done(self) -> bool:
        return True

    def feed(self, samples: np.ndarray) -> None:
        pass

    def step(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def drain(self) -> None:
        pass


_STOP = object()
_REWIND_EPSILON_S = 0.02


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
        ring_preroll_s: float = 0.2,
        use_ring_spans: bool = False,
        finalizing_timeout_s: float = 8.0,
    ):
        self.transcriber = transcriber
        self.bus = bus
        self.sample_rate = sample_rate
        self.max_queue = max_queue
        self.max_draining_sessions = max_draining_sessions
        self.ring_preroll_s = ring_preroll_s
        self.use_ring_spans = use_ring_spans
        self.finalizing_timeout_s = finalizing_timeout_s

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False
        self._ready = threading.Event()
        self._load_error: Exception | None = None
        self._ring_buffer: AudioRingBuffer | None = None
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
        self._enqueue_control(_STOP, kind="stop")
        self._thread.join(timeout=2.0)
        self._thread = None

    def feed(self, frame: AudioFrame | np.ndarray) -> None:
        self._enqueue(frame, kind="audio")

    def set_ring_buffer(self, ring_buffer: AudioRingBuffer) -> None:
        self._ring_buffer = ring_buffer

    def open_session(
        self, t: float, utterance_id: int, speaker: int | None = None
    ) -> None:
        self._enqueue(
            _Open(t=t, utterance_id=utterance_id, speaker=speaker), kind="open"
        )

    def close_session(self, t: float, utterance_id: int) -> None:
        self._enqueue(_Close(t=t, utterance_id=utterance_id), kind="close")

    def bind_speaker(self, utterance_id: int, speaker: int) -> None:
        self._enqueue(
            _SpeakerBind(utterance_id=utterance_id, speaker=speaker), kind="bind"
        )

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
            return
        except queue.Full:
            pass

        if kind == "audio":
            self._enqueue_audio_under_pressure(item)
        else:
            self._enqueue_control(item, kind=kind)

    def _enqueue_audio_under_pressure(self, item) -> None:
        if self._drop_oldest_audio_queued():
            try:
                self._queue.put_nowait(item)
                self.bus.publish(
                    QueueOverflow(component="transcriber", depth=self._queue.qsize())
                )
                return
            except queue.Full:
                pass
        self.bus.publish(
            QueueOverflow(component="transcriber", depth=self._queue.qsize())
        )

    def _enqueue_control(self, item, *, kind: str) -> None:
        if self._drop_oldest_audio_queued():
            try:
                self._queue.put_nowait(item)
                self.bus.publish(
                    QueueOverflow(component="transcriber", depth=self._queue.qsize())
                )
                return
            except queue.Full:
                pass
        try:
            self._queue.put(item, timeout=0.2)
            if kind != "stop":
                self.bus.publish(
                    WorkerBackpressure(
                        component="transcriber",
                        reason=f"{kind}_queue_full",
                        depth=self._queue.qsize(),
                    )
                )
        except queue.Full:
            self.bus.publish(
                WorkerBackpressure(
                    component="transcriber",
                    reason=f"{kind}_queue_full",
                    depth=self._queue.qsize(),
                )
            )

    def _drop_oldest_audio_queued(self) -> bool:
        kept = []
        dropped = False
        while True:
            try:
                queued = self._queue.get_nowait()
            except queue.Empty:
                break
            if not dropped and isinstance(queued, (AudioFrame, np.ndarray)):
                dropped = True
                continue
            kept.append(queued)

        for queued in kept:
            try:
                self._queue.put_nowait(queued)
            except queue.Full:
                if isinstance(queued, (AudioFrame, np.ndarray)):
                    dropped = True
                    continue
                try:
                    self._queue.put(queued, timeout=0.05)
                except queue.Full:
                    self.bus.publish(
                        WorkerBackpressure(
                            component="transcriber",
                            reason="control_requeue_full",
                            depth=self._queue.qsize(),
                        )
                    )
        return dropped

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
        rewind_replay_from_t: float | None = None
        # Sliding buffer of recent audio. Captures chunks that arrive in
        # this worker's queue before the corresponding _Open from the
        # coordinator (the speech-trigger chunk almost always lands here
        # first) and chunks that arrive while we're still draining the
        # previous session.
        pending: deque[AudioFrame | np.ndarray] = deque(maxlen=self.PENDING_MAXLEN)

        def feed_audio(state: _SessionState, item: AudioFrame | np.ndarray) -> int:
            if isinstance(item, AudioFrame):
                if item.t_end <= state.end_t:
                    return 0
                if item.t_start < state.end_t:
                    offset = round((state.end_t - item.t_start) * item.sample_rate)
                    chunk = item.samples[max(0, offset) :]
                else:
                    chunk = item.samples
                if len(chunk) == 0:
                    return 0
                state.session.feed(chunk)
                state.end_t = max(state.end_t, item.t_end)
                return len(chunk)

            state.session.feed(item)
            state.end_t += len(item) / self.sample_rate
            return len(item)

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
            nonlocal recording, rewind_replay_from_t
            recording = _SessionState(
                session=self.transcriber.start_session(),
                start_t=open_evt.t,
                end_t=open_evt.t,
                utterance_id=open_evt.utterance_id,
                phase=_SessionPhase.RECORDING,
                speaker=open_evt.speaker,
            )
            replayed = 0
            replay_source = "pending"
            if rewind_replay_from_t is not None and self._ring_buffer is not None:
                latest_t = self._ring_buffer.latest_t_end()
                if latest_t is not None:
                    t_start = max(0.0, min(open_evt.t, rewind_replay_from_t))
                    audio = self._ring_buffer.span(t_start, latest_t)
                    if len(audio):
                        recording.session.feed(audio)
                        recording.end_t = latest_t
                        replayed = len(audio)
                        replay_source = "rewind"
                rewind_replay_from_t = None
            elif self.use_ring_spans and self._ring_buffer is not None:
                latest_t = self._ring_buffer.latest_t_end()
                if latest_t is not None:
                    t_start = max(0.0, open_evt.t - self.ring_preroll_s)
                    audio = self._ring_buffer.span(t_start, latest_t)
                    if len(audio):
                        recording.session.feed(audio)
                        recording.end_t = latest_t
                        replayed = len(audio)
                        replay_source = "ring"
            if replayed == 0:
                for chunk in pending:
                    replayed += feed_audio(recording, chunk)
            pending.clear()
            debug_log(
                "tw",
                f"_Open t={open_evt.t:.2f}",
                f"utt={open_evt.utterance_id}",
                f"speaker={'S' + str(open_evt.speaker) if open_evt.speaker is not None else '?'}",
                f"replayed={replayed}",
                f"source={replay_source}",
                f"draining={len(draining)}",
            )
            update_state()

        def step_session(state: _SessionState) -> bool:
            return bool(state.session.step())

        def publish_interim(state: _SessionState, *, changed: bool) -> None:
            if changed and state.session.text != state.last_emit_text:
                state.last_emit_text = state.session.text
                self.bus.publish(
                    Interim(
                        text=state.session.text,
                        span=(state.start_t, state.end_t),
                        utterance_id=state.utterance_id,
                        speaker=state.speaker,
                    )
                )

        def finish_drained(state: _SessionState) -> None:
            text = state.session.text.strip()
            discarded = not text or is_hallucination(text)
            debug_log(
                "tw",
                f"drained text={text[:40]!r}",
                f"span=({state.start_t:.2f},{state.end_t:.2f})",
                f"utt={state.utterance_id}",
                f"discarded={discarded}",
                f"buffered={len(pending)}",
                f"remaining_drains={len(draining)}",
            )
            if discarded:
                self.bus.publish(
                    FinalDiscarded(
                        span=(state.start_t, state.end_t),
                        utterance_id=state.utterance_id,
                        speaker=state.speaker,
                        reason="empty" if not text else "hallucination",
                    )
                )
            else:
                self.bus.publish(
                    Final(
                        text=text,
                        span=(state.start_t, state.end_t),
                        utterance_id=state.utterance_id,
                        speaker=state.speaker,
                    )
                )

        def add_draining(state: _SessionState) -> None:
            state.finalizing_started_at = time.monotonic()
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

        def ring_final_state(
            *,
            start_t: float,
            end_t: float,
            utterance_id: int,
            speaker: int | None,
            one_shot: bool,
        ) -> tuple[_SessionState, int, str] | None:
            if self._ring_buffer is None:
                return None
            audio = self._ring_buffer.span(start_t, end_t)
            if len(audio) == 0:
                return None
            if one_shot:
                transcribe_once = getattr(self.transcriber, "transcribe_once", None)
                if callable(transcribe_once):
                    try:
                        text = transcribe_once(audio)
                        return (
                            _SessionState(
                                session=_CompletedTranscription(text),
                                start_t=start_t,
                                end_t=end_t,
                                utterance_id=utterance_id,
                                phase=_SessionPhase.FINALIZING,
                                speaker=speaker,
                            ),
                            len(audio),
                            "one_shot_rewind",
                        )
                    except Exception as exc:
                        debug_log("tw", f"one-shot rewind failed: {exc}")
            state = _SessionState(
                session=self.transcriber.start_session(),
                start_t=start_t,
                end_t=end_t,
                utterance_id=utterance_id,
                phase=_SessionPhase.FINALIZING,
                speaker=speaker,
            )
            state.session.feed(audio)
            state.session.close()
            return state, len(audio), "stream_rewind" if one_shot else "ring"

        def close_recording(close_evt: _Close) -> None:
            nonlocal recording, rewind_replay_from_t
            assert recording is not None
            rewinding = close_evt.t < recording.end_t - _REWIND_EPSILON_S
            if self.use_ring_spans or rewinding:
                built = ring_final_state(
                    start_t=recording.start_t,
                    end_t=close_evt.t,
                    utterance_id=recording.utterance_id,
                    speaker=recording.speaker,
                    one_shot=rewinding,
                )
                if built is not None:
                    final_state, sample_count, source = built
                    add_draining(final_state)
                    if rewinding:
                        rewind_replay_from_t = close_evt.t
                    debug_log(
                        "tw",
                        f"_Close t={close_evt.t:.2f}",
                        f"span=({final_state.start_t:.2f},{final_state.end_t:.2f})",
                        f"utt={final_state.utterance_id}",
                        f"source={source}",
                        f"samples={sample_count}",
                        f"draining={len(draining)}",
                    )
                    recording = None
                    update_state()
                    return
                if rewinding:
                    debug_log(
                        "tw",
                        "rewind unavailable",
                        f"t={close_evt.t:.2f}",
                        f"span=({recording.start_t:.2f},{recording.end_t:.2f})",
                        f"utt={recording.utterance_id}",
                    )

            recording.session.close()
            recording.end_t = max(recording.end_t, close_evt.t)
            recording.phase = _SessionPhase.FINALIZING
            add_draining(recording)
            debug_log(
                "tw",
                f"_Close t={close_evt.t:.2f}",
                f"span=({recording.start_t:.2f},{recording.end_t:.2f})",
                f"utt={recording.utterance_id}",
                "source=session",
                f"draining={len(draining)}",
            )
            recording = None
            update_state()

        def bind_speaker(bind_evt: _SpeakerBind) -> None:
            updated = False
            if (
                recording is not None
                and recording.utterance_id == bind_evt.utterance_id
            ):
                recording.speaker = bind_evt.speaker
                updated = True
            for state in draining:
                if state.utterance_id == bind_evt.utterance_id:
                    state.speaker = bind_evt.speaker
                    updated = True
            if updated:
                debug_log(
                    "tw",
                    f"SpeakerBind utt={bind_evt.utterance_id}",
                    f"S{bind_evt.speaker}",
                )

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
                if (
                    recording is not None
                    and recording.utterance_id == item.utterance_id
                ):
                    close_recording(item)
                elif recording is not None:
                    debug_log(
                        "tw",
                        f"_Close ignored t={item.t:.2f}",
                        f"utt={item.utterance_id}",
                        f"recording_utt={recording.utterance_id}",
                    )
                # if idle: ignore
            elif isinstance(item, _SpeakerBind):
                bind_speaker(item)
            elif isinstance(item, (AudioFrame, np.ndarray)):
                if recording is not None:
                    feed_audio(recording, item)
                else:
                    # No open recording session yet: park for the next session.
                    pending.append(item)

            if recording is not None:
                publish_interim(recording, changed=step_session(recording))

            for state in list(draining):
                changed = step_session(state)
                if state.phase.emits_interim:
                    publish_interim(state, changed=changed)
                timed_out = (
                    state.finalizing_started_at is not None
                    and self.finalizing_timeout_s > 0
                    and time.monotonic() - state.finalizing_started_at
                    >= self.finalizing_timeout_s
                )
                if timed_out:
                    debug_log(
                        "tw",
                        "finalizing timeout",
                        f"utt={state.utterance_id}",
                        f"timeout={self.finalizing_timeout_s:.1f}s",
                    )
                    self.bus.publish(
                        WorkerBackpressure(
                            component="transcriber",
                            reason="finalizing_timeout",
                            depth=len(draining),
                        )
                    )
                if state.session.done or timed_out:
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
