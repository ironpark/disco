"""Streaming speaker diarization via mlx-audio sortformer."""

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from mlx_audio.vad import load as load_vad

from disco.runtime.debug import enabled as debug_enabled
from disco.runtime.debug import log as debug_log


_RESET = object()  # control sentinel: reset streaming state
_STOP = object()  # control sentinel: stop worker


@dataclass(frozen=True)
class _Segment:
    """A diarization segment in real audio-time seconds."""

    start: float
    end: float
    speaker: int


# Old segments aren't queried — drop them to keep dominant_speaker_in scans cheap.
_SEGMENT_RETENTION_S = 600.0


class Diarizer:
    """Continuous-stream diarizer wrapping sortformer.

    The model is loaded and driven exclusively from a worker thread. MLX
    streams are thread-local, so every MLX op against the model — load,
    ``init_streaming_state``, and each ``feed`` — must happen in that one
    thread. Producers (the audio loop) only enqueue chunks or control
    sentinels and read shared state under a lock.

    Two things to know about the model's output that the wrapper papers over:

    1. ``out.segments`` from ``model.feed()`` contains only segments derived
       from the *current* chunk's predictions, not a cumulative history. We
       accumulate them ourselves under ``_segments``.

    2. The model's segment timestamps live in its internal frame-coordinate
       time (``state.frames_processed * frame_duration``), which advances
       *faster* than real audio time — for 100 ms input chunks the model's
       clock ticks ~0.16 s per chunk on the v2.1 4-speaker model. We convert
       each chunk's segments into real-audio-time before storing, using the
       running ratio between processed real audio and the model's clock.

    External time (``elapsed_seconds`` and the times accepted by
    ``dominant_speaker_in`` / ``speaker_at``) is **real audio time** counted
    from the most recent ``start()``.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32",
        sample_rate: int = 16000,
        max_queue: int = 200,
        on_overflow: Callable[[int], None] | None = None,
        on_activity: Callable[[float, float, int | None, tuple[int, ...]], None] | None = None,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.max_queue = max_queue
        self.on_overflow = on_overflow
        # Called once per processed chunk with
        # (chunk_t_start, chunk_t_end, primary_speaker, all_speakers).
        # primary_speaker is None when no segments were emitted (silence).
        self.on_activity = on_activity

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False
        self._ready_event = threading.Event()
        self._reset_event = threading.Event()

        self._lock = threading.Lock()
        self._model = None
        self._frame_duration: float = 0.0  # set in worker after model load
        self._segments: list[_Segment] = []  # cumulative, in real audio-time
        self._samples_fed: int = 0  # producer-side count, only mutated by feed()

    def load(self) -> None:
        """Start the worker and block until the model is loaded."""
        if self._thread is not None:
            return
        print(f"Loading diarization model: {self.model_name}")
        self._ready_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready_event.wait()
        print("Diarization model loaded!")

    def start(self) -> None:
        """Reset streaming state for a fresh diarization stream."""
        if self._thread is None:
            self.load()
        self._reset_event.clear()
        with self._lock:
            self._segments = []
            self._samples_fed = 0
        self._queue.put(_RESET)
        self._reset_event.wait()

    def stop(self) -> None:
        """Stop the worker thread. Call on shutdown."""
        if self._thread is None:
            return
        self._running = False
        self._queue.put(_STOP)
        self._thread.join(timeout=2.0)
        self._thread = None

    def feed(self, chunk: np.ndarray) -> None:
        """Queue a chunk for diarization. Cheap and thread-safe."""
        if not self._running or self._model is None:
            return
        if chunk.ndim > 1:
            chunk = chunk.reshape(-1)
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        self._samples_fed += len(chunk)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass
            if self.on_overflow is not None:
                self.on_overflow(self._queue.qsize())

    def elapsed_seconds(self) -> float:
        """Real audio seconds fed since the last ``start()``."""
        return self._samples_fed / self.sample_rate

    def dominant_speaker_in(self, t_start: float, t_end: float) -> int | None:
        """Return the speaker with the most overlap in [t_start, t_end], or None.

        Times are real audio seconds since the most recent ``start()``.
        """
        if t_end <= t_start:
            return None
        with self._lock:
            durations: dict[int, float] = {}
            for seg in self._segments:
                overlap = min(seg.end, t_end) - max(seg.start, t_start)
                if overlap > 0:
                    durations[seg.speaker] = durations.get(seg.speaker, 0.0) + overlap
        if not durations:
            return None
        return max(durations.items(), key=lambda kv: kv[1])[0]

    def speaker_at(self, t: float, tolerance: float = 0.3) -> int | None:
        """Speaker active at real-audio time ``t``, or None if uncertain.

        ``tolerance`` lets the lookup match segments that ended slightly
        before ``t`` — the worker is always somewhat behind real time
        because of queueing.
        """
        with self._lock:
            for seg in reversed(self._segments):
                if seg.start <= t <= seg.end + tolerance:
                    return seg.speaker
        return None

    # ---- worker thread ----

    def _worker(self) -> None:
        try:
            self._model = load_vad(self.model_name)
            state = self._model.init_streaming_state()
            proc = self._model._processor_config
            subsampling_factor = (
                self._model.config.fc_encoder_config.subsampling_factor
            )
            self._frame_duration = (proc.hop_length * subsampling_factor) / proc.sampling_rate
        except Exception as exc:
            print(f"Diarizer load failed: {exc}")
            self._ready_event.set()
            return
        self._ready_event.set()

        real_processed_s = 0.0  # cumulative real audio time processed by the model
        prev_sf_s = 0.0  # cumulative sortformer-time before this feed

        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _STOP:
                break
            if item is _RESET:
                state = self._model.init_streaming_state()
                real_processed_s = 0.0
                prev_sf_s = 0.0
                with self._lock:
                    self._segments = []
                self._reset_event.set()
                continue

            chunk = item
            chunk_real_dur = len(chunk) / self.sample_rate
            chunk_real_start = real_processed_s
            real_processed_s += chunk_real_dur

            try:
                out, state = self._model.feed(
                    chunk, state, sample_rate=self.sample_rate
                )
            except Exception as exc:
                print(f"Diarizer feed error: {exc}")
                continue

            sf_processed_s = state.frames_processed * self._frame_duration
            delta_sf = sf_processed_s - prev_sf_s

            # Map each segment from sortformer-time within this chunk's
            # sf-window onto the chunk's real-time window. If a segment's
            # sf range extends outside [prev_sf_s, sf_processed_s] it just
            # gets clipped to the chunk's real-time window.
            new_segs: list[_Segment] = []
            for seg in out.segments:
                if delta_sf > 0:
                    rel_start = (seg.start - prev_sf_s) / delta_sf
                    rel_end = (seg.end - prev_sf_s) / delta_sf
                    rel_start = max(0.0, min(1.0, rel_start))
                    rel_end = max(0.0, min(1.0, rel_end))
                else:
                    rel_start, rel_end = 0.0, 1.0
                start = chunk_real_start + rel_start * chunk_real_dur
                end = chunk_real_start + rel_end * chunk_real_dur
                if end > start:
                    new_segs.append(_Segment(start=start, end=end, speaker=int(seg.speaker)))

            prev_sf_s = sf_processed_s

            if debug_enabled("diar") and (new_segs or out.segments):
                tail_real = ", ".join(
                    f"S{s.speaker}:{s.start:.2f}-{s.end:.2f}" for s in new_segs[-3:]
                )
                debug_log(
                    "diar",
                    f"real={real_processed_s:.2f}s",
                    f"sf={sf_processed_s:.2f}s",
                    f"+sf={delta_sf:.2f}",
                    f"new=[{tail_real}]",
                    f"qsz={self._queue.qsize()}",
                )

            with self._lock:
                if new_segs:
                    self._segments.extend(new_segs)
                # Drop very old segments to keep scans bounded.
                cutoff = real_processed_s - _SEGMENT_RETENTION_S
                if cutoff > 0 and self._segments and self._segments[0].end < cutoff:
                    self._segments = [s for s in self._segments if s.end >= cutoff]

            if self.on_activity is not None:
                # Sum each speaker's duration within this chunk to pick a primary.
                durations: dict[int, float] = {}
                for seg in new_segs:
                    durations[seg.speaker] = (
                        durations.get(seg.speaker, 0.0) + seg.end - seg.start
                    )
                if durations:
                    primary = max(durations.items(), key=lambda kv: kv[1])[0]
                    all_speakers = tuple(sorted(durations.keys()))
                else:
                    primary = None
                    all_speakers = ()
                try:
                    self.on_activity(
                        chunk_real_start, real_processed_s, primary, all_speakers
                    )
                except Exception as exc:
                    print(f"Diarizer on_activity error: {exc}")
