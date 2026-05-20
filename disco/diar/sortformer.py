"""Streaming speaker diarization via mlx-audio sortformer."""

import queue
import threading
from collections.abc import Callable

import numpy as np
from mlx_audio.vad import load as load_vad


_RESET = object()  # control sentinel: reset streaming state
_STOP = object()  # control sentinel: stop worker


class Diarizer:
    """Continuous-stream diarizer wrapping sortformer.

    The model is loaded and driven exclusively from a worker thread. MLX
    streams are thread-local, so every MLX op against the model — load,
    ``init_streaming_state``, and each ``feed`` — must happen in that one
    thread. Producers (the audio loop) only enqueue chunks or control
    sentinels and read shared state under a lock.

    Segment timestamps are in seconds since the most recent ``start()``.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32",
        sample_rate: int = 16000,
        max_queue: int = 200,
        on_overflow: Callable[[int], None] | None = None,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.max_queue = max_queue
        self.on_overflow = on_overflow

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False
        self._ready_event = threading.Event()
        self._reset_event = threading.Event()

        self._lock = threading.Lock()
        self._model = None
        self._segments: list = []
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
        """Reset streaming state for a fresh diarization stream.

        Blocks until the worker has cleared the queue, reset its state,
        and is ready to accept new audio.
        """
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
            # Drop oldest non-control item to make room.
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
        """Seconds of audio fed since the last ``start()``."""
        return self._samples_fed / self.sample_rate

    def dominant_speaker_in(self, t_start: float, t_end: float) -> int | None:
        """Return the speaker with the most overlap in [t_start, t_end], or None."""
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
        """Speaker active at time ``t``, or None if uncertain.

        ``tolerance`` lets the lookup match segments that ended slightly before
        ``t`` — sortformer reports segments with some lag.
        """
        with self._lock:
            for seg in reversed(self._segments):
                if seg.start <= t <= seg.end + tolerance:
                    return seg.speaker
        return None

    def _worker(self) -> None:
        try:
            self._model = load_vad(self.model_name)
            state = self._model.init_streaming_state()
        except Exception as exc:
            print(f"Diarizer load failed: {exc}")
            self._ready_event.set()
            return
        self._ready_event.set()

        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _STOP:
                break
            if item is _RESET:
                state = self._model.init_streaming_state()
                self._reset_event.set()
                continue

            chunk = item
            try:
                out, state = self._model.feed(
                    chunk, state, sample_rate=self.sample_rate
                )
            except Exception as exc:
                print(f"Diarizer feed error: {exc}")
                continue
            with self._lock:
                self._segments = list(out.segments)
