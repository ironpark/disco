"""Streaming Silero VAD worker backed by mlx-audio."""

import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from disco.audio.frame import AudioFrame
from disco.runtime.debug import log as debug_log


_STOP = object()


class SileroVad:
    """Run ``mlx-community/silero-vad`` off the audio callback thread."""

    def __init__(
        self,
        *,
        model_name: str = "mlx-community/silero-vad",
        sample_rate: int = 16000,
        threshold: float | None = None,
        start_chunks: int = 1,
        end_chunks: int = 3,
        max_queue: int = 200,
        on_activity: Callable[[float, float, bool, float | None], None] | None = None,
        on_overflow: Callable[[int], None] | None = None,
        model: Any | None = None,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.start_chunks = start_chunks
        self.end_chunks = end_chunks
        self.max_queue = max_queue
        self.on_activity = on_activity
        self.on_overflow = on_overflow

        self._model = model
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False
        self._ready = threading.Event()
        self._load_error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=120.0):
            self._running = False
            raise TimeoutError("Timed out loading VAD model")
        if self._load_error is not None:
            raise RuntimeError("VAD model failed to load") from self._load_error

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

    def feed(self, frame: AudioFrame) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass
            if self.on_overflow is not None:
                self.on_overflow(self._queue.qsize())

    def _worker(self) -> None:
        try:
            if self._model is None:
                from mlx_audio.vad import load as load_vad

                self._model = load_vad(self.model_name)
            state = self._model.initial_state(sample_rate=self.sample_rate)
            model_threshold = getattr(getattr(self._model, "config", None), "threshold", 0.5)
            threshold = float(self.threshold if self.threshold is not None else model_threshold)
            chunk_size = self._chunk_size()
        except Exception as exc:
            self._load_error = exc
            self._ready.set()
            return
        self._ready.set()

        speech = False
        above_run = 0
        below_run = 0
        buffered = np.array([], dtype=np.float32)
        buffered_t_start: float | None = None

        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _STOP:
                break

            frame = item
            samples = frame.samples
            if samples.ndim > 1:
                samples = samples.reshape(-1)
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)

            if buffered_t_start is None:
                buffered_t_start = frame.t_start
            buffered = np.concatenate([buffered, samples])

            while len(buffered) >= chunk_size:
                chunk = buffered[:chunk_size]
                buffered = buffered[chunk_size:]
                assert buffered_t_start is not None
                chunk_t_start = buffered_t_start
                chunk_t_end = chunk_t_start + chunk_size / self.sample_rate
                buffered_t_start = chunk_t_end

                try:
                    probability, state = self._model.feed(
                        chunk, state, sample_rate=self.sample_rate
                    )
                    prob = float(probability.item())
                except Exception as exc:
                    print(f"VAD feed error: {exc}")
                    continue

                active = prob >= threshold
                if active:
                    above_run += 1
                    below_run = 0
                else:
                    below_run += 1
                    above_run = 0

                previous = speech
                if not speech and above_run >= self.start_chunks:
                    speech = True
                elif speech and below_run >= self.end_chunks:
                    speech = False

                if previous != speech:
                    debug_log(
                        "vad",
                        f"speech={speech}",
                        f"prob={prob:.3f}",
                        f"span=({chunk_t_start:.2f},{chunk_t_end:.2f})",
                    )

                if self.on_activity is not None:
                    self.on_activity(chunk_t_start, chunk_t_end, speech, prob)

            if len(buffered) == 0:
                buffered_t_start = None

    def _chunk_size(self) -> int:
        config = getattr(self._model, "config", None)
        branch = getattr(config, "branch_16k" if self.sample_rate == 16000 else "branch_8k", None)
        if branch is not None:
            return int(branch.chunk_size)
        return 512 if self.sample_rate == 16000 else 256
