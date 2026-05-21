"""Timestamped audio frames shared across the realtime pipeline."""

from collections import deque
from dataclasses import dataclass
import threading

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    """One chunk of audio on the pipeline's shared audio clock."""

    seq: int
    t_start: float
    t_end: float
    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class AudioRingBuffer:
    """Bounded frame buffer for consumers that need to recover audio spans."""

    def __init__(self, retention_s: float = 30.0):
        self.retention_s = retention_s
        self._frames: deque[AudioFrame] = deque()
        self._lock = threading.Lock()

    def append(self, frame: AudioFrame) -> None:
        with self._lock:
            self._frames.append(frame)
            cutoff = frame.t_end - self.retention_s
            while self._frames and self._frames[0].t_end < cutoff:
                self._frames.popleft()

    def span(self, t_start: float, t_end: float) -> np.ndarray:
        """Return concatenated samples overlapping ``[t_start, t_end]``."""
        if t_end <= t_start:
            return np.array([], dtype=np.float32)
        chunks: list[np.ndarray] = []
        with self._lock:
            frames = list(self._frames)
        for frame in frames:
            if frame.t_end <= t_start or frame.t_start >= t_end:
                continue
            start_offset = max(0, int((t_start - frame.t_start) * frame.sample_rate))
            end_offset = min(
                len(frame.samples),
                int((t_end - frame.t_start) * frame.sample_rate),
            )
            if end_offset > start_offset:
                chunks.append(frame.samples[start_offset:end_offset])
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)
