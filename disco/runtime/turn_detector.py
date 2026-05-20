"""VAD + diarizer-driven turn boundary detection."""

import queue
import threading

import numpy as np

from disco.diar.sortformer import Diarizer
from disco.runtime.events import (
    EventBus,
    QueueOverflow,
    SpeakerChange,
    SpeechEnd,
    SpeechStart,
)
from disco.vad.silero import SileroVAD


_STOP = object()


class TurnDetector:
    """AudioConsumer that emits utterance lifecycle events.

    Combines silero VAD for silence-based turn boundaries with sortformer
    queries for sustained speaker-change boundaries. Owns its own thread
    so VAD inference doesn't ride the sounddevice callback.
    """

    def __init__(
        self,
        vad: SileroVAD,
        diarizer: Diarizer,
        bus: EventBus,
        *,
        sample_rate: int = 16000,
        silence_duration: float = 0.5,
        min_utterance_duration: float = 0.5,
        speaker_change_hold: float = 0.4,
        speaker_lag: float = 0.2,
        block_duration: float = 0.1,
        max_queue: int = 200,
    ):
        self.vad = vad
        self.diarizer = diarizer
        self.bus = bus
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.min_utterance_duration = min_utterance_duration
        self.speaker_change_hold = speaker_change_hold
        self.speaker_lag = speaker_lag
        self.block_duration = block_duration
        self.max_queue = max_queue

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._running = False
        self._queue.put(_STOP)
        self._thread.join(timeout=2.0)
        self._thread = None

    def feed(self, chunk: np.ndarray) -> None:
        if not self._running:
            return
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
            self.bus.publish(
                QueueOverflow(component="turn_detector", depth=self._queue.qsize())
            )

    def _worker(self) -> None:
        required_silence_chunks = max(
            1, int(self.silence_duration / self.block_duration)
        )
        min_samples = int(self.min_utterance_duration * self.sample_rate)

        state: str = "quiet"
        silence_chunks = 0
        samples_in_utterance = 0
        bound_speaker: int | None = None
        speaker_change_start: float | None = None

        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _STOP:
                break

            chunk = item
            has_speech = self.vad.is_speech_chunk(chunk)
            t_now = self.diarizer.elapsed_seconds()

            if state == "quiet":
                if has_speech:
                    state = "speaking"
                    silence_chunks = 0
                    samples_in_utterance = chunk.size
                    bound_speaker = None
                    speaker_change_start = None
                    self.bus.publish(SpeechStart(t=t_now))
                continue

            # state == "speaking"
            samples_in_utterance += chunk.size
            if has_speech:
                silence_chunks = 0
            else:
                silence_chunks += 1

            # Lazily bind primary speaker once diarizer has output.
            if bound_speaker is None:
                bound_speaker = self.diarizer.dominant_speaker_in(
                    t_now - samples_in_utterance / self.sample_rate, t_now
                )
            else:
                latest = self.diarizer.speaker_at(t_now - self.speaker_lag)
                if latest is not None and latest != bound_speaker:
                    if speaker_change_start is None:
                        speaker_change_start = t_now
                    elif t_now - speaker_change_start >= self.speaker_change_hold:
                        self.bus.publish(
                            SpeakerChange(
                                from_speaker=bound_speaker,
                                to_speaker=latest,
                                t=t_now,
                            )
                        )
                        state = "quiet"
                        silence_chunks = 0
                        samples_in_utterance = 0
                        bound_speaker = None
                        speaker_change_start = None
                        continue
                else:
                    speaker_change_start = None

            if (
                silence_chunks >= required_silence_chunks
                and samples_in_utterance >= min_samples
            ):
                self.bus.publish(SpeechEnd(t=t_now))
                state = "quiet"
                silence_chunks = 0
                samples_in_utterance = 0
                bound_speaker = None
                speaker_change_start = None
