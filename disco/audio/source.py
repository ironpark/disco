"""Audio source that fans incoming chunks out to multiple consumers."""

from typing import Protocol

import numpy as np
import sounddevice as sd


class AudioConsumer(Protocol):
    """Anything that accepts raw float32 audio chunks from a callback thread.

    Implementations must return quickly — sounddevice's callback thread is
    drumming at ~100 ms and any work belongs on a consumer-owned worker.
    """

    def feed(self, chunk: np.ndarray) -> None: ...


class AudioSource:
    """sounddevice ``InputStream`` with subscriber fan-out.

    The stream callback runs on a sounddevice-owned thread. We simply
    iterate over the registered consumers and call ``feed`` on each.
    Each consumer is expected to enqueue and return immediately so the
    callback never blocks the audio device.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        block_duration: float = 0.1,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.block_duration = block_duration
        self.blocksize = int(sample_rate * block_duration)

        self._consumers: list[AudioConsumer] = []
        self._stream: sd.InputStream | None = None

    def subscribe(self, consumer: AudioConsumer) -> None:
        self._consumers.append(consumer)

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"Audio status: {status}")
        # Copy once; consumers may queue references for later processing.
        chunk = indata.copy()
        for consumer in self._consumers:
            try:
                consumer.feed(chunk)
            except Exception as exc:
                print(f"AudioSource consumer error: {exc}")

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._callback,
            blocksize=self.blocksize,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "AudioSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
