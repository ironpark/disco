"""Audio capture using sounddevice."""

import queue
from typing import Callable

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Handles audio capture from microphone."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        block_duration: float = 0.1,
    ):
        """Initialize audio capture.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            device: Input device ID (int), name (str), or None for default
            block_duration: Duration of each audio block in seconds
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.block_duration = block_duration
        self.blocksize = int(sample_rate * block_duration)

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def start(self) -> None:
        """Start the audio stream."""
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=self.blocksize,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        """Get an audio chunk from the queue.

        Args:
            timeout: Timeout in seconds

        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def list_devices() -> list[dict]:
    """List available audio input devices.

    Returns:
        List of device info dictionaries
    """
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append({
                "id": i,
                "name": dev["name"],
                "is_default": i == sd.default.device[0],
            })
    return input_devices


def get_device_info(device_id: int) -> dict:
    """Get info for a specific device."""
    return sd.query_devices(device_id)
