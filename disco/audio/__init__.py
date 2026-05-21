"""Audio capture and utilities."""

from disco.audio.utils import calculate_rms, save_to_wav

__all__ = ["AudioCapture", "calculate_rms", "save_to_wav"]


def __getattr__(name: str):
    if name == "AudioCapture":
        from disco.audio.capture import AudioCapture

        return AudioCapture
    raise AttributeError(name)
