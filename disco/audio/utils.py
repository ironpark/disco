"""Audio utility functions."""

import tempfile
import wave
from pathlib import Path

import numpy as np


def calculate_rms(audio_data: np.ndarray) -> float:
    """Calculate RMS of audio data."""
    return float(np.sqrt(np.mean(audio_data**2)))


def save_to_wav(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """Save audio data to a temporary WAV file.

    Args:
        audio_data: Audio samples as float32 array
        sample_rate: Sample rate in Hz
        channels: Number of audio channels

    Returns:
        Path to the temporary WAV file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_file.name, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())
    return temp_file.name


def cleanup_temp_file(path: str) -> None:
    """Remove a temporary file if it exists."""
    Path(path).unlink(missing_ok=True)
