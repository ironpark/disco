"""Silero VAD wrapper."""

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad


class SileroVAD:
    """Voice Activity Detection using Silero VAD."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
    ):
        """Initialize Silero VAD.

        Args:
            sample_rate: Audio sample rate in Hz
            threshold: Speech probability threshold
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._model = None

    def load(self) -> None:
        """Load the VAD model."""
        if self._model is None:
            print("Loading Silero VAD model...")
            self._model = load_silero_vad()
            print("VAD model loaded!")

    @property
    def model(self):
        """Get the VAD model, loading if necessary."""
        if self._model is None:
            self.load()
        return self._model

    def has_speech(self, audio_data: np.ndarray) -> bool:
        """Check if audio contains speech.

        Args:
            audio_data: Audio samples as numpy array

        Returns:
            True if speech is detected
        """
        audio_flat = audio_data.flatten()
        audio_tensor = torch.from_numpy(audio_flat).float()
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
        )
        return len(speech_timestamps) > 0

    def is_speech_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Check if a small audio chunk contains speech.

        Uses window-based analysis for short chunks.

        Args:
            audio_chunk: Audio samples as numpy array

        Returns:
            True if majority of windows contain speech
        """
        audio_flat = audio_chunk.flatten()
        # Silero VAD requires exactly 512 samples for 16kHz
        window_size = 512

        if len(audio_flat) < window_size:
            return False

        num_windows = len(audio_flat) // window_size
        speech_count = 0

        for i in range(num_windows):
            window = audio_flat[i * window_size : (i + 1) * window_size]
            audio_tensor = torch.from_numpy(window).float()
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            if speech_prob > self.threshold:
                speech_count += 1

        return speech_count > num_windows / 2
