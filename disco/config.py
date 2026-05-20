"""Configuration and data classes for disco."""

from dataclasses import dataclass, field


@dataclass
class TranscriptEntry:
    """A timestamped transcription entry."""

    timestamp: float
    text: str


@dataclass
class ASRConfig:
    """Configuration for real-time ASR."""

    model_name: str = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    sample_rate: int = 16000
    channels: int = 1
    # Minimum audio fed into a session before a VAD-triggered finalize.
    chunk_duration: float = 0.5
    silence_threshold: float = 0.01
    silence_duration: float = 0.5
    device: int | str | None = None
    language: str = "English"
    translate_to_korean: bool = False


# Language name to ISO 639-1 code mapping
LANG_CODE_MAP: dict[str, str] = {
    "english": "en",
    "korean": "ko",
    "japanese": "ja",
    "chinese": "zh",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
}
