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

    asr_backend: str = "voxtral"
    # When None, the factory picks the backend's default checkpoint.
    model_name: str | None = None
    translation_model: str | None = None
    smart_turn: bool = False
    smart_turn_model: str = "mlx-community/smart-turn-v3"
    smart_turn_threshold: float = 0.5
    sample_rate: int = 16000
    channels: int = 1
    # Minimum audio fed into a session before a VAD-triggered finalize.
    chunk_duration: float = 0.5
    silence_threshold: float = 0.01
    silence_duration: float = 0.5
    max_utterance_duration: float = 10.0
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
