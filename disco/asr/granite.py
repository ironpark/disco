"""IBM Granite Speech backend via mlx-audio."""

from disco.asr.hallucination import is_hallucination
from disco.asr.qwen import QwenSession


class GraniteSpeechTranscriber:
    """Loads Granite Speech and hands out blob-style ASR sessions."""

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-speech-4.1-2b",
        sample_rate: int = 16000,
        language: str | None = None,
        interim_interval_s: float = 2.0,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.language = language
        self.interim_interval_s = interim_interval_s
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from mlx_audio.stt import load as load_stt

            print(f"Loading ASR model: {self.model_name}")
            self._model = load_stt(self.model_name)
            print("ASR model loaded!")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def start_session(self) -> QwenSession:
        return QwenSession(
            self.model,
            self.sample_rate,
            language=self.language,
            interim_interval_s=self.interim_interval_s,
        )


__all__ = ["GraniteSpeechTranscriber", "is_hallucination"]
