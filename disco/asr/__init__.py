"""Automatic Speech Recognition module."""

from typing import Any

from disco.asr.hallucination import is_hallucination


BACKEND_DEFAULT_MODELS = {
    "voxtral": "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
    "qwen3-asr": "mlx-community/Qwen3-ASR-1.7B-8bit",
}


def make_transcriber(
    backend: str,
    *,
    sample_rate: int = 16000,
    model_name: str | None = None,
    **kwargs: Any,
):
    """Construct a transcriber for the named backend.

    ``backend`` selects the model family; ``model_name`` overrides the
    default checkpoint for that backend. Extra kwargs flow through to
    the concrete class (e.g. ``transcription_delay_ms`` for Voxtral,
    ``language`` for Qwen3-ASR).
    """
    backend = backend.lower()
    name = model_name or BACKEND_DEFAULT_MODELS.get(backend)
    if name is None:
        raise ValueError(
            f"Unknown ASR backend: {backend!r}. "
            f"Choose one of {sorted(BACKEND_DEFAULT_MODELS)}."
        )
    if backend == "voxtral":
        from disco.asr.transcriber import Transcriber

        kwargs.pop("language", None)
        return Transcriber(model_name=name, sample_rate=sample_rate, **kwargs)
    if backend == "qwen3-asr":
        from disco.asr.qwen import Qwen3Transcriber

        return Qwen3Transcriber(model_name=name, sample_rate=sample_rate, **kwargs)
    raise ValueError(f"Unsupported ASR backend: {backend!r}")


__all__ = [
    "BACKEND_DEFAULT_MODELS",
    "is_hallucination",
    "make_transcriber",
]
