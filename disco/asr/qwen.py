"""Qwen3-ASR backend.

Qwen3-ASR is a blob-style transcriber: ``model.generate(audio)`` takes a
complete audio array (or path) and returns the transcript. With
``stream=True`` it returns a generator that yields text deltas during
decoding.

We adapt that to TranscriberWorker's per-session interface by:

- buffering raw chunks as they arrive (``feed``),
- kicking off the generator at ``close()``,
- pulling one delta per ``step()`` from the generator (so decoding can
  interleave with the worker's other duties).

Interim text from this backend therefore appears *after* speech ends —
the model can't start decoding until VAD/Coordinator closes the
session. Voxtral remains the better choice for live captions; Qwen3-ASR
is here for comparison and for languages/accents where its accuracy is
preferable.
"""

import numpy as np
from mlx_audio.stt import load as load_stt

from disco.asr.transcriber import is_hallucination  # re-used; module-level helper


class QwenSession:
    """One Qwen3-ASR transcription session driven from the TW worker thread."""

    def __init__(self, model, sample_rate: int, language: str | None = None):
        self._model = model
        self._sample_rate = sample_rate
        self._language = language
        self._chunks: list[np.ndarray] = []
        self._text = ""
        self._done = False
        self._gen = None
        self._closed = False

    @property
    def text(self) -> str:
        return self._text

    @property
    def done(self) -> bool:
        return self._done

    def feed(self, samples: np.ndarray) -> None:
        if samples.ndim > 1:
            samples = samples.reshape(-1)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._chunks.append(samples)

    def step(self) -> bool:
        """Pull one delta from the active generator. Returns True if text changed."""
        if self._done or self._gen is None:
            return False
        try:
            result = next(self._gen)
        except StopIteration:
            self._done = True
            return False
        except Exception as exc:
            print(f"Qwen3-ASR decode error: {exc}")
            self._done = True
            return False
        # mlx-audio's Qwen3-ASR streamer yields a StreamingResult per token.
        delta = getattr(result, "text", "") if not isinstance(result, str) else result
        if not delta:
            return False
        self._text += delta
        if getattr(result, "is_final", False):
            self._done = True
        return True

    def close(self) -> None:
        """Kick off the generator on the buffered audio."""
        if self._closed:
            return
        self._closed = True
        if not self._chunks:
            self._done = True
            return
        audio = np.concatenate(self._chunks)
        # Drop the buffer reference; the model has the data now.
        self._chunks = []
        gen_kwargs: dict = {"stream": True}
        if self._language is not None:
            gen_kwargs["language"] = self._language
        try:
            self._gen = self._model.generate(audio, **gen_kwargs)
        except Exception as exc:
            print(f"Qwen3-ASR generate() failed: {exc}")
            self._done = True

    def drain(self) -> None:
        while not self._done:
            self.step()


class Qwen3Transcriber:
    """Loads ``Qwen3-ASR`` and hands out per-utterance sessions.

    Matches the duck-typed interface used by TranscriberWorker:
    ``load()`` + ``start_session() -> session``.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-ASR-1.7B-8bit",
        sample_rate: int = 16000,
        language: str | None = None,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.language = language
        self._model = None

    def load(self) -> None:
        if self._model is None:
            print(f"Loading ASR model: {self.model_name}")
            self._model = load_stt(self.model_name)
            print("ASR model loaded!")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def start_session(self) -> QwenSession:
        return QwenSession(self.model, self.sample_rate, self.language)


__all__ = ["Qwen3Transcriber", "QwenSession", "is_hallucination"]
