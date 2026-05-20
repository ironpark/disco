"""Qwen3-ASR backend.

Qwen3-ASR is a blob-style transcriber: ``model.generate(audio)`` takes a
complete audio array and returns the transcript. It can't be fed
incrementally, so to give the live-caption experience that Voxtral
provides, we periodically run ``generate()`` on the audio collected so
far during recording. Each partial pass decodes from scratch — there
is no streaming state to carry across passes — so this trades compute
for latency: an N-second utterance with an ``interim_interval_s`` of
2 s decodes roughly N/2 + 1 times. Only one partial runs at a time,
so a slow decode just skips the next interim trigger.

At ``close()`` we run the final pass on the complete buffer. Its
output replaces whatever the last partial showed.
"""

import numpy as np
from mlx_audio.stt import load as load_stt

from disco.asr.transcriber import is_hallucination  # re-used; module-level helper


def _delta_from_result(result) -> tuple[str, bool]:
    """Pull the text delta and is_final flag from a Qwen3 StreamingResult."""
    if isinstance(result, str):
        return result, False
    return getattr(result, "text", "") or "", bool(getattr(result, "is_final", False))


class QwenSession:
    """One Qwen3-ASR transcription session.

    State (single-threaded; lives on the TranscriberWorker thread):
        recording, no gen      idle waiting for the next interim trigger
        recording, gen active  pulling deltas from a partial generator
        closed, no gen         about to start the final generator
        closed, gen active     pulling deltas from the final generator
        done                   terminal
    """

    def __init__(
        self,
        model,
        sample_rate: int,
        language: str | None = None,
        interim_interval_s: float = 2.0,
    ):
        self._model = model
        self._sample_rate = sample_rate
        self._language = language
        self._interim_interval = max(0.5, interim_interval_s)
        self._chunks: list[np.ndarray] = []
        self._cur_samples = 0
        self._next_interim_at = int(self._interim_interval * sample_rate)
        self._text = ""
        self._partial_text = ""
        self._gen = None
        self._gen_is_final = False
        self._closed = False
        self._done = False

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
        self._cur_samples += len(samples)

    def step(self) -> bool:
        """Advance one decoder step. Returns True iff displayed text changed."""
        # 1. Active generator — pull one delta.
        if self._gen is not None:
            return self._pump_gen()

        # 2. No generator running.
        if self._done:
            return False

        if self._closed:
            # Buffer is final; start the final pass.
            self._start_gen(final=True)
            return False

        # 3. Recording: maybe trigger a partial pass.
        if self._cur_samples >= self._next_interim_at and self._chunks:
            self._start_gen(final=False)
            self._next_interim_at = self._cur_samples + int(
                self._interim_interval * self._sample_rate
            )
        return False

    def close(self) -> None:
        """Mark the buffer as final. The next ``step()`` starts the final pass."""
        self._closed = True
        # If a partial gen is in flight, let it finish naturally in step().
        # When it ends, step() sees _closed and kicks off the final pass.

    def drain(self) -> None:
        while not self._done:
            self.step()

    # ---- internals ----

    def _start_gen(self, *, final: bool) -> None:
        if not self._chunks:
            if final:
                self._done = True
            return
        audio = np.concatenate(self._chunks)
        gen_kwargs: dict = {"stream": True}
        if self._language is not None:
            gen_kwargs["language"] = self._language
        try:
            self._gen = self._model.generate(audio, **gen_kwargs)
            self._gen_is_final = final
            self._partial_text = ""
        except Exception as exc:
            print(f"Qwen3-ASR generate() failed: {exc}")
            self._gen = None
            if final:
                self._done = True

    def _pump_gen(self) -> bool:
        try:
            result = next(self._gen)
        except StopIteration:
            # Partial pass finished. Promote its text to the displayed text
            # so we don't briefly fall back to a shorter accumulator on the
            # next partial.
            if self._gen_is_final:
                self._done = True
            self._gen = None
            return False
        except Exception as exc:
            print(f"Qwen3-ASR decode error: {exc}")
            if self._gen_is_final:
                self._done = True
            self._gen = None
            return False

        delta, is_final_token = _delta_from_result(result)
        if not delta:
            return False
        self._partial_text += delta
        self._text = self._partial_text
        if is_final_token and self._gen_is_final:
            self._done = True
        return True


class Qwen3Transcriber:
    """Loads Qwen3-ASR and hands out per-utterance sessions."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-ASR-1.7B-8bit",
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


__all__ = ["Qwen3Transcriber", "QwenSession", "is_hallucination"]
