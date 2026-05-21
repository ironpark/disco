"""IBM Granite Speech backend via mlx-audio."""

import time
from collections.abc import Iterator

import numpy as np

from disco.asr.hallucination import is_hallucination


class GraniteSpeechSession:
    """Blob-style Granite Speech session with streaming generator output."""

    def __init__(
        self,
        model,
        sample_rate: int,
        interim_interval_s: float = 2.0,
        language: str | None = None,
        emit_interval_s: float = 0.5,
    ):
        self._model = model
        self._sample_rate = sample_rate
        self._interim_interval = max(0.5, interim_interval_s)
        self._emit_interval_s = max(0.1, emit_interval_s)
        self._language = language
        self._chunks: list[np.ndarray] = []
        self._cur_samples = 0
        self._next_interim_at = int(self._interim_interval * sample_rate)
        self._text = ""
        self._tokens: list[int] = []
        self._last_emit_at = 0.0
        self._gen: Iterator[int] | None = None
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
        if self._gen is not None:
            return self._pump_gen()
        if self._done:
            return False
        if self._closed:
            self._start_gen(final=True)
            return False
        if self._cur_samples >= self._next_interim_at and self._chunks:
            self._start_gen(final=False)
            self._next_interim_at = self._cur_samples + int(
                self._interim_interval * self._sample_rate
            )
        return False

    def close(self) -> None:
        self._closed = True

    def drain(self) -> None:
        while not self._done:
            self.step()

    def _start_gen(self, *, final: bool) -> None:
        if not self._chunks:
            if final:
                self._done = True
            return
        audio = np.concatenate(self._chunks)
        try:
            self._gen = self._token_stream(audio)
            self._gen_is_final = final
            self._tokens = []
            self._last_emit_at = 0.0
        except Exception as exc:
            print(f"Granite Speech generate() failed: {exc}")
            self._gen = None
            if final:
                self._done = True

    def _pump_gen(self) -> bool:
        try:
            result = next(self._gen)
        except StopIteration:
            if self._gen_is_final:
                self._done = True
            self._gen = None
            return self._publish_tokens(force=True)
        except Exception as exc:
            print(f"Granite Speech decode error: {exc}")
            if self._gen_is_final:
                self._done = True
            self._gen = None
            return False

        self._tokens.append(int(result))
        return self._publish_tokens(force=False)

    def _token_stream(self, audio: np.ndarray) -> Iterator[int]:
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        prompt = None
        if self._language is not None:
            prompt = f"Translate the speech to {self._language}."

        audio_data = self._model._load_audio(audio)
        input_features, num_audio_tokens = self._model._extract_features(audio_data)
        audio_features = self._model.get_audio_features(input_features)
        mx.eval(audio_features)

        prompt_ids = self._model._build_prompt(num_audio_tokens, prompt)
        inputs_embeds = self._model._build_inputs_embeds(prompt_ids, audio_features)
        mx.eval(inputs_embeds)

        sampler = make_sampler(0.0, top_p=1.0, min_p=0.0, top_k=0)
        logits_processors = make_logits_processors(
            repetition_penalty=None,
            repetition_context_size=100,
        )
        eos_token_id = self._model._tokenizer.eos_token_id

        for token, _ in generate_step(
            prompt=prompt_ids,
            input_embeddings=inputs_embeds.squeeze(0),
            model=self._model,
            max_tokens=4096,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=2048,
        ):
            if int(token) == eos_token_id:
                break
            yield int(token)

    def _publish_tokens(self, *, force: bool) -> bool:
        text = self._decode_tokens().strip()
        if not text or text == self._text:
            return False
        now = time.monotonic()
        throttled = (
            self._last_emit_at and now - self._last_emit_at < self._emit_interval_s
        )
        if not force and throttled:
            return False
        self._text = text
        self._last_emit_at = now
        return True

    def _decode_tokens(self) -> str:
        return self._model._tokenizer.decode(
            self._tokens,
            skip_special_tokens=True,
        )


class GraniteSpeechTranscriber:
    """Loads Granite Speech and hands out blob-style ASR sessions."""

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-speech-4.1-2b",
        sample_rate: int = 16000,
        language: str | None = None,
        interim_interval_s: float = 2.0,
        emit_interval_s: float = 0.5,
        translate_speech: bool = False,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.language = language if translate_speech else None
        self.interim_interval_s = interim_interval_s
        self.emit_interval_s = emit_interval_s
        self.translate_speech = translate_speech
        self._model = None

    def load(self) -> None:
        if self._model is None:
            import mlx.core as mx
            from mlx_audio.stt import load as load_stt

            print(f"Loading ASR model: {self.model_name}")
            self._model = load_stt(self.model_name)
            fixed = self._fix_conv1d_weight_layout()
            if fixed:
                mx.eval(self._model.parameters())
                print(f"Fixed Granite Conv1d weight layout ({fixed} tensors)")
            print("ASR model loaded!")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def start_session(self) -> GraniteSpeechSession:
        return GraniteSpeechSession(
            self.model,
            self.sample_rate,
            interim_interval_s=self.interim_interval_s,
            language=self.language,
            emit_interval_s=self.emit_interval_s,
        )

    def _fix_conv1d_weight_layout(self) -> int:
        """Patch mlx-audio Granite 1x1 conv weights saved in PyTorch layout.

        Some Granite checkpoints/load paths leave Conformer ``up_conv`` and
        ``down_conv`` weights as ``(out, in, 1)``. MLX ``Conv1d`` expects
        ``(out, kernel, in)``, so the first decode fails with a channel
        mismatch like ``input: (..., 1024)`` vs ``weight: (4096, 1024, 1)``.
        """
        fixed = 0
        for name, module in self._model.named_modules():
            if not (name.endswith("up_conv") or name.endswith("down_conv")):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or len(weight.shape) != 3:
                continue
            if weight.shape[2] == 1 and weight.shape[1] != 1:
                module.weight = weight.transpose(0, 2, 1)
                fixed += 1
        return fixed


__all__ = ["GraniteSpeechSession", "GraniteSpeechTranscriber", "is_hallucination"]
