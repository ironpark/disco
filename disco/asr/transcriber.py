"""Streaming ASR via mlx-audio voxtral_realtime."""

import numpy as np

from disco.asr.hallucination import is_hallucination


def _normalize_audio(samples: np.ndarray) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    return samples


def _result_text(result) -> str:
    if isinstance(result, str):
        return result
    return getattr(result, "text", "") or ""


class StreamingTranscription:
    """One streaming utterance.

    Drive from a single thread: feed() pushes samples (cheap), step()
    runs a bounded amount of MLX work and accumulates emitted text.
    """

    def __init__(self, session, max_decode_tokens: int = 8):
        self._session = session
        self._max_decode_tokens = max_decode_tokens
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    @property
    def done(self) -> bool:
        return self._session.done

    def feed(self, samples: np.ndarray) -> None:
        self._session.feed(_normalize_audio(samples))

    def step(self) -> bool:
        """Decode up to ``max_decode_tokens`` tokens; return True if text changed."""
        deltas = self._session.step(max_decode_tokens=self._max_decode_tokens)
        if not deltas:
            return False
        text = self._decode_generated()
        if not text:
            text = self._text + "".join(deltas)
        if text == self._text:
            return False
        self._text = text
        return True

    def _decode_generated(self) -> str:
        generated = getattr(self._session, "generated", None)
        model = getattr(self._session, "model", None)
        tokenizer = getattr(model, "_tokenizer", None)
        config = getattr(model, "config", None)
        if generated is None or tokenizer is None:
            return ""
        eos = getattr(config, "eos_token_id", None)
        tokens = [int(token) for token in generated if eos is None or token != eos]
        if not tokens:
            return ""
        return tokenizer.decode(tokens)

    def close(self) -> None:
        self._session.close()

    def drain(self) -> None:
        """Step until the session reports done."""
        while not self._session.done:
            self.step()


class Transcriber:
    """Loads the Voxtral realtime model and hands out streaming sessions."""

    def __init__(
        self,
        model_name: str = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
        sample_rate: int = 16000,
        transcription_delay_ms: int = 480,
        max_decode_tokens: int = 8,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.transcription_delay_ms = transcription_delay_ms
        self.max_decode_tokens = max_decode_tokens
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from mlx_audio.stt import load as load_asr

            print(f"Loading ASR model: {self.model_name}")
            self._model = load_asr(self.model_name)
            print("ASR model loaded!")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def start_session(self) -> StreamingTranscription:
        session = self.model.create_streaming_session(
            transcription_delay_ms=self.transcription_delay_ms,
        )
        return StreamingTranscription(session, max_decode_tokens=self.max_decode_tokens)

    def transcribe_once(self, samples: np.ndarray) -> str:
        """Transcribe a complete audio span without streaming session state."""
        result = self.model.generate(
            _normalize_audio(samples),
            max_tokens=4096,
            temperature=0.0,
            verbose=False,
            stream=False,
            transcription_delay_ms=self.transcription_delay_ms,
        )
        return _result_text(result).strip()
