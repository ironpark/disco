"""Smart Turn endpoint gate backed by mlx-audio."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from disco.audio.frame import AudioRingBuffer
from disco.runtime.debug import log as debug_log


@dataclass(frozen=True)
class SmartTurnDecision:
    complete: bool
    probability: float


class SmartTurnEndpoint:
    """Decide whether a silence boundary is a real turn endpoint.

    Smart Turn is not a replacement for VAD or diarization. It is run after
    lightweight VAD has detected silence, using the full current-turn audio
    from the ring buffer.
    """

    def __init__(
        self,
        *,
        model_name: str = "mlx-community/smart-turn-v3",
        sample_rate: int = 16000,
        threshold: float = 0.5,
        strict: bool = True,
        model: Any | None = None,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.strict = strict
        self._model = model

    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_audio.vad import load as load_vad

        print(f"Loading Smart Turn model: {self.model_name}")
        self._model = load_vad(self.model_name, strict=self.strict)
        print("Smart Turn model loaded!")

    def predict(self, audio: np.ndarray) -> SmartTurnDecision:
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if len(audio) == 0:
            return SmartTurnDecision(complete=True, probability=1.0)

        self.load()
        result = self._model.predict_endpoint(
            audio,
            sample_rate=self.sample_rate,
            threshold=self.threshold,
        )
        return SmartTurnDecision(
            complete=bool(result.prediction),
            probability=float(result.probability),
        )

    def should_end(
        self,
        ring_buffer: AudioRingBuffer,
        turn_start_t: float,
        observed_end_t: float,
    ) -> bool:
        audio = ring_buffer.span(turn_start_t, observed_end_t)
        decision = self.predict(audio)
        debug_log(
            "smart_turn",
            f"complete={decision.complete}",
            f"prob={decision.probability:.3f}",
            f"span=({turn_start_t:.2f},{observed_end_t:.2f})",
            f"samples={len(audio)}",
        )
        return decision.complete
