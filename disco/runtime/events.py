"""Pipeline events and a small synchronous EventBus.

All ``t``/``span`` values are diarizer-time seconds (``Diarizer.elapsed_seconds()``),
the project's single audio clock.
"""

import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TurnRef:
    """Transcript ownership metadata shared across enriched stages."""

    utterance_id: int
    utterance_ids: tuple[int, ...]
    span: tuple[float, float]
    speaker: int | None = None

    @classmethod
    def single(
        cls,
        *,
        utterance_id: int,
        span: tuple[float, float],
        speaker: int | None = None,
    ) -> "TurnRef":
        return cls(
            utterance_id=utterance_id,
            utterance_ids=(utterance_id,),
            span=span,
            speaker=speaker,
        )

    def with_speaker(self, speaker: int | None) -> "TurnRef":
        return TurnRef(
            utterance_id=self.utterance_id,
            utterance_ids=self.utterance_ids,
            span=self.span,
            speaker=speaker,
        )

    def merged_with(self, other: "TurnRef", *, speaker: int | None) -> "TurnRef":
        utterance_ids: list[int] = []
        for value in (*self.utterance_ids, *other.utterance_ids):
            if value not in utterance_ids:
                utterance_ids.append(value)
        return TurnRef(
            utterance_id=self.utterance_id,
            utterance_ids=tuple(utterance_ids),
            span=(self.span[0], other.span[1]),
            speaker=speaker,
        )


@dataclass(frozen=True)
class SpeakerActivity:
    """Per-chunk diarizer output.

    ``primary_speaker`` is the speaker with the most overlap inside the
    chunk window, or ``None`` if no segments were emitted for this chunk
    (i.e. silence as far as the model can tell). ``all_speakers`` lists
    every active speaker so consumers can detect overlap.
    """

    t_start: float
    t_end: float
    primary_speaker: int | None
    all_speakers: tuple[int, ...]


@dataclass(frozen=True)
class VadActivity:
    """Speech activity from a VAD on the shared audio clock."""

    t_start: float
    t_end: float
    speech: bool
    confidence: float | None = None


@dataclass(frozen=True)
class SpeechStart:
    t: float
    utterance_id: int
    speaker: int | None = None


@dataclass(frozen=True)
class SpeechEnd:
    t: float
    utterance_id: int
    speaker: int | None = None


@dataclass(frozen=True)
class SpeakerChange:
    """Sustained speaker change detected within an open utterance."""

    from_speaker: int | None
    to_speaker: int
    t: float
    utterance_id: int
    next_utterance_id: int


@dataclass(frozen=True)
class SpeakerBind:
    """Late speaker attribution for an already-open utterance."""

    utterance_id: int
    speaker: int


@dataclass(frozen=True)
class Interim:
    text: str
    span: tuple[float, float]
    utterance_id: int
    speaker: int | None = None


@dataclass(frozen=True)
class EnrichedInterim:
    text: str
    span: tuple[float, float]
    utterance_id: int
    speaker: int | None = None
    translation: str | None = None


@dataclass(frozen=True)
class Final:
    """Raw final emitted by the transcriber worker; no translation yet."""

    text: str
    span: tuple[float, float]
    utterance_id: int
    speaker: int | None = None


@dataclass(frozen=True)
class FinalDiscarded:
    """A turn finalized without user-visible transcript text."""

    span: tuple[float, float]
    utterance_id: int
    speaker: int | None = None
    reason: str = "empty"


@dataclass(frozen=True)
class LabeledFinal:
    """Final with speaker attribution; translation is added downstream."""

    text: str
    ref: TurnRef

    @property
    def span(self) -> tuple[float, float]:
        return self.ref.span

    @property
    def utterance_id(self) -> int:
        return self.ref.utterance_id

    @property
    def utterance_ids(self) -> tuple[int, ...]:
        return self.ref.utterance_ids

    @property
    def speaker(self) -> int | None:
        return self.ref.speaker


@dataclass(frozen=True)
class EnrichedFinal:
    text: str
    ref: TurnRef
    translation: str | None = None

    @property
    def span(self) -> tuple[float, float]:
        return self.ref.span

    @property
    def utterance_id(self) -> int:
        return self.ref.utterance_id

    @property
    def utterance_ids(self) -> tuple[int, ...]:
        return self.ref.utterance_ids

    @property
    def speaker(self) -> int | None:
        return self.ref.speaker


@dataclass(frozen=True)
class QueueOverflow:
    """A bounded worker queue dropped a chunk."""

    component: str
    depth: int


@dataclass(frozen=True)
class WorkerBackpressure:
    """A worker hit an internal pressure threshold."""

    component: str
    reason: str
    depth: int


class EventBus:
    """Synchronous, thread-safe pub/sub keyed by event class.

    ``publish`` invokes subscribers on the caller's thread. The bus only
    guards the subscriber registry — subscriber callbacks must be safe
    against concurrent invocation themselves.
    """

    def __init__(self) -> None:
        self._subscribers: dict[type, list[Callable]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, event_type: type, callback: Callable) -> None:
        with self._lock:
            self._subscribers[event_type].append(callback)

    def publish(self, event) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(type(event), ()))
        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:
                print(f"EventBus subscriber error ({type(event).__name__}): {exc}")
