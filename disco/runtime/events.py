"""Pipeline events and a small synchronous EventBus.

All ``t``/``span`` values are diarizer-time seconds (``Diarizer.elapsed_seconds()``),
the project's single audio clock.
"""

import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass


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
class SpeechStart:
    t: float
    utterance_id: int


@dataclass(frozen=True)
class SpeechEnd:
    t: float
    utterance_id: int


@dataclass(frozen=True)
class SpeakerChange:
    """Sustained speaker change detected within an open utterance."""

    from_speaker: int | None
    to_speaker: int
    t: float
    utterance_id: int
    next_utterance_id: int


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
    """Raw final emitted by the transcriber worker; no speaker/translation yet."""

    text: str
    span: tuple[float, float]
    utterance_id: int


@dataclass(frozen=True)
class LabeledFinal:
    """Final with speaker attribution; translation is added downstream."""

    text: str
    span: tuple[float, float]
    utterance_id: int
    speaker: int | None = None


@dataclass(frozen=True)
class EnrichedFinal:
    text: str
    span: tuple[float, float]
    utterance_id: int
    speaker: int | None = None
    translation: str | None = None


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
