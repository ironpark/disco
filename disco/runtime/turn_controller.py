"""Single-threaded owner for turn-reduction activity events."""

import queue
import threading

from disco.runtime.coordinator import Coordinator
from disco.runtime.events import (
    EventBus,
    SpeakerActivity,
    VadActivity,
    WorkerBackpressure,
)


_STOP = object()


class TurnController:
    """Serialize VAD and diarization activity before mutating turn state.

    VAD and diarization run on separate worker threads and publish activity
    at different latencies. The controller is the only runtime component that
    calls the Coordinator, so turn lifecycle decisions are made in one place
    and in one thread.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        coordinator: Coordinator,
        max_queue: int = 0,
    ):
        self.bus = bus
        self.coordinator = coordinator
        self.max_queue = max_queue
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._running = False
        self._enqueue(_STOP, publish_overflow=False)
        self._thread.join(timeout=2.0)
        self._thread = None

    def submit_vad(self, event: VadActivity) -> None:
        self._enqueue(event)

    def submit_speaker(self, event: SpeakerActivity) -> None:
        self._enqueue(event)

    def snapshot(self) -> dict[str, int]:
        return {"queue_depth": self._queue.qsize()}

    def _enqueue(self, item, *, publish_overflow: bool = True) -> None:
        if not self._running and item is not _STOP:
            return
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass

        if item is _STOP:
            try:
                self._queue.put(item, timeout=0.25)
            except queue.Full:
                pass
            return

        try:
            self._queue.put(item, timeout=0.25)
        except queue.Full:
            if publish_overflow:
                self.bus.publish(
                    WorkerBackpressure(
                        component="turn_controller",
                        reason="queue_full",
                        depth=self._queue.qsize(),
                    )
                )

    def _worker(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._running:
                    continue
                break

            if item is _STOP:
                break
            if isinstance(item, VadActivity):
                self._publish_all(self.coordinator.reduce_vad_activity(item))
            elif isinstance(item, SpeakerActivity):
                self._publish_all(self.coordinator.reduce_speaker_activity(item))

    def _publish_all(self, events: tuple[object, ...]) -> None:
        for event in events:
            self.bus.publish(event)
