"""Sortformer-led turn detector.

Subscribes to ``SpeakerActivity`` events from the Diarizer and decides
when utterances start/end and when the primary speaker changes. Replaces
silero VAD with the diarizer's own per-chunk view of who is active.

Trade-off vs the previous TurnDetector: SpeechStart fires one chunk
later than silero would (you wait for the diarizer to process the chunk
before you know whether anything was said), but turn boundaries align
exactly with the diarizer's segments so the speaker label on Final is
always coherent with the audio span.
"""

from disco.runtime.debug import log as debug_log
from disco.runtime.events import (
    EventBus,
    SpeakerActivity,
    SpeakerChange,
    SpeechEnd,
    SpeechStart,
)


class Coordinator:
    """Turn-detection state machine driven by SpeakerActivity events.

    Subscribers register themselves on the bus; this class just publishes.
    All state lives in the bus-callback thread (the Diarizer worker
    thread, since that's what fires SpeakerActivity), so no locking.
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        silence_chunks_for_end: int = 5,
        min_utterance_chunks: int = 5,
        speaker_change_chunks: int = 4,
    ):
        """
        Args:
            bus: event bus to publish lifecycle events on.
            silence_chunks_for_end: consecutive silent chunks before
                SpeechEnd fires. At ~0.1 s / chunk, 5 ≈ 500 ms.
            min_utterance_chunks: don't end an utterance shorter than this.
            speaker_change_chunks: consecutive chunks of a different
                primary before SpeakerChange fires.
        """
        self.bus = bus
        self.silence_chunks_for_end = silence_chunks_for_end
        self.min_utterance_chunks = min_utterance_chunks
        self.speaker_change_chunks = speaker_change_chunks

        self._state: str = "quiet"
        self._silence_run = 0
        self._utterance_chunks = 0
        self._bound_speaker: int | None = None
        self._change_candidate: int | None = None
        self._change_run = 0
        self._utterance_start_t: float = 0.0

    def on_activity(self, event: SpeakerActivity) -> None:
        if self._state == "quiet":
            if event.primary_speaker is not None:
                self._state = "speaking"
                self._silence_run = 0
                self._utterance_chunks = 1
                self._bound_speaker = event.primary_speaker
                self._change_candidate = None
                self._change_run = 0
                self._utterance_start_t = event.t_start
                debug_log(
                    "coord",
                    f"SpeechStart t={event.t_start:.2f}",
                    f"S{event.primary_speaker}",
                    f"all={event.all_speakers}",
                )
                self.bus.publish(SpeechStart(t=event.t_start))
            return

        # state == "speaking"
        self._utterance_chunks += 1

        if event.primary_speaker is None:
            self._silence_run += 1
            self._change_candidate = None
            self._change_run = 0
            if (
                self._silence_run >= self.silence_chunks_for_end
                and self._utterance_chunks >= self.min_utterance_chunks
            ):
                debug_log(
                    "coord",
                    f"SpeechEnd t={event.t_end:.2f}",
                    f"utt_chunks={self._utterance_chunks}",
                    f"bound=S{self._bound_speaker}",
                )
                self.bus.publish(SpeechEnd(t=event.t_end))
                self._reset()
            return

        # Active speaker present.
        self._silence_run = 0

        if event.primary_speaker == self._bound_speaker:
            self._change_candidate = None
            self._change_run = 0
            return

        # Different primary speaker — start or extend candidate.
        if event.primary_speaker == self._change_candidate:
            self._change_run += 1
        else:
            self._change_candidate = event.primary_speaker
            self._change_run = 1

        if self._change_run >= self.speaker_change_chunks:
            debug_log(
                "coord",
                f"SpeakerChange S{self._bound_speaker}->S{event.primary_speaker}",
                f"t={event.t_end:.2f}",
                f"held={self._change_run} chunks",
            )
            self.bus.publish(
                SpeakerChange(
                    from_speaker=self._bound_speaker,
                    to_speaker=event.primary_speaker,
                    t=event.t_end,
                )
            )
            # Stay in "speaking", but rebind to the new speaker so the
            # next chunks belong to a fresh session whose primary is them.
            self._bound_speaker = event.primary_speaker
            self._utterance_chunks = 0
            self._silence_run = 0
            self._change_candidate = None
            self._change_run = 0
            self._utterance_start_t = event.t_end

    def _reset(self) -> None:
        self._state = "quiet"
        self._silence_run = 0
        self._utterance_chunks = 0
        self._bound_speaker = None
        self._change_candidate = None
        self._change_run = 0
