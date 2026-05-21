"""Timeline reducer that fuses VAD and diarization activity.

VAD owns speech/no-speech boundaries when available. Sortformer diarization
owns speaker binding and speaker-change splits. The reducer keeps a single
Turn as the source of truth and emits lifecycle events for downstream ASR.
"""

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from disco.runtime.debug import log as debug_log
from disco.runtime.events import (
    EventBus,
    SpeakerActivity,
    SpeakerBind,
    SpeakerChange,
    SpeechEnd,
    SpeechStart,
    VadActivity,
)


@dataclass
class _Turn:
    utterance_id: int
    start_t: float
    speaker: int | None


@dataclass
class _ActivityFrame:
    t_start: float
    t_end: float
    vad_speech: bool | None = None
    speech_prob: float | None = None
    primary_speaker: int | None = None
    all_speakers: tuple[int, ...] = ()

    @property
    def diar_speech(self) -> bool:
        return self.primary_speaker is not None


class Coordinator:
    """Fuse activity events into turn lifecycle events.

    ``on_vad_activity`` and ``on_activity`` can be called from different
    worker threads, so state mutations are protected by a lock. All thresholds
    are converted to seconds internally; the public constructor still accepts
    chunk counts for compatibility with Runtime's existing configuration.
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        silence_chunks_for_end: int = 5,
        min_utterance_chunks: int = 5,
        speaker_change_chunks: int = 4,
        same_speaker_bridge_chunks: int = 8,
        chunk_duration_s: float = 0.1,
        max_utterance_duration_s: float = 10.0,
        vad_retention_s: float = 30.0,
        endpoint_complete: Callable[[float, float], bool] | None = None,
    ):
        self.bus = bus
        self.silence_s = silence_chunks_for_end * chunk_duration_s
        self.min_utterance_s = min_utterance_chunks * chunk_duration_s
        self.speaker_change_hold_s = speaker_change_chunks * chunk_duration_s
        self.same_speaker_bridge_s = same_speaker_bridge_chunks * chunk_duration_s
        self.max_utterance_duration_s = max_utterance_duration_s
        self.vad_retention_s = vad_retention_s
        self.endpoint_complete = endpoint_complete

        self._lock = threading.RLock()
        self._turn: _Turn | None = None
        self._next_utterance_id = 1
        self._latest_diar: _ActivityFrame | None = None
        self._vad_frames: deque[_ActivityFrame] = deque()
        self._silence_start_t: float | None = None
        self._pending_end_t: float | None = None
        self._pending_end_from_vad = False
        self._change_candidate: int | None = None
        self._change_candidate_start_t: float | None = None
        self._emitted: list[object] | None = None

    def set_endpoint_complete(
        self,
        endpoint_complete: Callable[[float, float], bool] | None,
    ) -> None:
        self.endpoint_complete = endpoint_complete

    def on_vad_activity(self, event: VadActivity) -> None:
        for emitted in self.reduce_vad_activity(event):
            self.bus.publish(emitted)

    def reduce_vad_activity(self, event: VadActivity) -> tuple[object, ...]:
        frame = _ActivityFrame(
            t_start=event.t_start,
            t_end=event.t_end,
            vad_speech=event.speech,
            speech_prob=event.confidence,
        )
        with self._lock:
            self._begin_reduce()
            try:
                self._remember_vad(frame)
                if event.speech:
                    if self._turn is None:
                        self._start_turn(
                            t=event.t_start,
                            speaker=None,
                            all_speakers=(),
                            reason="vad",
                        )
                    else:
                        self._resume_speech(frame, reason="vad")
                        self._maybe_split_long_turn(frame)
                    return self._finish_reduce()

                if self._turn is not None:
                    self._observe_silence(frame, reason="vad_silence")
                return self._finish_reduce()
            finally:
                self._emitted = None

    def on_activity(self, event: SpeakerActivity) -> None:
        for emitted in self.reduce_speaker_activity(event):
            self.bus.publish(emitted)

    def reduce_speaker_activity(self, event: SpeakerActivity) -> tuple[object, ...]:
        frame = _ActivityFrame(
            t_start=event.t_start,
            t_end=event.t_end,
            primary_speaker=event.primary_speaker,
            all_speakers=event.all_speakers,
        )
        with self._lock:
            self._begin_reduce()
            try:
                self._latest_diar = frame

                if event.primary_speaker is None:
                    self._observe_diarizer_silence(frame)
                    return self._finish_reduce()

                if self._turn is None:
                    vad_speech = self._vad_speech_for(frame)
                    if vad_speech is False:
                        return self._finish_reduce()
                    self._start_turn(
                        t=event.t_start,
                        speaker=event.primary_speaker,
                        all_speakers=event.all_speakers,
                        reason="diar",
                    )
                    return self._finish_reduce()

                self._observe_speaker(frame)
                self._maybe_split_long_turn(frame)
                return self._finish_reduce()
            finally:
                self._emitted = None

    def _observe_diarizer_silence(self, frame: _ActivityFrame) -> None:
        if self._turn is None:
            return
        vad_speech = self._vad_speech_for(frame)
        if vad_speech is True:
            self._clear_silence()
            return
        if vad_speech is False:
            return
        self._observe_silence(frame, reason="diar_silence")

    def _observe_speaker(self, frame: _ActivityFrame) -> None:
        assert self._turn is not None
        speaker = frame.primary_speaker
        assert speaker is not None

        if self._turn.speaker is None:
            self._bind_speaker(speaker)
            return

        if self._pending_end_t is not None:
            vad_speech = self._vad_speech_for(frame)
            if self._pending_end_from_vad and vad_speech is not True:
                return
            if vad_speech is False:
                return
            if speaker != self._turn.speaker:
                self._publish_end(
                    self._pending_end_t,
                    reason=f"new_speaker=S{speaker}",
                )
                self._start_turn(
                    t=frame.t_start,
                    speaker=speaker,
                    all_speakers=frame.all_speakers,
                    reason="after_pending_end",
                )
                return
            self._resume_speech(frame, reason="diar")

        if speaker == self._turn.speaker:
            self._clear_speaker_change()
            return

        overlapping_bound_speaker = (
            self._turn.speaker in frame.all_speakers and len(frame.all_speakers) > 1
        )
        if overlapping_bound_speaker:
            debug_log(
                "coord",
                f"overlap candidate=S{speaker}",
                f"bound=S{self._turn.speaker}",
                f"active={frame.all_speakers}",
            )

        if speaker == self._change_candidate:
            candidate_start = self._change_candidate_start_t or frame.t_start
        else:
            self._change_candidate = speaker
            self._change_candidate_start_t = frame.t_start
            candidate_start = frame.t_start

        if frame.t_end - candidate_start + 1e-9 >= self.speaker_change_hold_s:
            self._publish_speaker_change(
                speaker=speaker,
                split_t=candidate_start,
                frame=frame,
            )

    def _maybe_split_long_turn(self, frame: _ActivityFrame) -> None:
        if self._turn is None:
            return
        if self.max_utterance_duration_s <= 0:
            return
        if frame.t_end - self._turn.start_t + 1e-9 < self.max_utterance_duration_s:
            return
        self._publish_turn_split(
            t=frame.t_end,
            speaker=self._turn.speaker,
            reason="max_utterance_duration",
        )

    def _observe_silence(self, frame: _ActivityFrame, *, reason: str) -> None:
        assert self._turn is not None
        self._clear_speaker_change()
        if frame.t_end - self._turn.start_t < self.min_utterance_s:
            return
        if self._silence_start_t is None:
            self._silence_start_t = frame.t_start

        silence_elapsed = frame.t_end - self._silence_start_t
        if silence_elapsed + 1e-9 < self.silence_s:
            return

        if self._pending_end_t is None:
            self._pending_end_t = self._silence_start_t
            self._pending_end_from_vad = reason == "vad_silence"
            debug_log(
                "coord",
                f"SpeechEnd pending t={self._pending_end_t:.2f}",
                f"reason={reason}",
                f"silence={silence_elapsed:.2f}s",
                f"utt={self._turn.utterance_id}",
                f"bound={'S' + str(self._turn.speaker) if self._turn.speaker is not None else '?'}",
            )

        if silence_elapsed + 1e-9 >= self.silence_s + self.same_speaker_bridge_s:
            self._publish_end(
                self._pending_end_t,
                reason="bridge_expired",
                endpoint_t=frame.t_end,
            )

    def _resume_speech(self, frame: _ActivityFrame, *, reason: str) -> None:
        if self._pending_end_t is not None:
            debug_log(
                "coord",
                f"SpeechEnd bridged t={self._pending_end_t:.2f}",
                f"reason={reason}",
                f"span=({frame.t_start:.2f},{frame.t_end:.2f})",
            )
        self._silence_start_t = None
        self._pending_end_t = None
        self._pending_end_from_vad = False

    def _start_turn(
        self,
        *,
        t: float,
        speaker: int | None,
        all_speakers: tuple[int, ...],
        reason: str,
    ) -> None:
        turn = _Turn(
            utterance_id=self._allocate_utterance_id(),
            start_t=t,
            speaker=speaker,
        )
        self._turn = turn
        self._silence_start_t = None
        self._pending_end_t = None
        self._pending_end_from_vad = False
        self._clear_speaker_change()
        debug_log(
            "coord",
            f"SpeechStart t={t:.2f}",
            f"utt={turn.utterance_id}",
            f"speaker={'S' + str(speaker) if speaker is not None else '?'}",
            f"all={all_speakers}",
            f"reason={reason}",
        )
        self._emit(
            SpeechStart(t=t, utterance_id=turn.utterance_id, speaker=speaker)
        )

    def _publish_end(
        self,
        t: float,
        *,
        reason: str,
        endpoint_t: float | None = None,
    ) -> bool:
        if self._turn is None:
            return False
        turn = self._turn
        if (
            reason == "bridge_expired"
            and self.endpoint_complete is not None
            and not self._endpoint_allows_end(turn, endpoint_t or t)
        ):
            return False
        debug_log(
            "coord",
            f"SpeechEnd t={t:.2f}",
            f"utt={turn.utterance_id}",
            f"reason={reason}",
            f"bound={'S' + str(turn.speaker) if turn.speaker is not None else '?'}",
        )
        self._emit(
            SpeechEnd(t=t, utterance_id=turn.utterance_id, speaker=turn.speaker)
        )
        self._reset_turn()
        return True

    def _publish_speaker_change(
        self, *, speaker: int, split_t: float, frame: _ActivityFrame
    ) -> None:
        assert self._turn is not None
        old_turn = self._turn
        next_utterance_id = self._allocate_utterance_id()
        debug_log(
            "coord",
            f"SpeakerChange S{old_turn.speaker}->S{speaker}",
            f"t={split_t:.2f}",
            f"detected={frame.t_end:.2f}",
            f"utt={old_turn.utterance_id}->{next_utterance_id}",
            f"hold={frame.t_end - split_t:.2f}s",
        )
        self._emit(
            SpeakerChange(
                from_speaker=old_turn.speaker,
                to_speaker=speaker,
                t=split_t,
                utterance_id=old_turn.utterance_id,
                next_utterance_id=next_utterance_id,
            )
        )
        self._turn = _Turn(
            utterance_id=next_utterance_id,
            start_t=split_t,
            speaker=speaker,
        )
        self._silence_start_t = None
        self._pending_end_t = None
        self._pending_end_from_vad = False
        self._clear_speaker_change()

    def _publish_turn_split(
        self,
        *,
        t: float,
        speaker: int | None,
        reason: str,
    ) -> None:
        if self._turn is None:
            return
        old_turn = self._turn
        next_utterance_id = self._allocate_utterance_id()
        debug_log(
            "coord",
            f"TurnSplit t={t:.2f}",
            f"utt={old_turn.utterance_id}->{next_utterance_id}",
            f"reason={reason}",
            f"speaker={'S' + str(speaker) if speaker is not None else '?'}",
        )
        self._emit(
            SpeechEnd(t=t, utterance_id=old_turn.utterance_id, speaker=speaker)
        )
        self._turn = _Turn(
            utterance_id=next_utterance_id,
            start_t=t,
            speaker=speaker,
        )
        self._silence_start_t = None
        self._pending_end_t = None
        self._pending_end_from_vad = False
        self._clear_speaker_change()
        self._emit(
            SpeechStart(t=t, utterance_id=next_utterance_id, speaker=speaker)
        )

    def _bind_speaker(self, speaker: int) -> None:
        if self._turn is None:
            return
        self._turn.speaker = speaker
        debug_log("coord", f"SpeakerBind utt={self._turn.utterance_id}", f"S{speaker}")
        self._emit(SpeakerBind(utterance_id=self._turn.utterance_id, speaker=speaker))

    def _allocate_utterance_id(self) -> int:
        utterance_id = self._next_utterance_id
        self._next_utterance_id += 1
        return utterance_id

    def _begin_reduce(self) -> None:
        self._emitted = []

    def _finish_reduce(self) -> tuple[object, ...]:
        return tuple(self._emitted or ())

    def _emit(self, event: object) -> None:
        if self._emitted is None:
            self.bus.publish(event)
            return
        self._emitted.append(event)

    def _endpoint_allows_end(self, turn: _Turn, observed_end_t: float) -> bool:
        try:
            complete = self.endpoint_complete(turn.start_t, observed_end_t)
        except Exception as exc:
            print(f"Endpoint detector error: {exc}")
            return True
        if complete:
            return True
        debug_log(
            "coord",
            "SpeechEnd vetoed",
            f"utt={turn.utterance_id}",
            f"span=({turn.start_t:.2f},{observed_end_t:.2f})",
            "reason=smart_turn_incomplete",
        )
        self._silence_start_t = None
        self._pending_end_t = None
        self._pending_end_from_vad = False
        self._clear_speaker_change()
        return False

    def _remember_vad(self, frame: _ActivityFrame) -> None:
        self._vad_frames.append(frame)
        cutoff = frame.t_end - self.vad_retention_s
        while self._vad_frames and self._vad_frames[0].t_end < cutoff:
            self._vad_frames.popleft()

    def _vad_speech_for(self, frame: _ActivityFrame) -> bool | None:
        """Return the dominant VAD state overlapping this frame's time range."""

        if not self._vad_frames:
            return None

        speech_s = 0.0
        silence_s = 0.0
        for vad_frame in reversed(self._vad_frames):
            if vad_frame.t_end <= frame.t_start:
                break
            overlap = min(vad_frame.t_end, frame.t_end) - max(
                vad_frame.t_start,
                frame.t_start,
            )
            if overlap <= 0:
                continue
            if vad_frame.vad_speech:
                speech_s += overlap
            else:
                silence_s += overlap

        if speech_s == 0.0 and silence_s == 0.0:
            return None
        return speech_s >= silence_s

    def _clear_silence(self) -> None:
        self._silence_start_t = None
        self._pending_end_t = None
        self._pending_end_from_vad = False

    def _clear_speaker_change(self) -> None:
        self._change_candidate = None
        self._change_candidate_start_t = None

    def _reset_turn(self) -> None:
        self._turn = None
        self._silence_start_t = None
        self._pending_end_t = None
        self._pending_end_from_vad = False
        self._clear_speaker_change()
