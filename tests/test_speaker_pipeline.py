import time
import unittest

import numpy as np

from disco.audio.frame import AudioFrame, AudioRingBuffer
from disco.runtime.coordinator import Coordinator
from disco.runtime.enricher import FinalEnricher
from disco.runtime.events import (
    EventBus,
    Final,
    FinalDiscarded,
    LabeledFinal,
    SpeakerBind,
    SpeakerActivity,
    SpeakerChange,
    SpeechEnd,
    SpeechStart,
    VadActivity,
)
from disco.runtime.transcriber_worker import TranscriberWorker
from disco.runtime.turn_controller import TurnController
from disco.vad import SileroVad, SmartTurnEndpoint


class FakeDiarizer:
    def __init__(self, speaker: int | None):
        self.speaker = speaker

    def dominant_speaker_in(self, t_start: float, t_end: float) -> int | None:
        return self.speaker

    def elapsed_seconds(self) -> float:
        return 0.0


class FakeStreamingSession:
    def __init__(self):
        self.sample_count = 0
        self.text = ""
        self.done = False

    def feed(self, audio) -> None:
        self.sample_count += len(audio)
        self.text = f"samples:{self.sample_count}"

    def step(self) -> bool:
        return True

    def close(self) -> None:
        self.text = f"samples:{self.sample_count}"
        self.done = True

    def drain(self) -> None:
        self.done = True


class FakeNoTextSession(FakeStreamingSession):
    def feed(self, audio) -> None:
        self.sample_count += len(audio)
        self.text = "."

    def close(self) -> None:
        self.text = "."
        self.done = True


class FakeNeverDoneSession(FakeStreamingSession):
    def feed(self, audio) -> None:
        self.sample_count += len(audio)
        self.text = f"stuck:{self.sample_count}"

    def close(self) -> None:
        self.text = f"stuck:{self.sample_count}"
        self.done = False

    def drain(self) -> None:
        self.done = True


class FakeTranscriber:
    def load(self) -> None:
        pass

    def start_session(self) -> FakeStreamingSession:
        return FakeStreamingSession()


class FakeNoTextTranscriber(FakeTranscriber):
    def start_session(self) -> FakeNoTextSession:
        return FakeNoTextSession()


class FakeNeverDoneTranscriber(FakeTranscriber):
    def start_session(self) -> FakeNeverDoneSession:
        return FakeNeverDoneSession()


class FakeOneShotTranscriber(FakeTranscriber):
    def __init__(self):
        self.one_shot_lengths: list[int] = []

    def transcribe_once(self, audio) -> str:
        self.one_shot_lengths.append(len(audio))
        return f"oneshot:{len(audio)}"


class FakeSileroState:
    pass


class FakeSileroConfig:
    threshold = 0.5

    class branch_16k:
        chunk_size = 512


class FakeSileroModel:
    config = FakeSileroConfig()

    def __init__(self, probabilities: list[float]):
        self.probabilities = probabilities
        self.index = 0

    def initial_state(self, sample_rate: int):
        return FakeSileroState()

    def feed(self, chunk, state, sample_rate: int):
        probability = self.probabilities[min(self.index, len(self.probabilities) - 1)]
        self.index += 1
        return np.array([[probability]], dtype=np.float32), state


class FakeSmartTurnResult:
    def __init__(self, prediction: int, probability: float):
        self.prediction = prediction
        self.probability = probability


class FakeSmartTurnModel:
    def __init__(self, predictions: list[int]):
        self.predictions = predictions
        self.calls: list[tuple[int, int, float]] = []

    def predict_endpoint(self, audio, sample_rate: int, threshold: float):
        self.calls.append((len(audio), sample_rate, threshold))
        prediction = self.predictions[min(len(self.calls) - 1, len(self.predictions) - 1)]
        probability = 0.9 if prediction else 0.1
        return FakeSmartTurnResult(prediction, probability)


def make_frame(seq: int, t_start: float, t_end: float) -> AudioFrame:
    sample_rate = 16000
    samples = np.ones(round((t_end - t_start) * sample_rate), dtype=np.float32)
    return AudioFrame(
        seq=seq,
        t_start=t_start,
        t_end=t_end,
        samples=samples,
        sample_rate=sample_rate,
    )


class SpeakerPipelineTest(unittest.TestCase):
    def test_speaker_change_splits_at_candidate_start(self) -> None:
        bus = EventBus()
        starts: list[SpeechStart] = []
        changes: list[SpeakerChange] = []
        bus.subscribe(SpeechStart, starts.append)
        bus.subscribe(SpeakerChange, changes.append)
        coordinator = Coordinator(
            bus,
            silence_chunks_for_end=5,
            min_utterance_chunks=1,
            speaker_change_chunks=3,
            same_speaker_bridge_chunks=8,
        )

        coordinator.on_activity(
            SpeakerActivity(0.0, 0.1, primary_speaker=0, all_speakers=(0,))
        )
        coordinator.on_activity(
            SpeakerActivity(0.1, 0.2, primary_speaker=1, all_speakers=(1,))
        )
        coordinator.on_activity(
            SpeakerActivity(0.2, 0.3, primary_speaker=1, all_speakers=(1,))
        )
        coordinator.on_activity(
            SpeakerActivity(0.3, 0.4, primary_speaker=1, all_speakers=(1,))
        )

        self.assertEqual(1, len(starts))
        self.assertEqual(0, starts[0].speaker)
        self.assertEqual(1, len(changes))
        self.assertEqual(0.1, changes[0].t)
        self.assertEqual(0, changes[0].from_speaker)
        self.assertEqual(1, changes[0].to_speaker)

    def test_coordinator_reducer_returns_events_without_publishing(self) -> None:
        bus = EventBus()
        starts: list[SpeechStart] = []
        bus.subscribe(SpeechStart, starts.append)
        coordinator = Coordinator(bus, min_utterance_chunks=1)

        events = coordinator.reduce_vad_activity(
            VadActivity(0.0, 0.1, speech=True, confidence=1.0)
        )

        self.assertEqual([], starts)
        self.assertEqual(1, len(events))
        self.assertIsInstance(events[0], SpeechStart)
        bus.publish(events[0])
        self.assertEqual(1, len(starts))

    def test_brief_overlap_does_not_trigger_speaker_change(self) -> None:
        bus = EventBus()
        changes: list[SpeakerChange] = []
        bus.subscribe(SpeakerChange, changes.append)
        coordinator = Coordinator(
            bus,
            min_utterance_chunks=1,
            speaker_change_chunks=3,
        )

        coordinator.on_activity(
            SpeakerActivity(0.0, 0.1, primary_speaker=0, all_speakers=(0,))
        )
        coordinator.on_activity(
            SpeakerActivity(0.1, 0.2, primary_speaker=1, all_speakers=(0, 1))
        )
        coordinator.on_activity(
            SpeakerActivity(0.2, 0.3, primary_speaker=0, all_speakers=(0,))
        )

        self.assertEqual([], changes)

    def test_sustained_new_primary_splits_even_when_old_speaker_is_listed(self) -> None:
        bus = EventBus()
        changes: list[SpeakerChange] = []
        bus.subscribe(SpeakerChange, changes.append)
        coordinator = Coordinator(
            bus,
            min_utterance_chunks=1,
            speaker_change_chunks=2,
        )

        coordinator.on_activity(
            SpeakerActivity(0.0, 0.1, primary_speaker=0, all_speakers=(0,))
        )
        coordinator.on_activity(
            SpeakerActivity(0.1, 0.2, primary_speaker=1, all_speakers=(0, 1))
        )
        coordinator.on_activity(
            SpeakerActivity(0.2, 0.3, primary_speaker=1, all_speakers=(0, 1))
        )

        self.assertEqual(1, len(changes))
        self.assertEqual(0.1, changes[0].t)
        self.assertEqual(0, changes[0].from_speaker)
        self.assertEqual(1, changes[0].to_speaker)

    def test_long_continuous_turn_is_split_without_silence(self) -> None:
        bus = EventBus()
        starts: list[SpeechStart] = []
        ends: list[SpeechEnd] = []
        bus.subscribe(SpeechStart, starts.append)
        bus.subscribe(SpeechEnd, ends.append)
        coordinator = Coordinator(
            bus,
            min_utterance_chunks=1,
            max_utterance_duration_s=0.3,
        )

        coordinator.on_vad_activity(
            VadActivity(0.0, 0.1, speech=True, confidence=1.0)
        )
        coordinator.on_activity(
            SpeakerActivity(0.0, 0.1, primary_speaker=0, all_speakers=(0,))
        )
        coordinator.on_vad_activity(
            VadActivity(0.1, 0.2, speech=True, confidence=1.0)
        )
        coordinator.on_vad_activity(
            VadActivity(0.2, 0.3, speech=True, confidence=1.0)
        )

        self.assertEqual(2, len(starts))
        self.assertEqual(1, len(ends))
        self.assertEqual(0.3, ends[0].t)
        self.assertEqual(0.3, starts[1].t)
        self.assertEqual(0, starts[1].speaker)

    def test_vad_silence_is_not_bridged_by_late_diarizer_activity(self) -> None:
        bus = EventBus()
        ends: list[SpeechEnd] = []
        bus.subscribe(SpeechEnd, ends.append)
        coordinator = Coordinator(
            bus,
            silence_chunks_for_end=1,
            min_utterance_chunks=1,
            same_speaker_bridge_chunks=1,
        )

        coordinator.on_vad_activity(
            VadActivity(0.0, 0.1, speech=True, confidence=1.0)
        )
        coordinator.on_activity(
            SpeakerActivity(0.0, 0.1, primary_speaker=0, all_speakers=(0,))
        )
        coordinator.on_vad_activity(
            VadActivity(0.1, 0.2, speech=False, confidence=0.0)
        )
        coordinator.on_activity(
            SpeakerActivity(0.2, 0.3, primary_speaker=0, all_speakers=(0,))
        )
        coordinator.on_vad_activity(
            VadActivity(0.2, 0.3, speech=False, confidence=0.0)
        )

        self.assertEqual(1, len(ends))
        self.assertEqual(0.1, ends[0].t)

    def test_vad_can_start_turn_before_diarizer_binds_speaker(self) -> None:
        bus = EventBus()
        starts: list[SpeechStart] = []
        binds: list[SpeakerBind] = []
        bus.subscribe(SpeechStart, starts.append)
        bus.subscribe(SpeakerBind, binds.append)
        coordinator = Coordinator(bus, min_utterance_chunks=1)

        coordinator.on_vad_activity(
            VadActivity(0.0, 0.1, speech=True, confidence=1.0)
        )
        coordinator.on_activity(
            SpeakerActivity(0.1, 0.2, primary_speaker=2, all_speakers=(2,))
        )

        self.assertEqual(1, len(starts))
        self.assertIsNone(starts[0].speaker)
        self.assertEqual(1, len(binds))
        self.assertEqual(starts[0].utterance_id, binds[0].utterance_id)
        self.assertEqual(2, binds[0].speaker)

    def test_late_diarizer_bind_uses_vad_state_for_that_time_range(self) -> None:
        bus = EventBus()
        starts: list[SpeechStart] = []
        binds: list[SpeakerBind] = []
        changes: list[SpeakerChange] = []
        bus.subscribe(SpeechStart, starts.append)
        bus.subscribe(SpeakerBind, binds.append)
        bus.subscribe(SpeakerChange, changes.append)
        coordinator = Coordinator(
            bus,
            silence_chunks_for_end=1,
            min_utterance_chunks=1,
            same_speaker_bridge_chunks=3,
        )

        coordinator.on_vad_activity(
            VadActivity(0.0, 0.1, speech=True, confidence=1.0)
        )
        coordinator.on_vad_activity(
            VadActivity(0.1, 0.2, speech=False, confidence=0.0)
        )
        coordinator.on_activity(
            SpeakerActivity(0.0, 0.1, primary_speaker=2, all_speakers=(2,))
        )

        self.assertEqual(1, len(starts))
        self.assertEqual(1, len(binds))
        self.assertEqual(starts[0].utterance_id, binds[0].utterance_id)
        self.assertEqual(2, binds[0].speaker)
        self.assertEqual([], changes)

    def test_diarizer_activity_inside_known_vad_silence_does_not_start_turn(self) -> None:
        bus = EventBus()
        starts: list[SpeechStart] = []
        bus.subscribe(SpeechStart, starts.append)
        coordinator = Coordinator(bus, min_utterance_chunks=1)

        coordinator.on_vad_activity(
            VadActivity(1.0, 1.1, speech=False, confidence=0.0)
        )
        coordinator.on_activity(
            SpeakerActivity(1.0, 1.1, primary_speaker=0, all_speakers=(0,))
        )

        self.assertEqual([], starts)

    def test_smart_turn_veto_delays_vad_silence_endpoint(self) -> None:
        bus = EventBus()
        ends: list[SpeechEnd] = []
        bus.subscribe(SpeechEnd, ends.append)
        decisions = iter([False, True])
        coordinator = Coordinator(
            bus,
            silence_chunks_for_end=1,
            min_utterance_chunks=1,
            same_speaker_bridge_chunks=1,
            endpoint_complete=lambda _start, _end: next(decisions),
        )

        coordinator.on_vad_activity(
            VadActivity(0.0, 0.1, speech=True, confidence=1.0)
        )
        coordinator.on_vad_activity(
            VadActivity(0.1, 0.2, speech=False, confidence=0.0)
        )
        coordinator.on_vad_activity(
            VadActivity(0.2, 0.3, speech=False, confidence=0.0)
        )

        self.assertEqual([], ends)

        coordinator.on_vad_activity(
            VadActivity(0.3, 0.4, speech=False, confidence=0.0)
        )
        coordinator.on_vad_activity(
            VadActivity(0.4, 0.5, speech=False, confidence=0.0)
        )

        self.assertEqual(1, len(ends))
        self.assertEqual(0.3, ends[0].t)

    def test_turn_controller_serializes_activity_events(self) -> None:
        bus = EventBus()
        starts: list[SpeechStart] = []
        binds: list[SpeakerBind] = []
        bus.subscribe(SpeechStart, starts.append)
        bus.subscribe(SpeakerBind, binds.append)
        controller = TurnController(
            bus=bus,
            coordinator=Coordinator(bus, min_utterance_chunks=1),
        )

        controller.start()
        try:
            controller.submit_vad(VadActivity(0.0, 0.1, speech=True, confidence=1.0))
            controller.submit_speaker(
                SpeakerActivity(0.0, 0.1, primary_speaker=3, all_speakers=(3,))
            )
            deadline = time.time() + 2.0
            while len(binds) < 1 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            controller.stop()

        self.assertEqual(1, len(starts))
        self.assertIsNone(starts[0].speaker)
        self.assertEqual(1, len(binds))
        self.assertEqual(starts[0].utterance_id, binds[0].utterance_id)
        self.assertEqual(3, binds[0].speaker)

    def test_turn_controller_default_queue_is_unbounded(self) -> None:
        controller = TurnController(
            bus=EventBus(),
            coordinator=Coordinator(EventBus(), min_utterance_chunks=1),
        )

        self.assertEqual(0, controller._queue.maxsize)

    def test_final_enricher_prefers_utterance_owner_speaker(self) -> None:
        bus = EventBus()
        enricher = FinalEnricher(bus=bus, diarizer=FakeDiarizer(speaker=9))

        labeled = enricher._enrich(
            Final(text="hello", span=(0.0, 1.0), utterance_id=1, speaker=2)
        )

        self.assertIsInstance(labeled, LabeledFinal)
        self.assertEqual(2, labeled.speaker)

    def test_final_enricher_falls_back_to_diarizer_speaker(self) -> None:
        bus = EventBus()
        enricher = FinalEnricher(bus=bus, diarizer=FakeDiarizer(speaker=3))

        labeled = enricher._enrich(
            Final(text="hello", span=(0.0, 1.0), utterance_id=1)
        )

        self.assertEqual(3, labeled.speaker)

    def test_transcriber_rewinds_overfed_audio_on_speaker_change(self) -> None:
        bus = EventBus()
        finals: list[Final] = []
        bus.subscribe(Final, finals.append)
        ring = AudioRingBuffer()
        worker = TranscriberWorker(
            transcriber=FakeTranscriber(),
            bus=bus,
            sample_rate=16000,
        )
        worker.set_ring_buffer(ring)

        worker.start()
        try:
            worker.open_session(0.0, 1, speaker=0)
            frames = [
                make_frame(0, 0.0, 0.1),
                make_frame(1, 0.1, 0.2),
                make_frame(2, 0.2, 0.3),
                make_frame(3, 0.3, 0.4),
            ]
            for frame in frames:
                ring.append(frame)
                worker.feed(frame)
            time.sleep(0.1)

            worker.close_session(0.1, 1)
            worker.open_session(0.1, 2, speaker=1)
            time.sleep(0.1)
            worker.close_session(0.4, 2)

            deadline = time.time() + 2.0
            while len(finals) < 2 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            worker.stop()

        by_id = {event.utterance_id: event for event in finals}
        self.assertEqual((0.0, 0.1), by_id[1].span)
        self.assertEqual(0, by_id[1].speaker)
        self.assertEqual("samples:1600", by_id[1].text)
        self.assertEqual((0.1, 0.4), by_id[2].span)
        self.assertEqual(1, by_id[2].speaker)
        self.assertEqual("samples:4800", by_id[2].text)

    def test_transcriber_uses_one_shot_context_for_rewind_final(self) -> None:
        bus = EventBus()
        finals: list[Final] = []
        bus.subscribe(Final, finals.append)
        ring = AudioRingBuffer()
        transcriber = FakeOneShotTranscriber()
        worker = TranscriberWorker(
            transcriber=transcriber,
            bus=bus,
            sample_rate=16000,
        )
        worker.set_ring_buffer(ring)

        worker.start()
        try:
            worker.open_session(0.0, 1, speaker=0)
            for frame in [
                make_frame(0, 0.0, 0.1),
                make_frame(1, 0.1, 0.2),
            ]:
                ring.append(frame)
                worker.feed(frame)
            time.sleep(0.1)

            worker.close_session(0.1, 1)
            deadline = time.time() + 2.0
            while len(finals) < 1 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            worker.stop()

        self.assertEqual([1600], transcriber.one_shot_lengths)
        self.assertEqual(1, len(finals))
        self.assertEqual((0.0, 0.1), finals[0].span)
        self.assertEqual("oneshot:1600", finals[0].text)

    def test_transcriber_trims_pending_audio_to_open_time(self) -> None:
        bus = EventBus()
        finals: list[Final] = []
        bus.subscribe(Final, finals.append)
        worker = TranscriberWorker(
            transcriber=FakeTranscriber(),
            bus=bus,
            sample_rate=16000,
        )

        worker.start()
        try:
            worker.feed(make_frame(0, 0.0, 0.1))
            time.sleep(0.1)
            worker.open_session(0.05, 1, speaker=0)
            time.sleep(0.1)
            worker.close_session(0.1, 1)

            deadline = time.time() + 2.0
            while len(finals) < 1 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            worker.stop()

        self.assertEqual(1, len(finals))
        self.assertEqual((0.05, 0.1), finals[0].span)
        self.assertEqual(0, finals[0].speaker)
        self.assertEqual("samples:800", finals[0].text)

    def test_transcriber_discards_finalized_hallucination_with_clear_event(self) -> None:
        bus = EventBus()
        finals: list[Final] = []
        discarded: list[FinalDiscarded] = []
        bus.subscribe(Final, finals.append)
        bus.subscribe(FinalDiscarded, discarded.append)
        worker = TranscriberWorker(
            transcriber=FakeNoTextTranscriber(),
            bus=bus,
            sample_rate=16000,
        )

        worker.start()
        try:
            worker.open_session(0.0, 1, speaker=0)
            worker.feed(make_frame(0, 0.0, 0.1))
            time.sleep(0.1)
            worker.close_session(0.1, 1)

            deadline = time.time() + 2.0
            while len(discarded) < 1 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            worker.stop()

        self.assertEqual([], finals)
        self.assertEqual(1, len(discarded))
        self.assertEqual(1, discarded[0].utterance_id)
        self.assertEqual((0.0, 0.1), discarded[0].span)
        self.assertEqual("hallucination", discarded[0].reason)

    def test_transcriber_finalizing_timeout_forces_best_available_final(self) -> None:
        bus = EventBus()
        finals: list[Final] = []
        bus.subscribe(Final, finals.append)
        worker = TranscriberWorker(
            transcriber=FakeNeverDoneTranscriber(),
            bus=bus,
            sample_rate=16000,
            finalizing_timeout_s=0.05,
        )

        worker.start()
        try:
            worker.open_session(0.0, 1, speaker=0)
            worker.feed(make_frame(0, 0.0, 0.1))
            time.sleep(0.1)
            worker.close_session(0.1, 1)

            deadline = time.time() + 2.0
            while len(finals) < 1 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            worker.stop()

        self.assertEqual(1, len(finals))
        self.assertEqual(1, finals[0].utterance_id)
        self.assertEqual("stuck:1600", finals[0].text)

    def test_transcriber_queue_preserves_close_under_audio_pressure(self) -> None:
        worker = TranscriberWorker(
            transcriber=FakeTranscriber(),
            bus=EventBus(),
            sample_rate=16000,
            max_queue=2,
        )
        worker._running = True
        try:
            worker.feed(make_frame(0, 0.0, 0.1))
            worker.feed(make_frame(1, 0.1, 0.2))
            worker.close_session(0.2, 7)

            queued = []
            while not worker._queue.empty():
                queued.append(worker._queue.get_nowait())
        finally:
            worker._running = False

        self.assertTrue(any(type(item).__name__ == "_Close" for item in queued))
        self.assertFalse(
            any(isinstance(item, AudioFrame) and item.seq == 0 for item in queued)
        )

    def test_transcriber_audio_overflow_does_not_drop_existing_control(self) -> None:
        worker = TranscriberWorker(
            transcriber=FakeTranscriber(),
            bus=EventBus(),
            sample_rate=16000,
            max_queue=2,
        )
        worker._running = True
        try:
            worker.close_session(0.1, 3)
            worker.feed(make_frame(0, 0.0, 0.1))
            worker.feed(make_frame(1, 0.1, 0.2))

            queued = []
            while not worker._queue.empty():
                queued.append(worker._queue.get_nowait())
        finally:
            worker._running = False

        self.assertTrue(any(type(item).__name__ == "_Close" for item in queued))
        self.assertTrue(any(isinstance(item, AudioFrame) for item in queued))

    def test_silero_vad_emits_smoothed_activity(self) -> None:
        events: list[tuple[float, float, bool, float | None]] = []
        vad = SileroVad(
            model=FakeSileroModel([0.8, 0.2, 0.2]),
            start_chunks=1,
            end_chunks=2,
            on_activity=lambda *event: events.append(event),
        )
        vad.start()
        try:
            vad.feed(
                AudioFrame(
                    seq=0,
                    t_start=0.0,
                    t_end=512 * 3 / 16000,
                    samples=np.ones(512 * 3, dtype=np.float32),
                    sample_rate=16000,
                )
            )
            deadline = time.time() + 2.0
            while len(events) < 3 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            vad.stop()

        self.assertEqual([True, True, False], [event[2] for event in events])

    def test_smart_turn_endpoint_uses_ring_buffer_turn_audio(self) -> None:
        model = FakeSmartTurnModel([1])
        endpoint = SmartTurnEndpoint(
            sample_rate=16000,
            threshold=0.7,
            model=model,
        )
        ring = AudioRingBuffer()
        ring.append(make_frame(0, 0.0, 0.1))
        ring.append(make_frame(1, 0.1, 0.2))

        self.assertTrue(endpoint.should_end(ring, 0.05, 0.2))
        self.assertEqual([(2400, 16000, 0.7)], model.calls)


if __name__ == "__main__":
    unittest.main()
