import time
import unittest

from disco.runtime.events import (
    EnrichedFinal,
    EnrichedInterim,
    EventBus,
    Interim,
    LabeledFinal,
    TurnRef,
)
from disco.runtime.translation_service import TranslationService


class SlowTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text: str, source_lang: str, **kwargs) -> str:
        self.calls.append((text, source_lang, kwargs))
        time.sleep(0.05)
        return f"ko:{text}"


class TranslationServiceTest(unittest.TestCase):
    def test_in_flight_interim_is_dropped_after_final(self) -> None:
        bus = EventBus()
        interims: list[EnrichedInterim] = []
        finals: list[EnrichedFinal] = []
        bus.subscribe(EnrichedInterim, interims.append)
        bus.subscribe(EnrichedFinal, finals.append)
        translator = SlowTranslator()
        service = TranslationService(
            bus=bus,
            translator=translator,
            interim_interval_s=0.0,
            interim_min_chars=1,
        )

        service.start()
        try:
            service.submit_interim(
                Interim(
                    text="hello there",
                    span=(0.0, 1.0),
                    utterance_id=1,
                    speaker=0,
                )
            )
            time.sleep(0.02)
            service.submit_final(
                LabeledFinal(
                    text="hello there",
                    ref=TurnRef.single(
                        utterance_id=1,
                        span=(0.0, 1.0),
                        speaker=0,
                    ),
                )
            )
            time.sleep(0.2)
        finally:
            service.stop()

        self.assertEqual([], interims)
        self.assertEqual(1, len(finals))
        self.assertEqual("ko:hello there", finals[0].translation)

    def test_final_context_is_passed_to_following_translations(self) -> None:
        bus = EventBus()
        finals: list[EnrichedFinal] = []
        bus.subscribe(EnrichedFinal, finals.append)
        translator = SlowTranslator()
        service = TranslationService(
            bus=bus,
            translator=translator,
            interim_interval_s=0.0,
            interim_min_chars=1,
        )

        service.start()
        try:
            service.submit_final(
                LabeledFinal(
                    text="The build failed.",
                    ref=TurnRef.single(
                        utterance_id=1,
                        span=(0.0, 1.0),
                        speaker=0,
                    ),
                )
            )
            service.submit_final(
                LabeledFinal(
                    text="It needs a clean retry.",
                    ref=TurnRef.single(
                        utterance_id=2,
                        span=(1.0, 2.0),
                        speaker=1,
                    ),
                )
            )
            deadline = time.time() + 2.0
            while len(finals) < 2 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            service.stop()

        second_context = translator.calls[1][2]["context"]
        self.assertEqual(1, len(second_context))
        self.assertEqual(0, second_context[0].speaker)
        self.assertEqual("The build failed.", second_context[0].text)
        self.assertEqual("ko:The build failed.", second_context[0].translation)
        self.assertEqual("final", translator.calls[1][2]["mode"])

    def test_interim_translation_uses_short_recent_context(self) -> None:
        bus = EventBus()
        interims: list[EnrichedInterim] = []
        finals: list[EnrichedFinal] = []
        bus.subscribe(EnrichedInterim, interims.append)
        bus.subscribe(EnrichedFinal, finals.append)
        translator = SlowTranslator()
        service = TranslationService(
            bus=bus,
            translator=translator,
            interim_interval_s=0.0,
            interim_min_chars=1,
            interim_context_size=1,
        )

        service.start()
        try:
            service.submit_final(
                LabeledFinal(
                    text="We changed the API.",
                    ref=TurnRef.single(
                        utterance_id=1,
                        span=(0.0, 1.0),
                        speaker=0,
                    ),
                )
            )
            deadline = time.time() + 2.0
            while len(finals) < 1 and time.time() < deadline:
                time.sleep(0.02)
            service.submit_interim(
                Interim(
                    text="that means",
                    span=(1.0, 1.5),
                    utterance_id=2,
                    speaker=0,
                )
            )
            deadline = time.time() + 2.0
            while len(interims) < 1 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            service.stop()

        interim_call = translator.calls[-1]
        self.assertEqual("interim", interim_call[2]["mode"])
        self.assertEqual(1, len(interim_call[2]["context"]))
        self.assertEqual("We changed the API.", interim_call[2]["context"][0].text)
        self.assertEqual(
            "ko:We changed the API.",
            interim_call[2]["context"][0].translation,
        )


if __name__ == "__main__":
    unittest.main()
