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
    def translate(self, text: str, source_lang: str) -> str:
        time.sleep(0.05)
        return f"ko:{text}"


class TranslationServiceTest(unittest.TestCase):
    def test_in_flight_interim_is_dropped_after_final(self) -> None:
        bus = EventBus()
        interims: list[EnrichedInterim] = []
        finals: list[EnrichedFinal] = []
        bus.subscribe(EnrichedInterim, interims.append)
        bus.subscribe(EnrichedFinal, finals.append)
        service = TranslationService(
            bus=bus,
            translator=SlowTranslator(),
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


if __name__ == "__main__":
    unittest.main()
