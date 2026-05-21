import unittest

import numpy as np

from disco.audio.frame import AudioFrame
from disco.audio.source import AudioSource


class CapturingConsumer:
    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []

    def feed(self, frame: AudioFrame) -> None:
        self.frames.append(frame)


class AudioSourceTest(unittest.TestCase):
    def test_multichannel_callback_downmixes_without_stretching_time(self) -> None:
        source = AudioSource(sample_rate=16000, channels=2)
        consumer = CapturingConsumer()
        source.subscribe(consumer)

        left = np.ones(1600, dtype=np.float32)
        right = np.zeros(1600, dtype=np.float32)
        source._callback(np.column_stack((left, right)), 1600, None, 0)

        self.assertEqual(1, len(consumer.frames))
        frame = consumer.frames[0]
        self.assertEqual(1600, len(frame.samples))
        self.assertAlmostEqual(0.0, frame.t_start)
        self.assertAlmostEqual(0.1, frame.t_end)
        np.testing.assert_allclose(0.5, frame.samples)


if __name__ == "__main__":
    unittest.main()
