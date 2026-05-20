"""Main entry point for disco real-time ASR."""

import threading
import time

import sounddevice as sd

from disco.asr import Transcriber
from disco.asr.transcriber import StreamingTranscription, is_hallucination
from disco.audio.capture import AudioCapture, get_device_info
from disco.cli import args_to_config, parse_args, print_devices
from disco.config import ASRConfig, LANG_CODE_MAP, TranscriptEntry
from disco.diar import Diarizer
from disco.output import ConsoleOutput
from disco.translation import KoreanTranslator
from disco.vad import SileroVAD


class RealtimeASR:
    """Real-time Automatic Speech Recognition driven by a Voxtral streaming session."""

    SPEAKER_CHANGE_HOLD = 0.4

    def __init__(
        self,
        config: ASRConfig,
        vad: SileroVAD,
        transcriber: Transcriber,
        output: ConsoleOutput,
        diarizer: Diarizer,
        translator: KoreanTranslator | None = None,
    ):
        self.config = config
        self.vad = vad
        self.transcriber = transcriber
        self.output = output
        self.diarizer = diarizer
        self.translator = translator

        self.is_recording = False
        self.transcript_history: list[TranscriptEntry] = []

    def _finalize(self, session: StreamingTranscription, diar_start: float) -> None:
        session.close()
        session.drain()
        text = session.text.strip()
        if not text or is_hallucination(text):
            return

        speaker = self.diarizer.dominant_speaker_in(
            diar_start, self.diarizer.elapsed_seconds()
        )

        self.transcript_history.append(
            TranscriptEntry(timestamp=time.time(), text=text)
        )

        translation = None
        if self.translator:
            source_lang = LANG_CODE_MAP.get(self.config.language.lower(), "en")
            translation = self.translator.translate(text, source_lang)

        self.output.show_final(text, translation, speaker=speaker)

    def _process_audio(self, audio_capture: AudioCapture) -> None:
        required_silence_chunks = max(1, int(self.config.silence_duration / 0.1))
        min_utterance_samples = int(self.config.chunk_duration * self.config.sample_rate)

        session: StreamingTranscription | None = None
        samples_fed = 0
        silence_chunks = 0
        last_interim_text = ""

        diar_start: float = 0.0
        bound_speaker: int | None = None
        speaker_change_start: float | None = None

        self.diarizer.start()

        try:
            while self.is_recording:
                chunk = audio_capture.get_chunk(timeout=0.05)

                if chunk is not None:
                    samples = chunk.reshape(-1)
                    has_speech = self.vad.is_speech_chunk(chunk)
                    self.diarizer.feed(samples)

                    if has_speech:
                        if session is None:
                            session = self.transcriber.start_session()
                            diar_start = self.diarizer.elapsed_seconds()
                            samples_fed = 0
                            last_interim_text = ""
                            bound_speaker = None
                            speaker_change_start = None
                        silence_chunks = 0
                    elif session is not None:
                        silence_chunks += 1

                    if session is not None:
                        session.feed(samples)
                        samples_fed += len(samples)

                if session is None:
                    continue

                cur_t = self.diarizer.elapsed_seconds()
                if bound_speaker is None:
                    bound_speaker = self.diarizer.dominant_speaker_in(diar_start, cur_t)
                else:
                    latest = self.diarizer.speaker_at(cur_t - 0.2)
                    if latest is not None and latest != bound_speaker:
                        if speaker_change_start is None:
                            speaker_change_start = cur_t
                        elif cur_t - speaker_change_start >= self.SPEAKER_CHANGE_HOLD:
                            self._finalize(session, diar_start)
                            session = None
                            samples_fed = 0
                            silence_chunks = 0
                            last_interim_text = ""
                            bound_speaker = None
                            speaker_change_start = None
                            continue
                    else:
                        speaker_change_start = None

                if session.step() and session.text != last_interim_text:
                    last_interim_text = session.text
                    self.output.show_interim(last_interim_text, speaker=bound_speaker)

                if (
                    silence_chunks >= required_silence_chunks
                    and samples_fed >= min_utterance_samples
                ):
                    self._finalize(session, diar_start)
                    session = None
                    samples_fed = 0
                    silence_chunks = 0
                    last_interim_text = ""
                    bound_speaker = None
                    speaker_change_start = None

            if session is not None:
                self._finalize(session, diar_start)
        finally:
            self.diarizer.stop()

    def start(self) -> None:
        self.output.show_start(
            self.config.language,
            translate=self.config.translate_to_korean,
        )

        self.is_recording = True
        self.transcript_history = []

        audio_capture = AudioCapture(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            device=self.config.device,
        )

        process_thread = threading.Thread(
            target=self._process_audio,
            args=(audio_capture,),
        )

        try:
            with audio_capture:
                process_thread.start()
                while self.is_recording:
                    sd.sleep(100)
        except KeyboardInterrupt:
            self.output.show_stop()
        finally:
            self.is_recording = False
            process_thread.join()

        self.output.show_done()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.list_devices:
        print_devices()
        return

    config = args_to_config(args)

    if config.device is not None:
        device_info = get_device_info(config.device)
        print(f"Using device: {device_info['name']}")

    vad = SileroVAD(sample_rate=config.sample_rate)
    vad.load()

    transcriber = Transcriber(
        model_name=config.model_name,
        sample_rate=config.sample_rate,
    )
    transcriber.load()

    diarizer = Diarizer(sample_rate=config.sample_rate)
    diarizer.load()

    translator = None
    if config.translate_to_korean:
        translator = KoreanTranslator()
        translator.load()

    output = ConsoleOutput(show_translation=config.translate_to_korean)

    asr = RealtimeASR(
        config=config,
        vad=vad,
        transcriber=transcriber,
        output=output,
        diarizer=diarizer,
        translator=translator,
    )
    asr.start()


if __name__ == "__main__":
    main()
