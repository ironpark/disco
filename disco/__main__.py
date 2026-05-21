"""Main entry point for disco real-time ASR."""

import sounddevice as sd

from disco.asr import make_transcriber
from disco.audio.capture import get_device_info
from disco.audio.source import AudioSource
from disco.cli import args_to_config, parse_args, print_devices
from disco.diar import Diarizer
from disco.output import ConsoleOutput
from disco.runtime.events import EnrichedFinal, EventBus, Interim
from disco.runtime.runtime import Runtime
from disco.translation import KoreanTranslator


def main() -> None:
    args = parse_args()

    if args.list_devices:
        print_devices()
        return

    config = args_to_config(args)

    if config.device is not None:
        device_info = get_device_info(config.device)
        print(f"Using device: {device_info['name']}")

    transcriber = make_transcriber(
        config.asr_backend,
        model_name=config.model_name,
        sample_rate=config.sample_rate,
        language=config.language,
    )
    diarizer = Diarizer(sample_rate=config.sample_rate)
    translator = KoreanTranslator() if config.translate_to_korean else None
    if translator is not None:
        translator.load()

    output = ConsoleOutput(show_translation=config.translate_to_korean)

    bus = EventBus()
    bus.subscribe(Interim, lambda e: output.show_interim(e.text, speaker=e.speaker))
    bus.subscribe(
        EnrichedFinal,
        lambda e: output.show_final(e.text, e.translation, speaker=e.speaker),
    )

    runtime = Runtime(
        bus=bus,
        transcriber=transcriber,
        diarizer=diarizer,
        translator=translator,
        language=config.language,
        sample_rate=config.sample_rate,
        silence_duration=config.silence_duration,
        min_utterance_duration=config.chunk_duration,
    )

    output.show_start(config.language, translate=config.translate_to_korean)

    source = AudioSource(
        sample_rate=config.sample_rate,
        channels=config.channels,
        device=config.device,
    )
    runtime.start(source)

    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        output.show_stop()
    finally:
        runtime.stop()

    output.show_done()


if __name__ == "__main__":
    main()
