"""Command-line interface for disco."""

import argparse

from disco.config import ASRConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Real-time ASR using mlx-audio")
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=None,
        help="Input device ID (use --list-devices to see available devices)",
    )
    parser.add_argument(
        "-l",
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.5,
        help="Minimum audio fed into a session before a VAD-triggered finalize (seconds)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.01,
        help="RMS threshold for silence detection",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.5,
        help="Silence duration to trigger transcription (seconds)",
    )
    parser.add_argument(
        "--max-utterance-duration",
        type=float,
        default=10.0,
        help="Force a turn split after this many seconds of continuous speech",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language for transcription (default: English)",
    )
    parser.add_argument(
        "-k",
        "--translate-korean",
        action="store_true",
        help="Translate transcriptions to Korean",
    )
    parser.add_argument(
        "--asr-backend",
        type=str,
        default="voxtral",
        choices=["voxtral", "qwen3-asr", "granite-speech"],
        help=(
            "ASR backend (default: voxtral — streaming; "
            "qwen3-asr/granite-speech are blob-based)"
        ),
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default=None,
        help="Override the backend's default model checkpoint",
    )
    parser.add_argument(
        "--translation-model",
        type=str,
        default=None,
        help="Override the Korean translation model checkpoint",
    )
    parser.add_argument(
        "--smart-turn",
        action="store_true",
        help="Use Smart Turn to confirm VAD silence endpoints",
    )
    parser.add_argument(
        "--smart-turn-model",
        type=str,
        default="mlx-community/smart-turn-v3",
        help="Smart Turn model checkpoint",
    )
    parser.add_argument(
        "--smart-turn-threshold",
        type=float,
        default=0.5,
        help="Smart Turn endpoint threshold",
    )

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> ASRConfig:
    """Convert parsed arguments to ASRConfig.

    Args:
        args: Parsed command-line arguments

    Returns:
        ASRConfig instance
    """
    return ASRConfig(
        device=args.device,
        chunk_duration=args.chunk_duration,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        max_utterance_duration=args.max_utterance_duration,
        language=args.language,
        translate_to_korean=args.translate_korean,
        asr_backend=args.asr_backend,
        model_name=args.asr_model,
        translation_model=args.translation_model,
        smart_turn=args.smart_turn,
        smart_turn_model=args.smart_turn_model,
        smart_turn_threshold=args.smart_turn_threshold,
    )


def print_devices() -> None:
    """Print available audio input devices."""
    from disco.audio.capture import list_devices as get_audio_devices

    print("\nAvailable input devices:")
    print("-" * 50)
    devices = get_audio_devices()
    for dev in devices:
        default = " (default)" if dev["is_default"] else ""
        print(f"  {dev['id']}: {dev['name']}{default}")
    print()
