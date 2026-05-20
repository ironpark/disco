"""Command-line interface for disco."""

import argparse

from disco.audio.capture import list_devices as get_audio_devices
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
        language=args.language,
        translate_to_korean=args.translate_korean,
    )


def print_devices() -> None:
    """Print available audio input devices."""
    print("\nAvailable input devices:")
    print("-" * 50)
    devices = get_audio_devices()
    for dev in devices:
        default = " (default)" if dev["is_default"] else ""
        print(f"  {dev['id']}: {dev['name']}{default}")
    print()
