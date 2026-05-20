"""Web server entry point."""

import argparse

from disco.audio.capture import list_devices as get_audio_devices


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Disco Web UI - Real-time ASR")
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
        "--min-utterance-duration",
        type=float,
        default=0.5,
        help="Minimum audio fed into a session before a VAD-triggered finalize (seconds)",
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
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    return parser.parse_args()


def print_devices() -> None:
    """Print available audio input devices."""
    print("\nAvailable input devices:")
    print("-" * 50)
    devices = get_audio_devices()
    for dev in devices:
        default = " (default)" if dev["is_default"] else ""
        print(f"  {dev['id']}: {dev['name']}{default}")
    print()


def main():
    """Run the web server."""
    args = parse_args()

    if args.list_devices:
        print_devices()
        return

    import uvicorn

    from disco.audio.capture import get_device_info
    from disco.web.app import app, set_config

    # Set configuration from CLI args
    set_config(
        device=args.device,
        language=args.language,
        translate_korean=args.translate_korean,
        silence_duration=args.silence_duration,
        min_utterance_duration=args.min_utterance_duration,
    )

    if args.device is not None:
        device_info = get_device_info(args.device)
        print(f"Using device: {device_info['name']}")

    print(f"\n  Disco Web UI")
    print(f"  Language: {args.language}")
    if args.translate_korean:
        print(f"  Translation: Korean")
    print(f"\n  Open http://{args.host}:{args.port} in your browser\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
