"""Console output formatting."""


class ConsoleOutput:
    """Handle terminal output for ASR results."""

    def __init__(self, show_translation: bool = False):
        """Initialize console output.

        Args:
            show_translation: Whether to display translations
        """
        self.show_translation = show_translation

    def clear_line(self) -> None:
        """Clear current line in terminal."""
        print("\r\033[K", end="", flush=True)

    def _speaker_prefix(self, speaker: int | None) -> str:
        return f"[S{speaker}] " if speaker is not None else ""

    def show_interim(self, text: str, speaker: int | None = None) -> None:
        """Show interim transcription result."""
        self.clear_line()
        print(f"  {self._speaker_prefix(speaker)}{text}", end="", flush=True)

    def show_final(
        self,
        text: str,
        translation: str | None = None,
        speaker: int | None = None,
    ) -> None:
        """Show final transcription result."""
        self.clear_line()
        prefix = self._speaker_prefix(speaker)
        if self.show_translation and translation:
            print(f"  {prefix}{text}\n   {translation}")
        else:
            print(f"  {prefix}{text}")

    def show_start(self, language: str, translate: bool = False) -> None:
        """Show startup message.

        Args:
            language: Language being transcribed
            translate: Whether translation is enabled
        """
        translate_info = " + Korean translation" if translate else ""
        print(f"\n Starting real-time ASR (Language: {language}{translate_info})...")
        print("Speak into your microphone. Press Ctrl+C to stop.\n")
        print("-" * 50)

    def show_stop(self) -> None:
        """Show stop message."""
        print("\n\n  Stopping...")

    def show_done(self) -> None:
        """Show completion message."""
        print("Done!")
