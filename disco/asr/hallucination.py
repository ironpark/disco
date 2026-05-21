"""Shared ASR text filters."""


HALLUCINATIONS = frozenset({
    "okay",
    "okay.",
    "ok",
    "ok.",
    "thank you.",
    "thanks.",
    "bye.",
    "yes.",
    "no.",
    "...",
    ".",
    "",
})


def is_hallucination(text: str) -> bool:
    return text.strip().lower() in HALLUCINATIONS
