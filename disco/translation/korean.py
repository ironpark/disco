"""Korean translation using mlx-lm translation and chat models."""

import threading
from collections import OrderedDict
from dataclasses import dataclass


DEFAULT_TRANSLATION_MODEL = "mlx-community/translategemma-4b-it-8bit"


@dataclass(frozen=True)
class TranslationContextItem:
    speaker: int | None
    text: str
    translation: str | None = None


class KoreanTranslator:
    """Translate realtime ASR text to Korean using an mlx-lm model."""

    def __init__(
        self,
        model_name: str = DEFAULT_TRANSLATION_MODEL,
        cache_size: int = 128,
    ):
        """Initialize the translator.

        Args:
            model_name: HuggingFace model name for translation
            cache_size: Maximum number of translation results to keep
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[
            tuple[str, str, str, tuple[tuple[str, str], ...]], str
        ] = OrderedDict()

    def load(self) -> None:
        """Load the translation model."""
        if self._model is None:
            from mlx_lm import load as load_lm

            print(f"Loading translation model: {self.model_name}")
            self._model, self._tokenizer = load_lm(self.model_name)
            if self._uses_translate_gemma() and hasattr(
                self._tokenizer, "add_eos_token"
            ):
                self._tokenizer.add_eos_token("<end_of_turn>")
            print("Translation model loaded!")

    @property
    def model(self):
        """Get the translation model, loading if necessary."""
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        *,
        context: tuple[TranslationContextItem, ...] = (),
        mode: str = "final",
    ) -> str:
        """Translate text to Korean.

        Args:
            text: Text to translate
            source_lang: Source language ISO 639-1 code
            context: Recent final utterances for disambiguation only
            mode: ``final`` or ``interim``; interim text may be incomplete

        Returns:
            Translated text in Korean
        """
        normalized = " ".join(text.split())
        if not normalized:
            return ""
        uses_translate_gemma = self._uses_translate_gemma()
        context_pairs = () if uses_translate_gemma else self._context_pairs(context)
        cache_key = (source_lang, mode, normalized, context_pairs)

        try:
            from mlx_lm import generate

            with self._lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
                    return cached

                messages = self._messages(
                    normalized,
                    source_lang=source_lang,
                    context_pairs=context_pairs,
                    mode=mode,
                )
                prompt = self._format_prompt(messages)
                response = self._generate(prompt, generate)
                translated = response.strip()
                if not uses_translate_gemma and self._needs_retry(
                    translated, source_lang=source_lang, source_text=normalized
                ):
                    retry_messages = self._messages(
                        normalized,
                        source_lang=source_lang,
                        context_pairs=context_pairs,
                        mode=mode,
                        retry=True,
                    )
                    retry_prompt = self._format_prompt(retry_messages)
                    retry_response = self._generate(retry_prompt, generate).strip()
                    if not self._needs_retry(
                        retry_response,
                        source_lang=source_lang,
                        source_text=normalized,
                    ):
                        translated = retry_response
                self._cache[cache_key] = translated
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return translated
        except Exception as e:
            return f"[Translation error: {e}]"

    def _context_pairs(
        self, context: tuple[TranslationContextItem, ...]
    ) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for item in context:
            text = " ".join(item.text.split())
            translation = " ".join((item.translation or "").split())
            if not text or not translation:
                continue
            pairs.append((text, translation))
        return tuple(pairs)

    def _messages(
        self,
        text: str,
        *,
        source_lang: str,
        context_pairs: tuple[tuple[str, str], ...],
        mode: str,
        retry: bool = False,
    ) -> list[dict]:
        if self._uses_translate_gemma():
            return self._translate_gemma_messages(text, source_lang=source_lang)

        messages: list[dict] = [
            {
                "role": "system",
                "content": self._system_prompt(mode=mode, retry=retry),
            }
        ]
        messages.extend(self._few_shot_messages(source_lang=source_lang))
        context_text = self._context_text(context_pairs)
        messages.append(
            {
                "role": "user",
                "content": self._user_prompt(
                    text,
                    source_lang=source_lang,
                    context_text=context_text,
                ),
            }
        )
        return messages

    def _translate_gemma_messages(self, text: str, *, source_lang: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": "ko",
                        "text": text,
                    }
                ],
            }
        ]

    def _format_prompt(self, messages: list[dict]) -> str:
        tokenizer = self.tokenizer
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )

        rendered: list[str] = []
        for message in messages:
            role = message["role"].upper()
            rendered.append(f"{role}:\n{message['content']}")
        rendered.append("ASSISTANT:\n")
        return "\n\n".join(rendered)

    def _system_prompt(self, *, mode: str, retry: bool = False) -> str:
        completeness = (
            "The current utterance may be an incomplete interim fragment."
            if mode == "interim"
            else "The current utterance is a finalized transcript segment."
        )
        lines = [
            "You are a translation engine, not a chat assistant.",
            "Task: translate only SOURCE_TEXT into Korean.",
            "Hard output contract: output exactly one Korean translation line and nothing else.",
            "The output is the Korean subtitle line that would replace SOURCE_TEXT.",
            "Always output natural Korean, even when SOURCE_TEXT is Japanese.",
            "Never output Japanese text, kana, romaji, source-language rewrites, explanations, labels, quotes, or markdown.",
            "Translate the spoken content directly instead of describing the utterance.",
            "The source text may contain ASR mistakes, missing words, repeated fragments, speaker changes, or incomplete sentences.",
            completeness,
            "Use recent context only to resolve ambiguity and obvious recognition mistakes.",
            "Translate only SOURCE_TEXT.",
            "Do not translate or summarize the context.",
            "Never mention the context, speaker, ASR, transcript, source text, or translation task in the output.",
            "If SOURCE_TEXT is fragmentary, translate the fragment as-is; do not complete or explain it.",
            "Keep uncertainty only when SOURCE_TEXT itself expresses uncertainty.",
            "Correct only obvious ASR mistakes when context strongly supports it.",
            "Do not add information that is not implied by the source or context.",
            "Output Korean only.",
        ]
        if retry:
            lines.append(
                "The previous attempt was invalid because it explained or commented. Return only the direct Korean translation of SOURCE_TEXT."
            )
        return "\n".join(lines)

    def _few_shot_messages(self, *, source_lang: str) -> list[dict]:
        if not source_lang.lower().startswith("ja"):
            return []
        return [
            {
                "role": "user",
                "content": self._user_prompt(
                    "誰に対して言うんだろう。",
                    source_lang=source_lang,
                    context_text="(none)",
                ),
            },
            {"role": "assistant", "content": "누구한테 말하는 걸까?"},
            {
                "role": "user",
                "content": self._user_prompt(
                    "誰に対して言うのか？",
                    source_lang=source_lang,
                    context_text="(none)",
                ),
            },
            {"role": "assistant", "content": "누구한테 말하는 건가?"},
            {
                "role": "user",
                "content": self._user_prompt(
                    "そういう判断になるのか。",
                    source_lang=source_lang,
                    context_text="1. Source: 書くかどうかっていう判断。\n   Korean: 쓸지 말지의 판단.",
                ),
            },
            {"role": "assistant", "content": "그런 판단이 되는 건가?"},
            {
                "role": "user",
                "content": self._user_prompt(
                    "それはちょっと違う方向かもしれない。",
                    source_lang=source_lang,
                    context_text="(none)",
                ),
            },
            {"role": "assistant", "content": "그건 조금 다른 방향일지도 몰라."},
        ]

    def _context_text(self, context_pairs: tuple[tuple[str, str], ...]) -> str:
        if not context_pairs:
            return "(none)"
        lines: list[str] = []
        for index, (source_text, translated_text) in enumerate(context_pairs, start=1):
            lines.append(f"{index}. Source: {source_text}")
            lines.append(f"   Korean: {translated_text}")
        return "\n".join(lines)

    def _user_prompt(
        self, text: str, *, source_lang: str, context_text: str
    ) -> str:
        return "\n".join(
            [
                f"Source language code: {source_lang}",
                "",
                "Recent context for reference only:",
                "<CONTEXT>",
                context_text,
                "</CONTEXT>",
                "",
                "SOURCE_TEXT:",
                "<SOURCE_TEXT>",
                text,
                "</SOURCE_TEXT>",
                "",
                "Korean translation only:",
            ]
        )

    def _generate(self, prompt: str, generate) -> str:
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=256,
            verbose=False,
        )

    def _uses_translate_gemma(self) -> bool:
        return "translategemma" in self.model_name.lower()

    def _needs_retry(
        self, text: str, *, source_lang: str, source_text: str
    ) -> bool:
        return self._looks_untranslated(
            text, source_lang=source_lang
        ) or self._looks_like_meta_output(text, source_text=source_text)

    def _looks_untranslated(self, text: str, *, source_lang: str) -> bool:
        if not source_lang.lower().startswith("ja"):
            return False
        has_kana = any(
            "\u3040" <= char <= "\u30ff" or "\uff66" <= char <= "\uff9f"
            for char in text
        )
        has_hangul = any("\uac00" <= char <= "\ud7a3" for char in text)
        return has_kana or not has_hangul

    def _looks_like_meta_output(self, text: str, *, source_text: str) -> bool:
        normalized = text.strip().lstrip("\"'`“”‘’").lower()
        meta_prefixes = (
            "이 말은",
            "이 문장은",
            "이 표현은",
            "이 발화는",
            "이는",
            "즉",
            "문맥상",
            "상황상",
            "화자가",
            "화자는",
            "원문은",
            "원문에서는",
            "출력은",
            "번역하면",
            "번역:",
            "한국어:",
            "한국어 번역",
            "현재 발화",
            "source_text",
        )
        if normalized.startswith(meta_prefixes):
            return True
        meta_fragments = (
            "라는 뜻",
            "라는 의미",
            "라고 말하는 상황",
            "상황을 설명",
            "문맥을 설명",
            "문맥상으로",
            "번역 결과",
            "원문에",
            "source text",
        )
        if any(fragment in normalized for fragment in meta_fragments):
            return True
        source_has_direction_or_error = any(
            token in source_text.lower()
            for token in ("方向", "間違", "wrong", "direction", "잘못", "방향")
        )
        return "잘못된 방향" in normalized and not source_has_direction_or_error
