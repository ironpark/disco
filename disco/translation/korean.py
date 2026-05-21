"""Korean translation using translategemma."""

import threading
from collections import OrderedDict


class KoreanTranslator:
    """Translate text to Korean using translategemma."""

    def __init__(
        self,
        model_name: str = "mlx-community/translategemma-4b-it-8bit",
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
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()

    def load(self) -> None:
        """Load the translation model."""
        if self._model is None:
            from mlx_lm import load as load_lm

            print(f"Loading translation model: {self.model_name}")
            self._model, self._tokenizer = load_lm(self.model_name)
            # Add <end_of_turn> as EOS token for proper stopping
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

    def translate(self, text: str, source_lang: str = "en") -> str:
        """Translate text to Korean.

        Args:
            text: Text to translate
            source_lang: Source language ISO 639-1 code

        Returns:
            Translated text in Korean
        """
        normalized = " ".join(text.split())
        if not normalized:
            return ""
        cache_key = (source_lang, normalized)

        try:
            from mlx_lm import generate

            with self._lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
                    return cached

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "source_lang_code": source_lang,
                                "target_lang_code": "ko",
                                "text": normalized,
                            }
                        ],
                    }
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=256,
                    verbose=False,
                )
                translated = response.strip()
                self._cache[cache_key] = translated
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return translated
        except Exception as e:
            return f"[Translation error: {e}]"
