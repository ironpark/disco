"""Korean translation using translategemma."""

import threading


class KoreanTranslator:
    """Translate text to Korean using translategemma."""

    def __init__(
        self,
        model_name: str = "mlx-community/translategemma-4b-it-8bit",
    ):
        """Initialize the translator.

        Args:
            model_name: HuggingFace model name for translation
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

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
        if not text:
            return ""

        try:
            from mlx_lm import generate

            with self._lock:
                messages = [
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
            return response.strip()
        except Exception as e:
            return f"[Translation error: {e}]"
