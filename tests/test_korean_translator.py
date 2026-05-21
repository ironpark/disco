import unittest

from disco.translation.korean import KoreanTranslator


class KoreanTranslatorPromptTest(unittest.TestCase):
    def test_translate_gemma_is_default_model(self) -> None:
        translator = KoreanTranslator()

        self.assertEqual(
            "mlx-community/translategemma-4b-it-8bit",
            translator.model_name,
        )
        self.assertTrue(translator._uses_translate_gemma())

    def test_translate_gemma_uses_structured_translation_message(self) -> None:
        translator = KoreanTranslator()

        messages = translator._messages(
            "hello",
            source_lang="en",
            context_pairs=(),
            mode="final",
        )

        self.assertEqual(1, len(messages))
        content = messages[0]["content"]
        self.assertEqual("user", messages[0]["role"])
        self.assertEqual("text", content[0]["type"])
        self.assertEqual("en", content[0]["source_lang_code"])
        self.assertEqual("ko", content[0]["target_lang_code"])
        self.assertEqual("hello", content[0]["text"])

    def test_translate_gemma_prompt_does_not_embed_chat_retry_context(self) -> None:
        translator = KoreanTranslator()

        messages = translator._messages(
            "hello",
            source_lang="en",
            context_pairs=(("source", "translation"),),
            mode="final",
            retry=True,
        )

        self.assertEqual(1, len(messages))
        self.assertEqual(
            [
                {
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "ko",
                    "text": "hello",
                }
            ],
            messages[0]["content"],
        )

    def test_chat_model_uses_subtitle_prompt(self) -> None:
        translator = KoreanTranslator(model_name="mlx-community/Qwen3.5-0.8B-MLX-8bit")

        messages = translator._messages(
            "誰に対して言うんだろう。",
            source_lang="ja",
            context_pairs=(),
            mode="final",
        )

        self.assertFalse(translator._uses_translate_gemma())
        self.assertEqual("system", messages[0]["role"])
        self.assertIn("Korean subtitle line", messages[0]["content"])

    def test_meta_explanation_outputs_trigger_retry(self) -> None:
        translator = KoreanTranslator()

        self.assertTrue(
            translator._looks_like_meta_output(
                "이 말은 누군가에게 말하는 상황을 설명합니다.",
                source_text="誰に対して言うんだろう。",
            )
        )
        self.assertTrue(
            translator._looks_like_meta_output(
                "문맥상 화자가 판단에 대해 말하는 상황입니다.",
                source_text="そういう判断になるのか。",
            )
        )
        self.assertTrue(
            translator._looks_like_meta_output(
                "이는 조금 다른 방향일 수 있다는 뜻입니다.",
                source_text="それはちょっと違う方向かもしれない。",
            )
        )

    def test_direct_translation_does_not_trigger_meta_retry(self) -> None:
        translator = KoreanTranslator()

        self.assertFalse(
            translator._looks_like_meta_output(
                "그건 조금 다른 방향일지도 몰라.",
                source_text="それはちょっと違う方向かもしれない。",
            )
        )
        self.assertFalse(
            translator._looks_like_meta_output(
                "그런 판단이 되는 건가?",
                source_text="そういう判断になるのか。",
            )
        )

    def test_system_prompt_uses_positive_output_contract(self) -> None:
        prompt = KoreanTranslator()._system_prompt(mode="final")

        self.assertIn("one Korean translation line", prompt)
        self.assertIn("Korean subtitle line", prompt)
        self.assertIn("translate the fragment as-is", prompt)
        self.assertNotIn("이 말은", prompt)
        self.assertNotIn("문맥상", prompt)
        self.assertNotIn("라는 뜻", prompt)


if __name__ == "__main__":
    unittest.main()
