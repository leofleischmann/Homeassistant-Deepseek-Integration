"""Tests for models: which endpoint this is, and what a model id still gets us.

Pure standard library, so they run without Home Assistant installed::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from integration_modules import load

const = load("const")
models = load("models")

is_official = models.is_official_deepseek_api_base_url
migrate = models.migrate_legacy_chat_model


class TestNormalizeModelId(unittest.TestCase):
    def test_missing_id_is_empty(self) -> None:
        self.assertEqual(models.normalize_model_id(None), "")
        self.assertEqual(models.normalize_model_id(""), "")

    def test_case_and_padding_are_removed(self) -> None:
        self.assertEqual(
            models.normalize_model_id("  DeepSeek-V4-Flash \n"), "deepseek-v4-flash"
        )


class TestIsOfficialBaseUrl(unittest.TestCase):
    """Vision, structured output and id migration all key off this."""

    def test_unset_base_url_means_the_official_api(self) -> None:
        # An entry saved before the base URL was configurable has no value.
        self.assertTrue(is_official(None))
        self.assertTrue(is_official(const.DEEPSEEK_API_BASE_URL))

    def test_suffixes_do_not_change_the_verdict(self) -> None:
        for value in (
            "https://api.deepseek.com",
            "https://api.deepseek.com/",
            "https://api.deepseek.com/v1",
            "https://api.deepseek.com/v1/",
            "https://api.deepseek.com/v1//",
            "  HTTPS://API.DeepSeek.com/V1  ",
            "http://api.deepseek.com",
        ):
            with self.subTest(base_url=value):
                self.assertTrue(is_official(value))

    def test_other_hosts_are_gateways(self) -> None:
        for value in (
            "https://openrouter.ai/api/v1",
            "http://localhost:11434/v1",
            "https://api.deepseek.com.example.net/v1",
            "https://proxy.example.com/api.deepseek.com",
            "https://api.deepseek.com/v2",
        ):
            with self.subTest(base_url=value):
                self.assertFalse(is_official(value))


class TestLegacyModelMigration(unittest.TestCase):
    def test_retired_ids_move_to_the_recommended_model(self) -> None:
        for legacy in const.LEGACY_CHAT_MODELS:
            with self.subTest(model=legacy):
                self.assertEqual(
                    migrate(legacy, base_url=None), const.RECOMMENDED_CHAT_MODEL
                )
                self.assertTrue(models.is_retired_chat_model(legacy, base_url=None))

    def test_current_ids_are_left_alone(self) -> None:
        for model, _label in const.CHAT_MODEL_OPTIONS:
            with self.subTest(model=model):
                self.assertIsNone(migrate(model, base_url=None))
                self.assertFalse(models.is_retired_chat_model(model, base_url=None))

    def test_a_gateway_may_still_route_a_retired_id(self) -> None:
        self.assertIsNone(
            migrate("deepseek-chat", base_url="https://openrouter.ai/api/v1")
        )

    def test_padding_and_case_still_match(self) -> None:
        self.assertEqual(
            migrate("  DeepSeek-Chat  ", base_url=None), const.RECOMMENDED_CHAT_MODEL
        )

    def test_no_model_is_not_a_retired_model(self) -> None:
        self.assertIsNone(migrate(None, base_url=None))
        self.assertIsNone(migrate("", base_url=None))


class TestThinkingApiApplies(unittest.TestCase):
    """extra_body.thinking is DeepSeek-only; a proxy must not be sent it."""

    def test_deepseek_ids_and_the_unset_default(self) -> None:
        for model in ("", "deepseek-v4-flash", "  DeepSeek-V4-Pro  "):
            with self.subTest(model=model):
                self.assertTrue(models.model_uses_deepseek_thinking_api(model))

    def test_foreign_ids_get_no_deepseek_fields(self) -> None:
        for model in ("gpt-4o", "llama3.1", "claude-opus-4"):
            with self.subTest(model=model):
                self.assertFalse(models.model_uses_deepseek_thinking_api(model))


if __name__ == "__main__":
    unittest.main()
