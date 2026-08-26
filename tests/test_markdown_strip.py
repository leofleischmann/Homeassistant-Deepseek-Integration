"""Tests for markdown_strip.

Pure standard library, so they run without Home Assistant installed::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import unittest

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "custom_components"
    / "deepseek_conversation"
    / "markdown_strip.py"
)
_SPEC = importlib.util.spec_from_file_location("markdown_strip", _MODULE_PATH)
assert _SPEC and _SPEC.loader
markdown_strip = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(markdown_strip)

strip_markdown = markdown_strip.strip_markdown
StreamingMarkdownStripper = markdown_strip.StreamingMarkdownStripper


class TestStripMarkdown(unittest.TestCase):
    """The one-pass form used for the final speech string."""

    def assert_strips(self, cases: dict[str, str]) -> None:
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(strip_markdown(text), expected)

    def test_plain_emphasis(self) -> None:
        self.assert_strips(
            {
                "This is **bold** text": "This is bold text",
                "This is *italic* text": "This is italic text",
                "This is __bold__ text": "This is bold text",
                "This is _italic_ text": "This is italic text",
                "This is ~~struck~~ text": "This is struck text",
                "***strong emphasis***": "strong emphasis",
            }
        )

    def test_cjk_emphasis(self) -> None:
        """``\\w`` matches CJK, which used to leave the markers behind."""
        self.assert_strips(
            {
                "今天天气**很好**啊": "今天天气很好啊",
                "建议你**多喝热水**注意休息": "建议你多喝热水注意休息",
                "这是*斜体*文字": "这是斜体文字",
                "这是~~删除~~文字": "这是删除文字",
                "設定は**オン**です": "設定はオンです",
                "온도는 **23도**입니다": "온도는 23도입니다",
            }
        )

    def test_emphasis_between_scripts(self) -> None:
        """A marker between CJK and ASCII is still a marker."""
        self.assert_strips(
            {
                "mixed **粗体**and english": "mixed 粗体and english",
                "这是**bold**text": "这是boldtext",
                "请看**Home Assistant**docs页面": "请看Home Assistantdocs页面",
                "温度**23**degrees现在": "温度23degrees现在",
            }
        )

    def test_underscores_inside_words_survive(self) -> None:
        """An underscore inside a word is not emphasis, in any script."""
        self.assert_strips(
            {
                "light.kitchen_lamp is on": "light.kitchen_lamp is on",
                "snake_case_name stays": "snake_case_name stays",
                "call the_function_ now": "call the_function_ now",
                "sensor.wohnzimmer_temperatur": "sensor.wohnzimmer_temperatur",
                "Café_test_ hier": "Café_test_ hier",
                "тест_имя_ конец": "тест_имя_ конец",
                "你_好_啊": "你_好_啊",
            }
        )

    def test_arithmetic_is_not_emphasis(self) -> None:
        """A run with a digit on both sides is an exponent or a product."""
        self.assert_strips(
            {
                "5**2 is not bold**": "5**2 is not bold**",
                "2**3 = 8 und 4**2 = 16": "2**3 = 8 und 4**2 = 16",
                "3*4 ist 12": "3*4 ist 12",
                "3*4 und *wichtig* dazu": "3*4 und wichtig dazu",
                "**23**度": "23度",
            }
        )

    def test_marker_needs_a_non_space_neighbour(self) -> None:
        self.assert_strips(
            {
                "a ** b ** c": "a ** b ** c",
                "2 * 3 * 4": "2 * 3 * 4",
            }
        )

    def test_block_and_link_rules(self) -> None:
        self.assert_strips(
            {
                "# Heading\n- item one\n> quote": "Heading\nitem one\nquote",
                "[link](http://x.y) and ![img](http://a.b)": "link and img",
                "`code` stays as text": "code stays as text",
                "```python\nprint(1)\n```": "print(1)",
                "a -> b": "a  b",
                "a → b": "a  b",
            }
        )

    def test_empty_input(self) -> None:
        self.assertEqual(strip_markdown(""), "")


class TestStreamingInvariant(unittest.TestCase):
    """``"".join(feed(c) for c in chunks) + flush() == strip_markdown(text)``."""

    #: Real replies plus the shapes that have broken the invariant before:
    #: a backtick that exposes a heading marker, an arrow that leaves trailing
    #: whitespace, and an inline rule that creates a list marker ("*-*" -> "-").
    TEXTS = (
        "今天**很好**啊",
        "你好**世界**",
        "**世界**你好",
        "温度**23**度，湿度**45%**",
        "a **b** c",
        "**A** b_c_ d",
        "x_y_ z",
        "# H\n- i\n> q",
        "a`b`c\n```py\nz\n```",
        "[l](u) x",
        "Café_t_ x",
        "one *two* 三",
        "#` >a",
        "((a[-\n`\n ",
        "x ->  ",
        "*-*\n~*",
        "`\n# a",
        "5**2 x",
    )

    def every_split(self, text: str):
        for cuts in itertools.chain.from_iterable(
            itertools.combinations(range(1, len(text)), r) for r in range(len(text))
        ):
            parts, prev = [], 0
            for cut in cuts:
                parts.append(text[prev:cut])
                prev = cut
            parts.append(text[prev:])
            yield parts

    def test_every_split_matches_one_pass(self) -> None:
        for text in self.TEXTS:
            expected = strip_markdown(text)
            with self.subTest(text=text):
                for parts in self.every_split(text):
                    stripper = StreamingMarkdownStripper()
                    got = "".join(stripper.feed(p) for p in parts) + stripper.flush()
                    self.assertEqual(got, expected, msg=f"split {parts!r}")

    def test_flush_releases_everything(self) -> None:
        stripper = StreamingMarkdownStripper()
        self.assertEqual(stripper.feed("**wich"), "")
        self.assertEqual(stripper.feed("tig**"), "")
        self.assertEqual(stripper.flush(), "wichtig")

    def test_empty_stream(self) -> None:
        stripper = StreamingMarkdownStripper()
        self.assertEqual(stripper.feed(""), "")
        self.assertEqual(stripper.flush(), "")


if __name__ == "__main__":
    unittest.main()
