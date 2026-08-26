"""Markdown removal for voice assistants, in one pass or across a stream.

``strip_markdown`` is the whole-text form used for the final speech string.
``StreamingMarkdownStripper`` is the incremental form used by conversation.py
while a reply is still arriving: Home Assistant forwards every delta straight to
the UI and to text-to-speech, so stripping only the finished answer came far too
late - the asterisks had already been spoken.

The stripper holds text back until a point where the rules provably cannot reach
across, then emits everything before it. The invariant, checked exhaustively for
every possible split of a text, is::

    "".join(feed(chunk) for chunk in chunks) + flush() == strip_markdown(text)

Nothing is ever lost: whatever is still held back is released by ``flush()`` at
the end of the stream.
"""

from __future__ import annotations

import re

#: Emphasis markers, longest run first so ``**`` is considered before ``*``.
_MARKERS = ("~~", "**", "__", "*", "_")

#: ASCII word characters. Python's ``\w`` also matches CJK, which made the
#: emphasis lookarounds fail for Chinese text (e.g. ``**很好**`` right next to
#: CJK, or ``好**啊``) and left stray asterisks in the spoken reply.
_ASCII_WORD_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"

#: Rules that can span a space but never a newline, applied in this order.
_INLINE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![A-Za-z0-9_])\*\*(?!\s)(.+?)(?<!\s)\*\*(?![A-Za-z0-9_])"), r"\1"),
    (re.compile(r"(?<![A-Za-z0-9_])\*(?!\s)(.+?)(?<!\s)\*(?![A-Za-z0-9_])"), r"\1"),
    (re.compile(r"(?<![A-Za-z0-9_])__(?!\s)(.+?)(?<!\s)__(?![A-Za-z0-9_])"), r"\1"),
    (re.compile(r"(?<![A-Za-z0-9_])_(?!\s)(.+?)(?<!\s)_(?![A-Za-z0-9_])"), r"\1"),
    (re.compile(r"(?<![A-Za-z0-9_])~~(?!\s)(.+?)(?<!\s)~~(?![A-Za-z0-9_])"), r"\1"),
    (re.compile(r"!\[(.*?)\]\(.*?\)"), r"\1"),
    (re.compile(r"\[(.*?)\]\(.*?\)"), r"\1"),
)

_CODE_FENCE_RE = re.compile(r"```[a-z]*\n?")
#: A line that is still nothing but a heading / list / quote marker.
_LINE_PREFIX_ONLY_RE = re.compile(r"\s*(?:#{1,6}|[-*+]|>)?\s*")
#: A fence sitting at the very end of what we were about to emit.
_TRAILING_FENCE_RE = re.compile(r"```[a-z]*$")
_BLOCKQUOTE_RE = re.compile(r"(?m)^\s*>\s+")
_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
_LIST_RE = re.compile(r"(?m)^\s*[-*+]\s+")

#: Stands in for "there is text before this, and it is not a line start".
#: Not a word character, so the ``(?<!\w)`` lookbehinds behave as they would
#: after the space we cut on; not a newline, so ``^`` cannot match behind it.
_CONTINUATION = "\x00"


def _strip_core(text: str, *, at_line_start: bool) -> str:
    """Apply every rule, without the outer whitespace trim.

    ``at_line_start`` says whether this fragment begins a line. When it does
    not, the line-anchored rules must not fire on its first line - it is the
    middle of a line that was already partly emitted.
    """
    if not text:
        return text

    guarded = text if at_line_start else _CONTINUATION + text

    guarded = _CODE_FENCE_RE.sub("", guarded)
    guarded = guarded.replace("`", "")
    guarded = _BLOCKQUOTE_RE.sub("", guarded)
    guarded = _HEADING_RE.sub("", guarded)
    for pattern, replacement in _INLINE_RULES:
        guarded = pattern.sub(replacement, guarded)
    guarded = _LIST_RE.sub("", guarded)
    guarded = guarded.replace("→", "").replace("->", "")

    if not at_line_start and guarded.startswith(_CONTINUATION):
        guarded = guarded[len(_CONTINUATION) :]
    return guarded.replace(_CONTINUATION, "")


def strip_markdown(text: str) -> str:
    """Strip markdown formatting from a complete text, for TTS readability."""
    if not text:
        return text
    return _strip_core(text, at_line_start=True).strip()


def _residue(line: str) -> str:
    """Return ``line`` with every complete inline construct resolved."""
    for pattern, replacement in _INLINE_RULES:
        line = pattern.sub(replacement, line)
    return line


def _has_open_construct(line: str) -> bool:
    """Whether ``line`` ends with a construct a later chunk could still close.

    Only markers that could actually open something count. A stray underscore
    inside ``light.kitchen_lamp`` is preceded by a word character, so no rule
    can start there and the stream does not have to wait for it.
    """
    residue = _residue(line)

    if "[" in residue:
        return True

    for index, char in enumerate(residue):
        if char not in "*_~":
            continue
        if index and residue[index - 1] in _ASCII_WORD_CHARS:
            continue  # (?<![A-Za-z0-9_]) fails here, so no rule can open
        run = residue[index:]
        marker = next((m for m in _MARKERS if run.startswith(m)), None)
        if marker is None:
            continue
        rest = run[len(marker) :]
        # (?!\s): a marker followed by a space opens nothing. An unfinished
        # chunk ending on the marker itself may still be followed by anything.
        if rest and rest[0].isspace():
            continue
        return True

    return False


class StreamingMarkdownStripper:
    """Strip markdown from deltas as they arrive, without splitting constructs.

    One instance per streamed response. Call :meth:`feed` for every delta and
    :meth:`flush` once the stream ends.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._at_line_start = True
        self._emitted_text = False

    def _safe_cut(self) -> int:
        """Return how much of the pending text can be emitted right now.

        A cut is only taken after whitespace: that keeps words intact and makes
        the ``(?<!\\w)`` lookbehinds see the same thing they would in one pass.
        The three ``continue`` cases are the constructs a later chunk could
        still extend.
        """
        for index in range(len(self._pending), 0, -1):
            if not self._pending[index - 1].isspace():
                continue
            body = self._pending[: len(self._pending[:index].rstrip())]
            if not body:
                continue
            line = body.rsplit("\n", 1)[-1]
            if _LINE_PREFIX_ONLY_RE.fullmatch(line):
                # "# " or "- " with nothing after it: those rules match their
                # own trailing whitespace, so they need the rest of the line.
                continue
            if _has_open_construct(line):
                continue
            if _TRAILING_FENCE_RE.search(body):
                # A fence swallows the newline behind it, which has not arrived.
                continue
            return len(body)
        return 0

    def _emit(self, raw: str) -> str:
        """Strip one fragment and track the position it leaves us in."""
        out = _strip_core(raw, at_line_start=self._at_line_start)
        self._at_line_start = raw.endswith("\n")
        if not self._emitted_text:
            # strip_markdown() trims the whole text once; the leading half of
            # that has to happen on the first fragment that carries anything.
            out = out.lstrip()
            if out:
                self._emitted_text = True
        return out

    def feed(self, delta: str) -> str:
        """Add a delta and return whatever is now safe to pass on."""
        if delta:
            self._pending += delta
        # _safe_cut already leaves the trailing whitespace behind: the one-pass
        # form trims the end of the whole reply, which is only possible here if
        # that whitespace is still in hand when the stream ends.
        cut = self._safe_cut()
        if not cut:
            return ""
        raw, self._pending = self._pending[:cut], self._pending[cut:]
        return self._emit(raw)

    def flush(self) -> str:
        """Release everything still held back, at the end of the stream."""
        if not self._pending:
            return ""
        raw, self._pending = self._pending, ""
        return self._emit(raw).rstrip()
