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

#: A single word character, for the ``_`` rules below. Kept as a compiled
#: pattern so the stream and the one-pass form ask exactly the same question.
_WORD_RE = re.compile(r"\w")

#: Rules that can span a space but never a newline, applied in this order.
#:
#: ``*`` and ``~~`` may open and close inside a word. Markdown allows that, and
#: forbidding it is what left stray asterisks in Chinese replies: there every
#: marker touches a word character on both sides (``今天**很好**啊``), and so
#: does every mixed sentence (``**粗体**and english``).
#:
#: ``_`` is the opposite - an underscore inside a word is not emphasis - so it
#: keeps its ``\w`` guards. ``\w`` is deliberate there: it spans every script,
#: which is what protects ``light.kitchen_lamp``, ``Café_test_`` and ``你_好_啊``
#: alike.
#:
#: The one thing kept out of the ``*`` rules is arithmetic: in ``5**2`` and
#: ``3*4`` the run has a digit on both sides and means an exponent or a product.
#: A digit on one side only is still emphasis, so ``**23**度`` is stripped.
#:
#: The single-character rules also refuse a marker that touches its own double
#: form. Whenever ``**`` declines - over arithmetic, or over the space in
#: ``a ** b ** c`` - the ``*`` rule would otherwise take half of each run and
#: leave the other half to be read out.
_INLINE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?!(?<=[0-9])\*\*[0-9])\*\*(?!\s)(.+?)(?<!\s)\*\*"), r"\1"),
    (re.compile(r"(?!(?<=[0-9])\*[0-9])(?<!\*)\*(?![\s*])(.+?)(?<![\s*])\*(?!\*)"), r"\1"),
    (re.compile(r"(?<!\w)__(?!\s)(.+?)(?<!\s)__(?!\w)"), r"\1"),
    (re.compile(r"(?<!\w)_(?![\s_])(.+?)(?<![\s_])_(?!\w)"), r"\1"),
    (re.compile(r"~~(?!\s)(.+?)(?<!\s)~~"), r"\1"),
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


def _without_code(text: str) -> str:
    """Return ``text`` as the line and inline rules will see it.

    Fences and backticks are removed first in :func:`_strip_core`, and that can
    hand a rule something the raw text never showed it: ```` #` ```` becomes a
    bare ``#``, which then swallows the space behind it as a heading.
    """
    return _CODE_FENCE_RE.sub("", text).replace("`", "")


def _strip_core(text: str, *, at_line_start: bool) -> str:
    """Apply every rule, without the outer whitespace trim.

    ``at_line_start`` says whether this fragment begins a line. When it does
    not, the line-anchored rules must not fire on its first line - it is the
    middle of a line that was already partly emitted.
    """
    if not text:
        return text

    guarded = text if at_line_start else _CONTINUATION + text

    guarded = _without_code(guarded)
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
    inside ``light.kitchen_lamp`` is preceded by a word character, so no ``_``
    rule can start there and the stream does not have to wait for it. ``*`` and
    ``~~`` carry no such guard, so for them any position is a possible opening.

    Erring towards ``True`` only delays a cut, so this stays deliberately
    coarse: the arithmetic and adjacency exceptions carried by the ``*``
    rules are not repeated here.
    """
    residue = _residue(line)

    if "[" in residue:
        return True

    for index, char in enumerate(residue):
        if char not in "*_~":
            continue
        run = residue[index:]
        marker = next((m for m in _MARKERS if run.startswith(m)), None)
        if marker is None:
            continue
        if marker in ("_", "__") and index and _WORD_RE.match(residue[index - 1]):
            continue  # (?<!\w) fails here, so no underscore rule can open
        rest = run[len(marker) :]
        if (
            marker in ("*", "**")
            and index
            and residue[index - 1].isdigit()
            and rest[:1].isdigit()
        ):
            continue  # 5**2 and 3*4: the star rules decline arithmetic
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
        self._refused_upto = 0

    def _safe_cut(self) -> int:
        """Return how much of the pending text can be emitted right now.

        A cut is only taken after whitespace: that keeps words intact and makes
        the lookbehinds see the same thing they would in one pass. Every
        ``continue`` below is a rule a later chunk could still let reach
        across the cut.

        Only positions the newest delta brought in are weighed. Every check
        below reads the pending text up to the cut and nothing after it, so a
        position that was refused once stays refused however much more arrives
        - and re-deciding it per delta made a reply the stripper cannot cut,
        an unclosed ``[`` say, cost time quadratic in its own length.
        """
        for index in range(len(self._pending), self._refused_upto, -1):
            if not self._pending[index - 1].isspace():
                continue
            body = self._pending[: len(self._pending[:index].rstrip())]
            if not body:
                continue
            # Cheap refusals first: stripping the whole body is by far the
            # costliest check here, and most candidates never reach it.
            #
            # The line-anchored rules see two different views of the body:
            # headings and quotes run on the code-stripped text, the list rule
            # only after the inline rules had their turn - and those can create
            # a marker that was not in the text ("*-*" leaves a bare "-"), so
            # both views are checked, one here and one below.
            line = _without_code(body).rsplit("\n", 1)[-1]
            if _LINE_PREFIX_ONLY_RE.fullmatch(line):
                # "# " or "- " with nothing after it: those rules match their
                # own trailing whitespace, so they need the rest of the line.
                continue
            if _has_open_construct(line):
                continue
            if _TRAILING_FENCE_RE.search(body):
                # A fence swallows the newline behind it, which has not arrived.
                continue

            stripped = _strip_core(body, at_line_start=self._at_line_start)
            if not stripped or stripped[-1].isspace():
                # Backticks, fences and the arrow rules delete text, so a body
                # that did not end in whitespace still can once it is stripped.
                # The one-pass form trims that off the end of the whole reply,
                # which is only possible while it is still in hand.
                continue
            if _LINE_PREFIX_ONLY_RE.fullmatch(stripped.rsplit("\n", 1)[-1]):
                continue
            return len(body)
        self._refused_upto = len(self._pending)
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
        # What is left starts at a new offset and follows different text, so
        # every position in it has to be weighed again.
        self._refused_upto = 0
        return self._emit(raw)

    def flush(self) -> str:
        """Release everything still held back, at the end of the stream."""
        if not self._pending:
            return ""
        raw, self._pending = self._pending, ""
        self._refused_upto = 0
        return self._emit(raw).rstrip()
