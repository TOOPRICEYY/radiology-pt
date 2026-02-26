"""Collapsible real-time streaming display for LLM output in the terminal."""

from __future__ import annotations

import os
import shutil
import sys
import time


# ── ANSI helpers ─────────────────────────────────────────────────────────────

_IS_TTY = sys.stderr.isatty()

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_CLEAR_LINE = "\033[2K"
_CURSOR_UP = "\033[{n}A"
_CLEAR_TO_END = "\033[J"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}" if _IS_TTY else text


def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


# ── Streaming display ────────────────────────────────────────────────────────

class StreamingDisplay:
    """Streams LLM tokens to stderr with a collapsible view.

    Usage:
        display = StreamingDisplay(label="Iteration 0", live=True)
        for chunk in openai_stream:
            token = chunk.choices[0].delta.content or ""
            display.feed(token)
        full_text = display.finish()

    After finish():
      - If live=True (default): collapses to a one-line summary
      - If live=False or not a TTY: just accumulates silently
      - If expand=True: leaves the full output visible
    """

    def __init__(
        self,
        label: str = "LLM",
        live: bool = True,
        expand: bool = False,
        max_visible_lines: int = 0,
    ):
        self._label = label
        self._live = live and _IS_TTY
        self._expand = expand
        self._max_visible = max_visible_lines or _max_display_lines()
        self._chunks: list[str] = []
        self._lines_printed = 0
        self._started = False
        self._start_time = 0.0
        self._token_count = 0

    def feed(self, token: str) -> None:
        """Feed a single streamed token."""
        if not token:
            return

        self._token_count += 1
        self._chunks.append(token)

        if not self._live:
            return

        if not self._started:
            self._start_time = time.monotonic()
            self._started = True
            # Print the header
            _write(_c(_BOLD + _CYAN, f"  ┌─ {self._label} "))
            _write(_c(_DIM, "(streaming...)"))
            _write("\n")
            _write(_HIDE_CURSOR)
            self._lines_printed = 1

        # Render the current visible window of output
        self._render_live()

    def finish(self) -> str:
        """Finalize the stream. Returns the full accumulated text."""
        full_text = "".join(self._chunks)
        elapsed = time.monotonic() - self._start_time if self._started else 0

        if not self._live:
            return full_text

        if not self._started:
            return full_text

        _write(_SHOW_CURSOR)

        if self._expand:
            # Replace the streaming header with a final one, leave content
            self._erase_output()
            _write(_c(_BOLD + _CYAN, f"  ┌─ {self._label} "))
            _write(_c(_GREEN, f"({len(full_text)} chars, {elapsed:.1f}s)"))
            _write("\n")
            # Print full content with border
            for line in full_text.splitlines():
                _write(f"  {_c(_DIM, '│')} {line}\n")
            _write(_c(_DIM, f"  └{'─' * 40}") + "\n")
        else:
            # Collapse: erase everything and print one summary line
            self._erase_output()
            tok_s = self._token_count / elapsed if elapsed > 0 else 0
            _write(_c(_BOLD + _CYAN, f"  ┌─ {self._label} "))
            _write(_c(_GREEN, f"({len(full_text)} chars, {elapsed:.1f}s, {tok_s:.0f} tok/s)"))
            _write("\n")
            # Show a few-line preview
            preview_lines = full_text.splitlines()[:3]
            for line in preview_lines:
                truncated = line[:_term_width() - 8]
                _write(f"  {_c(_DIM, '│')} {_c(_DIM, truncated)}\n")
            remaining = len(full_text.splitlines()) - 3
            if remaining > 0:
                _write(f"  {_c(_DIM, '│')} {_c(_DIM, f'... +{remaining} more lines')}\n")
            _write(_c(_DIM, f"  └{'─' * 40}") + "\n")

        return full_text

    def _render_live(self) -> None:
        """Re-render the visible portion of the streamed output."""
        full = "".join(self._chunks)
        lines = full.splitlines()
        # Keep only the tail that fits in our display window
        visible = lines[-self._max_visible:]

        # Erase previous content lines (not the header)
        content_lines = self._lines_printed - 1  # subtract header
        if content_lines > 0:
            # Move up and clear
            _write(f"\033[{content_lines}A")
            _write(_CLEAR_TO_END)

        # Print visible lines
        for line in visible:
            truncated = line[:_term_width() - 8]
            _write(f"  {_c(_DIM, '│')} {truncated}\n")

        # Status line
        elapsed = time.monotonic() - self._start_time
        n_chars = len(full)
        _write(f"  {_c(_DIM, '│')} {_c(_YELLOW, f'[{n_chars} chars, {elapsed:.1f}s]')}\n")

        self._lines_printed = 1 + len(visible) + 1  # header + content + status

    def _erase_output(self) -> None:
        """Erase all printed lines."""
        if self._lines_printed > 0:
            _write(f"\033[{self._lines_printed}A")
            _write(_CLEAR_TO_END)
            self._lines_printed = 0


class NullDisplay:
    """No-op display when streaming is disabled."""

    def feed(self, token: str) -> None:
        pass

    def finish(self) -> str:
        return ""


def _write(s: str) -> None:
    sys.stderr.write(s)
    sys.stderr.flush()


def _max_display_lines() -> int:
    """How many lines of LLM output to show in the scrolling view."""
    _, rows = shutil.get_terminal_size((80, 24))
    # Use about half the terminal height, min 8
    return max(8, rows // 2)
