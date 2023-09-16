from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent as _dedent


def dedent(text):
    return _dedent(text[1:])


def remove_trailing_spaces(text):
    return ''.join(f'{line.rstrip()}\n' for line in text.splitlines(True))


def get_current_timestamp():
    return datetime.now(timezone.utc).timestamp()
