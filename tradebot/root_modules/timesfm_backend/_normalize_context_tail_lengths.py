from __future__ import annotations

from .shared import *  # noqa: F401,F403


def _normalize_context_tail_lengths(context_tail_lengths: Sequence[int] | None) -> tuple[int, ...]:
    if not context_tail_lengths:
        return (0,)

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_value in context_tail_lengths:
        value = max(0, int(raw_value))
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized) if normalized else (0,)
