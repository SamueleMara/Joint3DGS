from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress_iter(iterable: Iterable[T], *, desc: str, total: int | None = None) -> Iterator[T]:
    yield from tqdm(iterable, desc=desc, total=total, leave=False)


def stage_bar(*, desc: str, total: int) -> tqdm:
    return tqdm(total=total, desc=desc, leave=True)
