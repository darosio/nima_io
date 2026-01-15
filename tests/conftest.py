"""Provide fixtures for nima_io tests shared across different modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

DATA_DIR = Path(__file__).parent / "data"
FILENAMES_TXT = Path(__file__).parent / "data.filenames.txt"


def _default_filenames() -> list[str]:
    if FILENAMES_TXT.is_file():
        return [
            line.strip()
            for line in FILENAMES_TXT.read_text().splitlines()
            if line.strip()
        ]
    return []


def require_test_data(filenames: Iterable[str] | None = None) -> None:
    """Skip tests that rely on external sample data when it is unavailable."""
    expected = list(filenames) if filenames is not None else _default_filenames()
    missing = [name for name in expected if not (DATA_DIR / name).is_file()]
    if missing:
        pytest.skip(
            (
                "Required test data missing: "
                f"{', '.join(sorted(set(missing)))}. "
                "See README Development section to fetch files "
                "(git annex pull in tests/data)."
            ),
            allow_module_level=True,
        )
