"""Module for testing command-line scripts."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from nima_io.__main__ import imgdiff

# tests path
tpath = Path(__file__).parent
datafolder = tpath / "data"


@pytest.fixture(
    params=[
        ("im1s1z3c5t_a.ome.tif", "im1s1z3c5t_b.ome.tif", "Files seem equal.\n"),
        ("im1s1z3c5t_a.ome.tif", "im1s1z2c5t_bmd.ome.tif", "Files differ.", "Metadata"),
        ("im1s1z3c5t_a.ome.tif", "im1s1z3c5t_bpix.ome.tif", "Files differ.\n"),
    ],
    ids=["Identical files", "Metadata diff", "Single-pixel diff"],
)
def image_pairs(request: pytest.FixtureRequest) -> tuple[Path, Path, str, str]:
    """Fixture that returns list of file paths and expected output."""
    file1, file2, expected_output, *rest = request.param
    matched = rest[0] if rest else None
    return datafolder / file1, datafolder / file2, expected_output, matched


def run_imgdiff(file1: Path, file2: Path) -> str:
    """Run imgdiff command and return the output."""
    runner = CliRunner()
    result = runner.invoke(imgdiff, [file1.as_posix(), file2.as_posix()])
    return result.output


def test_imgdiff(image_pairs: tuple[Path, Path, str, str]) -> None:
    """Test various image pairs."""
    file1, file2, expected_output, matched = image_pairs
    output = run_imgdiff(file1, file2)

    assert expected_output in output
    if matched:
        assert matched in output
