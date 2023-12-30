"""Module for testing command-line scripts."""
from __future__ import annotations

import subprocess
from pathlib import Path

# from typing import Any
# import pytest
# from click.testing import CliRunner
# from nima_io.__main__ import imgdiff

# tests path
tpath = Path(__file__).parent
datafolder = tpath / "data"


# @pytest.fixture(
#     params=[
#         ("im1s1z3c5t_a.ome.tif", "im1s1z3c5t_b.ome.tif", "Files seem equal."),
#         ("im1s1z3c5t_a.ome.tif", "im1s1z3c5t_bmd.ome.tif", "Files differ."),
#         ("im1s1z3c5t_a.ome.tif", "im1s1z3c5t_bpix.ome.tif", "Files differ."),
#     ]
# )
# def image_data(request: Any) -> tuple[str, str, str]:
#     """Fixture that returns list of file paths and expected output."""
#     file1, file2, expected_output = request.param
#     file1 = str(datafolder / file1)
#     file2 = str(datafolder / file2)
#     return file1, file2, expected_output


# def test_images(image_data: tuple[str, str, str]) -> None:
#     """Test that `run_imgdiff` returns the expected output."""
#     file1, file2, expected_output = image_data
#     runner = CliRunner()
#     result = runner.invoke(imgdiff, [file1, file2])
#     assert result.output == expected_output


class TestImgdiff:
    """Test the 'imgdiff' command through os.system/subprocess.

    Verify the behavior without directly invoking specific methods or units within
    the nima_io package.
    """

    fp_a: Path  # first image file
    fp_b: Path  # second image file
    fp_bmd: Path  # second image file with different metadata
    fp_bpix: Path  # second image file with different pixel data

    @classmethod
    def setup_class(cls) -> None:
        """Define data files for testing imgdiff."""
        cls.fp_a = datafolder / "im1s1z3c5t_a.ome.tif"
        cls.fp_b = datafolder / "im1s1z3c5t_b.ome.tif"
        cls.fp_bmd = datafolder / "im1s1z2c5t_bmd.ome.tif"
        cls.fp_bpix = datafolder / "im1s1z3c5t_bpix.ome.tif"

    def run_imgdiff(self, file1: Path, file2: Path) -> str:
        """Run imgdiff command and return the output."""
        cmd_line = ["imgdiff", str(file1), str(file2)]
        result = subprocess.run(cmd_line, capture_output=True, text=True, check=False)
        return result.stdout

    def test_equal_files(self) -> None:
        """Test equal files."""
        output = self.run_imgdiff(self.fp_a, self.fp_b)
        assert output == "Files seem equal.\n"

    def test_different_files(self) -> None:
        """Test different files."""
        output = self.run_imgdiff(self.fp_a, self.fp_bmd)
        assert "Metadata mismatch:" in output
        assert "Files differ." in output

    def test_singlepixeldifferent_files(self) -> None:
        """Test different pixels data, same metadata."""
        output = self.run_imgdiff(self.fp_a, self.fp_bpix)
        assert output == "Files differ.\n"
