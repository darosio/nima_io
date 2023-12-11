"""Module for testing command-line scripts."""
from __future__ import annotations

import subprocess
from pathlib import Path

# import pytest
# from click.testing import CliRunner
# from nima_io.__main__ import imgdiff

# tests path
tpath = Path(__file__).parent
datafolder = tpath / "data"

# @pytest.fixture
# def runner() -> CliRunner:
#     """Fixture for invoking command-line interfaces."""
#     return CliRunner()


class TestImgdiff:
    """Test the 'imgdiff' command through os.system/subprocess.

    Verify the behavior without directly invoking specific methods or units within
    the nima_io package.
    """

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
        result = subprocess.run(cmd_line, capture_output=True, text=True)
        return result.stdout

    def test_equal_files(self) -> None:
        """Test equal files."""
        output = self.run_imgdiff(self.fp_a, self.fp_b)
        assert output == "Files seem equal.\n"

    def test_different_files(self) -> None:
        """Test different files."""
        output = self.run_imgdiff(self.fp_a, self.fp_bmd)
        assert output == "Files differ.\n"

    def test_singlepixeldifferent_files(self) -> None:
        """Test different pixels data, same metadata."""
        output = self.run_imgdiff(self.fp_a, self.fp_bpix)
        assert output == "Files differ.\n"


# def test_imgdiff() -> None:
#     """Test default case."""
#     a = Path("/home/dan/workspace/nima_io/tests/data/im1s1z3c5t_a.ome.tif")
#     b = Path("/home/dan/workspace/nima_io/tests/data/im1s1z3c5t_b.ome.tif")
#     runner = CliRunner()
#     result = runner.invoke(imgdiff, [str(a), str(b)])
#     print(result.output)  # Add this line to print the output for debugging
#     assert result.exit_code == 0
#     assert "Files seem equal.\n" in result.output


# pytestmark = pytest.mark.usefixtures("jvm")


# @pytest.fixture
# def jvm():
#     """Ensure a running JVM."""
#     ir.ensure_vm()
#     yield
#     javabridge.kill_vm()


# def setup_module():
#     """Ensure a running JVM."""
#     ir.ensure_vm()


# def teardown_module():
#     """Try to detach from the JVM as we cannot kill it.
#     https://github.com/LeeKamentsky/python-javabridge/issues/88
#     """
#     print("Killing VirtualMachine")
#     javabridge.kill_vm()
