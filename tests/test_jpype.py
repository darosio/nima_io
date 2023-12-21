"""Module to test methods based on jpype1."""
from __future__ import annotations

from typing import Any

from test_read import check_core_md

import nima_io.read as ir  # type: ignore[import-untyped]


class TestScyjava:
    """Test metadata and data retrieval with different files.

    Uses scyjava. Files include OME and LIF formats.
    """

    @classmethod
    def setup_class(cls: type[TestJpype]) -> None:
        """Assign the `read` class attribute to the `ir.read_jpype` function."""
        cls.read = ir.read

    def test_metadata_data(self, read_all: tuple[dict, dict, Any]) -> None:
        """Test metadata and data retrieval."""
        test_d, md, wrapper = read_all
        check_core_md(md, test_d)
        # check_data(wrapper, test_d['data'])


class TestJpype:
    """Test metadata and data retrieval with different files.

    Uses jpype/javabridge OMEXmlMetadata integrated into the bioformats image reader.
    Files include OME and LIF formats.
    """

    @classmethod
    def setup_class(cls: type[TestJpype]) -> None:
        """Assign the `read` class attribute to the `ir.read_jpype` function."""
        cls.read = ir.read_jpype

    def test_metadata_data(self, read_all: tuple[dict, dict, Any]) -> None:
        """Test metadata and data retrieval."""
        test_d, md, wrapper = read_all
        check_core_md(md, test_d)
        # check_data(wrapper, test_d['data'])


class TestPims:
    """Test pims reading.

    Both metadata and data with all files (OME and LIF).
    """

    @classmethod
    def setup_class(cls) -> None:
        """Set up the TestPims class for testing."""
        cls.read = ir.read_pims

    def test_metadata_data(self, read_all) -> None:
        """Test core metadata and data reading."""
        test_d, md, wrapper = read_all
        check_core_md(md, test_d)
        # check_data(wrapper, test_d['data'])
