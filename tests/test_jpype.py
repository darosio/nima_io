"""Module to test methods based on jpype1."""
from typing import Any, Type

import pytest
from test_read import check_core_md, check_single_md

import nima_io.read as ir  # type: ignore[import-untyped]


class TestJpype:
    """Test metadata and data retrieval with different files.

    Uses jpype/javabridge OMEXmlMetadata integrated into the bioformats image reader.
    Files include OME and LIF formats.
    """

    @classmethod
    def setup_class(cls: Type["TestJpype"]) -> None:
        """Assign the `read` class attribute to the `ir.read_jpype` function."""
        cls.read = ir.read_jpype

    def test_metadata_data(self, read_all: tuple[dict, dict, Any]) -> None:
        """Test metadata and data retrieval."""
        test_d, md, wrapper = read_all
        check_core_md(md, test_d)
        # check_data(wrapper, test_d['data'])


class TestPims:
    """Test both metadata and data with all files, OME and LIF, using
    javabridge OMEXmlMetadata into bioformats image reader.

    """

    @classmethod
    def setup_class(cls) -> None:
        cls.read = ir.read_pims

    def test_metadata_data(self, read_tif) -> None:
        test_d, md, wrapper = read_tif
        check_core_md(md, test_d)
        # check_data(wrapper, test_d['data'])

    @pytest.mark.parametrize(
        "key",
        [
            "SizeS",
            "SizeX",
            "SizeY",
            "SizeC",
            "SizeT",
            "SizeZ",
            pytest.param(
                "PhysicalSizeX",
                marks=pytest.mark.xfail(
                    raises=AssertionError,
                    reason="loci 5.7.0 divides for SizeX instead of SizeX-1",
                ),
            ),
        ],
    )
    def test_metadata_data_lif(self, read_lif, key) -> None:
        test_d, md, wrapper = read_lif
        check_single_md(md, test_d, key)
