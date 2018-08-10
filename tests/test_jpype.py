"""
Module to test methods based on jpype1.
"""
import pytest
import imgread.read as ir
from test_read import check_core_md, check_single_md


@pytest.mark.myjpype
class TestJpype:
    """Test both metadata and data with all files, OME and LIF, using
    javabridge OMEXmlMetadata into bioformats image reader.

    """

    def setup_class(cls):
        cls.read = ir.read_jpype

    def test_metadata_data(self, read_all):
        test_d, md, wrapper = read_all
        check_core_md(md, test_d)
        # check_data(wrapper, test_d['data'])


@pytest.mark.pims
class TestPims:
    """Test both metadata and data with all files, OME and LIF, using
    javabridge OMEXmlMetadata into bioformats image reader.

    """

    def setup_class(cls):
        cls.read = ir.read_pims

    def test_metadata_data(self, read_TIF):
        test_d, md, wrapper = read_TIF
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
    def test_metadata_data_LIF(self, read_LIF, key):
        test_d, md, wrapper = read_LIF
        check_single_md(md, test_d, key)
