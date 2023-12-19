"""Test read.py module.

It compares the functionality of the following components:
- showinf
- bioformats
- javabridge access to java classes
- OMEXMLMetadataImpl into image_reader
- [ ] pims
- [ ] jpype

Tests include:
- FEI multichannel
- FEI tiled
- OME std multichannel
- LIF

It also includes a test for FEI tiled with a void tile.
"""
from __future__ import annotations

import os

import bioformats
import pytest

import nima_io.read as ir  # type: ignore[import-untyped]
from nima_io.read import MDValueType


def check_core_md(md: MDValueType, test_md_data_dict: MDValueType) -> None:
    """Compare (read vs. expected) core metadata.

    Parameters
    ----------
    md : MDValueType
        Read metadata.
    test_md_data_dict : MDValueType
        Expected metadata as specified in the test data.

    """
    assert md["SizeS"] == test_md_data_dict.SizeS
    assert md["SizeX"] == test_md_data_dict.SizeX
    assert md["SizeY"] == test_md_data_dict.SizeY
    assert md["SizeC"] == test_md_data_dict.SizeC
    assert md["SizeT"] == test_md_data_dict.SizeT
    if "SizeZ" in md:
        assert md["SizeZ"] == test_md_data_dict.SizeZ
    else:
        for i, v in enumerate(test_md_data_dict.SizeZ):  # for LIF file
            assert md["series"][i]["SizeZ"] == v
    assert md["PhysicalSizeX"] == test_md_data_dict.PhysicalSizeX


def check_single_md(md: MDValueType, test_md_data_dict: MDValueType, key: str) -> None:
    """Compare (read vs. expected) single core metadata specified by key.

    Parameters
    ----------
    md : MDValueType
        Read metadata.
    test_md_data_dict : MDValueType
        Expected metadata as specified in the test data.
    key : str
        The key specifying the single core metadata.

    """
    if key in md:
        assert md[key] == getattr(test_md_data_dict, key)
    else:
        for i, v in enumerate(getattr(test_md_data_dict, key)):  # e.g. SizeZ in LIF
            assert md["series"][i][key] == v


def check_data(
    wrapper: bioformats.formatreader.ImageReader, data: list[list[float | int]]
) -> None:
    """Compare data values with the expected values.

    Parameters
    ----------
    wrapper : bioformats.formatreader.ImageReader
        An instance of the wrapper used for reading data.
    data : list[list[float | int]]
        A list of lists containing information about each test data.
        Each inner list should have the format [series, x, y, channel, time, z, value].

    """
    if data:
        for ls in data:
            series, x, y, channel, time, z, value = ls[:7]
            a = wrapper.read(c=channel, t=time, series=series, z=z, rescale=False)
            # Y then X
            assert a[y, x] == value


def test_file_not_found() -> None:
    """It raises the expected exception when attempting to read a non-existent file."""
    with pytest.raises(Exception) as excinfo:
        ir.read(os.path.join("datafolder", "pippo.tif"))
    expected_error_message = (
        f"File not found: {os.path.join('datafolder', 'pippo.tif')}"
    )
    assert expected_error_message in str(excinfo.value)


class TestMdData:
    """Test both metadata and data with all files, OME and LIF, using
    javabridge OMEXmlMetadata into bioformats image reader.

    """

    @classmethod
    def setup_class(cls) -> None:
        cls.read = ir.read
        print("Starting VirtualMachine")
        # ir.ensure_vm()

    @classmethod
    def teardown_class(cls) -> None:
        print("Stopping VirtualMachine")
        # ir.release_vm()

    def test_metadata_data(self, read_all) -> None:
        test_d, md, wrapper = read_all
        check_core_md(md, test_d)
        check_data(wrapper, test_d.data)

    def test_tile_stitch(self, read_all) -> None:
        if read_all[0].filename == "t4_1.tif":
            md, wrapper = read_all[1:]
            stitched_plane = ir.stitch(md, wrapper)
            # Y then X
            assert stitched_plane[861, 1224] == 7779
            assert stitched_plane[1222, 1416] == 9626
            stitched_plane = ir.stitch(md, wrapper, t=2, c=3)
            assert stitched_plane[1236, 1488] == 6294
            stitched_plane = ir.stitch(md, wrapper, t=1, c=2)
            assert stitched_plane[564, 1044] == 8560
        else:
            pytest.skip("Test file with a single tile.")

    def test_void_tile_stitch(self, read_void_tile) -> None:
        # ir.ensure_vm()
        # md, wrapper = ir.read(img_FEI_void_tiled)
        _, md, wrapper = read_void_tile
        stitched_plane = ir.stitch(md, wrapper, t=0, c=0)
        assert stitched_plane[1179, 882] == 6395
        stitched_plane = ir.stitch(md, wrapper, t=0, c=1)
        assert stitched_plane[1179, 882] == 3386
        stitched_plane = ir.stitch(md, wrapper, t=0, c=2)
        assert stitched_plane[1179, 882] == 1690
        stitched_plane = ir.stitch(md, wrapper, t=1, c=0)
        assert stitched_plane[1179, 882] == 6253
        stitched_plane = ir.stitch(md, wrapper, t=1, c=1)
        assert stitched_plane[1179, 882] == 3499
        stitched_plane = ir.stitch(md, wrapper, t=1, c=2)
        assert stitched_plane[1179, 882] == 1761
        stitched_plane = ir.stitch(md, wrapper, t=2, c=0)
        assert stitched_plane[1179, 882] == 6323
        stitched_plane = ir.stitch(md, wrapper, t=2, c=1)
        assert stitched_plane[1179, 882] == 3354
        stitched_plane = ir.stitch(md, wrapper, t=2, c=2)
        assert stitched_plane[1179, 882] == 1674
        stitched_plane = ir.stitch(md, wrapper, t=3, c=0)
        assert stitched_plane[1179, 882] == 6291
        stitched_plane = ir.stitch(md, wrapper, t=3, c=1)
        assert stitched_plane[1179, 882] == 3373
        stitched_plane = ir.stitch(md, wrapper, t=3, c=2)
        assert stitched_plane[1179, 882] == 1615
        stitched_plane = ir.stitch(md, wrapper, t=3, c=0)
        assert stitched_plane[1213, 1538] == 704
        stitched_plane = ir.stitch(md, wrapper, t=3, c=1)
        assert stitched_plane[1213, 1538] == 422
        stitched_plane = ir.stitch(md, wrapper, t=3, c=2)
        assert stitched_plane[1213, 1538] == 346
        # Void tiles are set to 0
        assert stitched_plane[2400, 2400] == 0
        assert stitched_plane[2400, 200] == 0


def test_first_nonzero_reverse() -> None:
    assert ir.first_nonzero_reverse([0, 0, 2, 0]) == -2
    assert ir.first_nonzero_reverse([0, 2, 1, 0]) == -2
    assert ir.first_nonzero_reverse([1, 2, 1, 0]) == -2
    assert ir.first_nonzero_reverse([2, 0, 0, 0]) == -4


def test__convert_num() -> None:
    """Test num conversions and raise with printout."""
    assert ir.convert_java_numeric_field(None) is None
    assert ir.convert_java_numeric_field("0.976") == 0.976
    assert ir.convert_java_numeric_field(0.976) == 0.976
    assert ir.convert_java_numeric_field(976) == 976
    assert ir.convert_java_numeric_field("976") == 976


def test_next_tuple() -> None:
    assert ir.next_tuple([1], True) == [2]
    assert ir.next_tuple([1, 1], False) == [2, 0]
    assert ir.next_tuple([0, 0, 0], True) == [0, 0, 1]
    assert ir.next_tuple([0, 0, 1], True) == [0, 0, 2]
    assert ir.next_tuple([0, 0, 2], False) == [0, 1, 0]
    assert ir.next_tuple([0, 1, 0], True) == [0, 1, 1]
    assert ir.next_tuple([0, 1, 1], True) == [0, 1, 2]
    assert ir.next_tuple([0, 1, 2], False) == [0, 2, 0]
    assert ir.next_tuple([0, 2, 0], False) == [1, 0, 0]
    assert ir.next_tuple([1, 0, 0], True) == [1, 0, 1]
    assert ir.next_tuple([1, 1, 1], False) == [1, 2, 0]
    assert ir.next_tuple([1, 2, 0], False) == [2, 0, 0]
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([2, 0, 0], False)
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([1, 0], False)
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([1], False)
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([], False)
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([], True)


def test_get_allvalues_grouped() -> None:
    # k = 'getLightPathExcitationFilterRef' # npar = 3 can be more tidied up
    # #k = 'getChannelLightSourceSettingsID' # npar = 2
    # #k = 'getPixelsSizeX' # npar = 1
    # #k = 'getExperimentType'
    # #k = 'getImageCount' # npar = 0
    # k = 'getPlanePositionZ'

    # get_allvalues(metadata, k, 2)
    pass


class TestMetadata2:
    @classmethod
    def setup_class(cls) -> None:
        cls.read = ir.read2
        print("Starting VirtualMachine")
        # ir.ensure_vm()

    @classmethod
    def teardown_class(cls) -> None:
        print("Better not Killing VirtualMachine")
        # javabridge.kill_vm()

    # def test_convert_value(self, filepath, SizeS, SizeX, SizeY, SizeC, SizeT,
    #                        SizeZ, PhysicalSizeX, data):
    #     """Test conversion from java metadata value."""
    #     print(filepath)

    def test_metadata_data2(self, read_all) -> None:
        test_d, md2, wrapper = read_all
        md = {
            "SizeS": md2["ImageCount"][0][1],
            "SizeX": md2["PixelsSizeX"][0][1],
            "SizeY": md2["PixelsSizeY"][0][1],
            "SizeC": md2["PixelsSizeC"][0][1],
            "SizeT": md2["PixelsSizeT"][0][1],
        }
        if len(md2["PixelsSizeZ"]) == 1:
            md["SizeZ"] = md2["PixelsSizeZ"][0][1]
        elif len(md2["PixelsSizeZ"]) > 1:
            md["series"] = [{"SizeZ": ls[1]} for ls in md2["PixelsSizeZ"]]
        if "PixelsPhysicalSizeX" in md2:
            # this is with unit
            md["PhysicalSizeX"] = round(md2["PixelsPhysicalSizeX"][0][1][0], 6)
        else:
            md["PhysicalSizeX"] = None
        check_core_md(md, test_d)
        check_data(wrapper, test_d.data)


def setup_module() -> None:
    ir.ensure_vm()


def teardown_module() -> None:
    # javabridge.kill_vm()
    ir.release_vm()
