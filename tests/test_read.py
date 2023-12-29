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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

import nima_io.read as ir  # type: ignore[import-untyped]
from nima_io.read import MDValueType

tpath = Path(__file__).parent / "data"


@dataclass
class TDataItem:
    """Represent a data item in the test metadata dictionary."""

    filename: str
    SizeS: int
    SizeX: list[int]
    SizeY: list[int]
    SizeC: list[int]
    SizeT: list[int]
    SizeZ: list[int]
    PhysicalSizeX: list[float | None]
    data: list[list[int | float]]  # S, X, Y, C, T, Z, value


# data: series, x, y, channel, time, z, value

# "lif" - LIF_multiseries
data: list[list[int | float]] = [
    [4, 256, 128, 2, 0, 21, 2],
    [4, 285, 65, 2, 0, 21, 16],
    [4, 285, 65, 0, 0, 21, 14],
]  # max = 255
fp = "2015Aug28_TransHXB2_50min+DMSO.lif"
td_lif = TDataItem(
    fp, 5, [512], [512], [3], [1], [41, 40, 43, 39, 37], [0.080245], data
)

# "img_tile" - FEI_tiled -  # C=3 T=4 S=15
data = [
    [14, 509, 231, 0, 2, 0, 14580],
    [14, 509, 231, 1, 2, 0, 8436],
    [14, 509, 231, 2, 2, 0, 8948],
    [14, 509, 231, 3, 2, 0, 8041],
    [7, 194, 192, 1, 0, 0, 3783],
    [7, 194, 192, 1, 1, 0, 3585],
    [7, 194, 192, 1, 2, 0, 3403],
]
td_img_tile = TDataItem("t4_1.tif", 15, [512], [256], [4], [3], [1], [0.133333], data)

# "img_void_tile" -  -  # C=4 T=3 S=14 scattered
td_img_void_tile = TDataItem("tile6_1.tif", 14, [512], [512], [3], [4], [1], [0.2], [])

# "imgsingle" - FEI_multichannel -  # C=2 T=81
data = [
    [0, 610, 520, 0, 80, 0, 142],  # max = 212
    [0, 610, 520, 1, 80, 0, 132],  # max = 184
]
td_imgsingle = TDataItem("exp2_2.tif", 1, [1600], [1200], [2], [81], [1], [0.74], data)

# "mcts" - ome_multichannel -  # C=3 T=7
data = [[0, 200, 20, 2, 6, 0, -1]]
td_mcts = TDataItem(
    "multi-channel-time-series.ome.tif", 1, [439], [167], [3], [7], [1], [None], data
)

# bigtiff = tdata / "LC26GFP_1.tf8"  # bigtiff

list_test_data = [td_img_tile, td_img_void_tile, td_imgsingle, td_lif, td_mcts]
ids = ["img_tile", "img_void_tile", "imgsingle", "lif", "mcts"]


@pytest.fixture(scope="class", params=list_test_data, ids=ids)
def tdata_all(
    request: pytest.FixtureRequest,
) -> tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]:
    """Yield test data and read for multitile file with missing tiles."""
    td = request.param
    read = request.cls.read
    # filepath = os.path.join(os.path.dirname(request.fspath), "data", td.filename)
    filepath = str(tpath / td.filename)
    md, wr = read(filepath)
    return td, md, wr
    # yield td, md, wr
    # print("closing fixture: " + str(request.cls.read))


@pytest.fixture(params=[ir.read3, ir.read_jpype3, ir.read_pims3])
def read_functions(request):
    yield request.param


@pytest.fixture(params=list_test_data, ids=ids)
def tdata_allr(
    request: pytest.FixtureRequest, read_functions
) -> tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]:
    """Yield test data and read for multitile file with missing tiles."""
    td = request.param
    read = read_functions
    filepath = str(tpath / td.filename)
    md, wr = read(filepath)
    return td, md, wr


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


# bioformats.formatreader.ImageReader
def check_data(wrapper: Any, data: list[list[float | int]]) -> None:
    """Compare data values with the expected values.

    Parameters
    ----------
    wrapper : Any
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

    read: Callable[[str], Any]

    @classmethod
    def setup_class(cls) -> None:
        cls.read = ir.read

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


def test_reading(tdata_allr) -> None:
    test_d, md, wrapper = tdata_allr
    # check_core_md(md, test_d)
    assert md.core.size_s == test_d.SizeS
    assert md.core.size_x == test_d.SizeX
    assert md.core.size_y == test_d.SizeY
    assert md.core.size_c == test_d.SizeC
    assert md.core.size_t == test_d.SizeT
    assert md.core.size_z == test_d.SizeZ
    assert [vs.x for vs in md.core.voxel_size] == test_d.PhysicalSizeX
    # check_data(wrapper, test_d.data)
    if test_d.data:
        for ls in test_d.data:
            series, x, y, channel, time, z, value = ls[:7]
            a = wrapper.read(c=channel, t=time, series=series, z=z, rescale=False)
            # Y then X
            assert a[y, x] == value


class TestMdData3:
    """Test both metadata and data with all files, OME and LIF, using
    javabridge OMEXmlMetadata into bioformats image reader.

    """

    read: Callable[[str], Any]

    @classmethod
    def setup_class(cls) -> None:
        cls.read = ir.read3

    def test_metadata_data(self, tdata_all) -> None:
        test_d, md, wrapper = tdata_all
        # check_core_md(md, test_d)
        assert md.core.size_s == test_d.SizeS
        assert md.core.size_x == test_d.SizeX
        assert md.core.size_y == test_d.SizeY
        assert md.core.size_c == test_d.SizeC
        assert md.core.size_t == test_d.SizeT
        assert md.core.size_z == test_d.SizeZ
        assert [vs.x for vs in md.core.voxel_size] == test_d.PhysicalSizeX
        # check_data(wrapper, test_d.data)
        if test_d.data:
            for ls in test_d.data:
                series, x, y, channel, time, z, value = ls[:7]
                a = wrapper.read(c=channel, t=time, series=series, z=z, rescale=False)
                # Y then X
                assert a[y, x] == value

    def test_void_tile_stitch3(self, tdata_all) -> None:
        td, md3, wrapper = tdata_all
        if not td.filename == "tile6_1.tif":
            return None
        md = md3.core
        stitched_plane = ir.stitch3(md, wrapper, t=0, c=0)
        assert stitched_plane[1179, 882] == 6395
        stitched_plane = ir.stitch3(md, wrapper, t=0, c=1)
        assert stitched_plane[1179, 882] == 3386
        stitched_plane = ir.stitch3(md, wrapper, t=0, c=2)
        assert stitched_plane[1179, 882] == 1690
        stitched_plane = ir.stitch3(md, wrapper, t=1, c=0)
        assert stitched_plane[1179, 882] == 6253
        stitched_plane = ir.stitch3(md, wrapper, t=1, c=1)
        assert stitched_plane[1179, 882] == 3499
        stitched_plane = ir.stitch3(md, wrapper, t=1, c=2)
        assert stitched_plane[1179, 882] == 1761
        stitched_plane = ir.stitch3(md, wrapper, t=2, c=0)
        assert stitched_plane[1179, 882] == 6323
        stitched_plane = ir.stitch3(md, wrapper, t=2, c=1)
        assert stitched_plane[1179, 882] == 3354
        stitched_plane = ir.stitch3(md, wrapper, t=2, c=2)
        assert stitched_plane[1179, 882] == 1674
        stitched_plane = ir.stitch3(md, wrapper, t=3, c=0)
        assert stitched_plane[1179, 882] == 6291
        stitched_plane = ir.stitch3(md, wrapper, t=3, c=1)
        assert stitched_plane[1179, 882] == 3373
        stitched_plane = ir.stitch3(md, wrapper, t=3, c=2)
        assert stitched_plane[1179, 882] == 1615
        stitched_plane = ir.stitch3(md, wrapper, t=3, c=0)
        assert stitched_plane[1213, 1538] == 704
        stitched_plane = ir.stitch3(md, wrapper, t=3, c=1)
        assert stitched_plane[1213, 1538] == 422
        stitched_plane = ir.stitch3(md, wrapper, t=3, c=2)
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
    read: Callable[[str], Any]

    @classmethod
    def setup_class(cls) -> None:
        cls.read = ir.read2

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
