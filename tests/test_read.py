"""Tests for the `read.py` module.

This module compares the functionality of different libraries for reading image data:
- scyjava + jpype
- jpype
- pims

The test cases cover various scenarios, including:
- FEI multichannel
- FEI tiled
- OME std multichannel
- LIF
- FEI tiled with void tiles

The tests focus on reading metadata and data, as well as stitching tiles.

"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

import nima_io.read as ir  # type: ignore[import-untyped]

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

# "bigtiff" - bigtiff - tdata / "LC26GFP_1.tf8"

list_test_data = [td_img_tile, td_img_void_tile, td_imgsingle, td_lif, td_mcts]
# Used to be ["FEI multichannel", "FEI multitiles", "OME std test", "Leica LIF"]
ids = ["img_tile", "img_void_tile", "imgsingle", "lif", "mcts"]


@pytest.fixture(params=[ir.read, ir.read_jpype, ir.read_pims])
def read_functions(
    request: pytest.FixtureRequest,
) -> Callable[[str], Any]:
    return request.param


def common_tdata(
    request: pytest.FixtureRequest, read_functions: Callable[[str], Any]
) -> tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]:
    """Yield test data and read for multitile file with missing tiles."""
    td = request.param
    read = read_functions
    filepath = str(tpath / td.filename)
    md, wr = read(filepath)
    return td, md, wr


@pytest.fixture(params=list_test_data, ids=ids)
def tdata_all(
    request: pytest.FixtureRequest, read_functions: Callable[[str], Any]
) -> tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]:
    return common_tdata(request, read_functions)


@pytest.fixture(params=[td_img_tile], ids=[ids[0]])
def tdata_img_tile(
    request: pytest.FixtureRequest, read_functions: Callable[[str], Any]
) -> tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]:
    return common_tdata(request, read_functions)


@pytest.fixture(params=[td_img_void_tile], ids=[ids[1]])
def tdata_img_void_tile(
    request: pytest.FixtureRequest, read_functions: Callable[[str], Any]
) -> tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]:
    return common_tdata(request, read_functions)


def test_file_not_found() -> None:
    """It raises the expected exception when attempting to read a non-existent file."""
    with pytest.raises(FileNotFoundError) as excinfo:
        ir.read(os.path.join("datafolder", "pippo.tif"))
    expected_error_message = (
        f"File not found: {os.path.join('datafolder', 'pippo.tif')}"
    )
    assert expected_error_message in str(excinfo.value)


def test_metadata_data(
    tdata_all: tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]
) -> None:
    test_d, md, wrapper = tdata_all
    # Check core_md
    assert md.core.size_s == test_d.SizeS
    assert md.core.size_x == test_d.SizeX
    assert md.core.size_y == test_d.SizeY
    assert md.core.size_c == test_d.SizeC
    assert md.core.size_t == test_d.SizeT
    assert md.core.size_z == test_d.SizeZ
    assert [vs.x for vs in md.core.voxel_size] == test_d.PhysicalSizeX
    # Check data
    if test_d.data:
        for ls in test_d.data:
            series, x, y, channel, time, z, value = ls[:7]
            a = wrapper.read(c=channel, t=time, series=series, z=z, rescale=False)
            assert a[y, x] == value  # Mind the order: Y then X


def test_tile_stitch(
    tdata_img_tile: tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]
) -> None:
    td, md3, wrapper = tdata_img_tile
    # if not td.filename == "t4_1.tif":
    #     pytest.skip("Test file with a single tile.")
    md = md3.core
    stitched_plane = ir.stitch(md, wrapper)
    # Y then X
    assert stitched_plane[861, 1224] == 7779
    assert stitched_plane[1222, 1416] == 9626
    stitched_plane = ir.stitch(md, wrapper, t=2, c=3)
    assert stitched_plane[1236, 1488] == 6294
    stitched_plane = ir.stitch(md, wrapper, t=1, c=2)
    assert stitched_plane[564, 1044] == 8560


def test_void_tile_stitch(
    tdata_img_void_tile: tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]
) -> None:
    td, md3, wrapper = tdata_img_void_tile
    md = md3.core
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
    # TODO: Add tests for Metadata.full.
    # k = 'getLightPathExcitationFilterRef' # npar = 3 can be more tidied up
    # #k = 'getChannelLightSourceSettingsID' # npar = 2
    # #k = 'getPixelsSizeX' # npar = 1
    # #k = 'getExperimentType'
    # #k = 'getImageCount' # npar = 0
    # k = 'getPlanePositionZ'

    # get_allvalues(metadata, k, 2)
    pass
