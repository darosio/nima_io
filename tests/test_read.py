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

import hashlib
import unittest.mock
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

import nima_io.read as ir

tpath = Path(__file__).parent / "data"


@pytest.fixture()
def mock_urlopen(monkeypatch: pytest.MonkeyPatch) -> unittest.mock.MagicMock:
    """Fixture that mocks urllib.request.urlopen calls."""
    mock_resp = unittest.mock.MagicMock()
    monkeypatch.setattr("urllib.request.urlopen", mock_resp)
    return mock_resp


@pytest.fixture()
def mock_path_write_bytes(monkeypatch: pytest.MonkeyPatch) -> unittest.mock.MagicMock:
    """Fixture that mocks the write_bytes method of pathlib.Path."""
    mock_write_bytes = unittest.mock.MagicMock()
    monkeypatch.setattr(Path, "write_bytes", mock_write_bytes)
    return mock_write_bytes


mock_file_content = b"file_contents"
actual_sha1sum = hashlib.sha1(mock_file_content).hexdigest()  # noqa: S324[256 np]


def test_download_loci_jar_valid_checksum(
    mock_urlopen: unittest.mock.MagicMock,
    mock_path_write_bytes: unittest.mock.MagicMock,
) -> None:
    """Test that the loci_tools file is written to disk with a valid checksum."""
    # Mock responses for the JAR file and checksum validation
    mock_urlopen.return_value.read.side_effect = [
        mock_file_content,
        actual_sha1sum.encode() + b" ",
    ]
    # Perform the test
    ir.download_loci_jar()
    # Assert that write_bytes was called once with the mocked file content
    mock_path_write_bytes.assert_called_once_with(mock_file_content)


def test_download_loci_jar_invalid_checksum(
    mock_urlopen: unittest.mock.MagicMock,
    mock_path_write_bytes: unittest.mock.MagicMock,
) -> None:
    """Test that an error is raised and no file is written with an invalid checksum."""
    # Mock responses with valid JAR content and invalid checksum
    mock_urlopen.return_value.read.side_effect = [b"file_contents", b"invalid_sha1sum "]
    expected_error_msg = (
        "Downloaded loci_tools.jar has an invalid checksum. Please try again."
    )
    with pytest.raises(OSError, match=expected_error_msg) as excinfo:
        ir.download_loci_jar()
    assert expected_error_msg in str(excinfo.value)
    assert not mock_path_write_bytes.called


@pytest.fixture()
def ome_store() -> ir.OMEPyramidStore:
    """Fixture for OME Store."""
    md, wr = ir.read(str(tpath / "tile6_1.tif"))
    return wr.rdr.getMetadataStore()


@pytest.fixture()
def ome_store_lif() -> ir.OMEPyramidStore:
    """Fixture for OME Store."""
    md, wr = ir.read(str(tpath / "2015Aug28_TransHXB2_50min+DMSO.lif"))
    return wr.rdr.getMetadataStore()


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
    data: list[
        tuple[int, int, int, int, int, int, int | float]
    ]  # S, X, Y, C, T, Z, value


# data: series, x, y, channel, time, z, value

# "lif" - LIF_multiseries
data_lif: list[tuple[int, int, int, int, int, int, int | float]] = [
    (4, 256, 128, 2, 0, 21, 2),
    (4, 285, 65, 2, 0, 21, 16),
    (4, 285, 65, 0, 0, 21, 14),
]  # max = 255
fp = "2015Aug28_TransHXB2_50min+DMSO.lif"
td_lif = TDataItem(
    fp, 5, [512], [512], [3], [1], [41, 40, 43, 39, 37], [0.080245], data_lif
)

# "img_tile" - FEI_tiled -  # C=3 T=4 S=15
data_tile: list[tuple[int, int, int, int, int, int, int | float]] = [
    (14, 509, 231, 0, 2, 0, 14580),
    (14, 509, 231, 1, 2, 0, 8436),
    (14, 509, 231, 2, 2, 0, 8948),
    (14, 509, 231, 3, 2, 0, 8041),
    (7, 194, 192, 1, 0, 0, 3783),
    (7, 194, 192, 1, 1, 0, 3585),
    (7, 194, 192, 1, 2, 0, 3403),
]
td_img_tile = TDataItem(
    "t4_1.tif", 15, [512], [256], [4], [3], [1], [0.133333], data_tile
)

# "img_void_tile" -  -  # C=4 T=3 S=14 scattered
td_img_void_tile = TDataItem("tile6_1.tif", 14, [512], [512], [3], [4], [1], [0.2], [])

# "imgsingle" - FEI_multichannel -  # C=2 T=81
data: list[tuple[int, int, int, int, int, int, int | float]] = [
    (0, 610, 520, 0, 80, 0, 142),  # max = 212
    (0, 610, 520, 1, 80, 0, 132),  # max = 184
]
td_imgsingle = TDataItem("exp2_2.tif", 1, [1600], [1200], [2], [81], [1], [0.74], data)

# "mcts" - ome_multichannel -  # C=3 T=7
data_: list[tuple[int, int, int, int, int, int, int | float]] = [
    (0, 200, 20, 2, 6, 0, -1)
]
td_mcts = TDataItem(
    "multi-channel-time-series.ome.tif", 1, [439], [167], [3], [7], [1], [None], data_
)

# "bigtiff" - bigtiff - tdata / "LC26GFP_1.tf8"

list_test_data = [td_img_tile, td_img_void_tile, td_imgsingle, td_lif, td_mcts]
# Used to be ["FEI multichannel", "FEI multitiles", "OME std test", "Leica LIF"]
ids = ["img_tile", "img_void_tile", "imgsingle", "lif", "mcts"]


@pytest.fixture(params=[ir.read, ir.read_jpype, ir.read_pims])
def read_functions(
    request: pytest.FixtureRequest,
) -> Callable[[str], tuple[ir.Metadata, ir.ImageReaderWrapper]]:
    """Fixture to parametrize different image reading functions."""
    return cast(
        Callable[[str], tuple[ir.Metadata, ir.ImageReaderWrapper]], request.param
    )


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
    """Fixture for test data and reader with various data and reading functions."""
    return common_tdata(request, read_functions)


@pytest.fixture(params=[td_img_tile], ids=[ids[0]])
def tdata_img_tile(
    request: pytest.FixtureRequest, read_functions: Callable[[str], Any]
) -> tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]:
    """Fixture for test data and reader of a tiled image with various functions."""
    return common_tdata(request, read_functions)


@pytest.fixture(params=[td_img_void_tile], ids=[ids[1]])
def tdata_img_void_tile(
    request: pytest.FixtureRequest, read_functions: Callable[[str], Any]
) -> tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]:
    """Fixture for test data and reader of a tiled image with various functions."""
    return common_tdata(request, read_functions)


def test_file_not_found() -> None:
    """It raises the expected exception when attempting to read a non-existent file."""
    with pytest.raises(FileNotFoundError) as excinfo:
        ir.read(str(tpath / "pippo.tif"))
    expected_error_message = f'File not found: {tpath / "pippo.tif"}'
    assert expected_error_message in str(excinfo.value)


def test_metadata_data(
    tdata_all: tuple[TDataItem, ir.Metadata, ir.ImageReaderWrapper]
) -> None:
    """Test metadata and data consistency."""
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
    """Test tile stitching for a specific image with tiles."""
    td, md3, wrapper = tdata_img_tile
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
    """Test void tile stitching for a specific image with void tiles."""
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
    """Test the first non-zero index in a list in reverse order."""
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
    """Test the function to generate the next tuple."""
    assert ir.next_tuple([1], increment_last=True) == [2]
    assert ir.next_tuple([1, 1], increment_last=False) == [2, 0]
    assert ir.next_tuple([0, 0, 0], increment_last=True) == [0, 0, 1]
    assert ir.next_tuple([0, 0, 1], increment_last=True) == [0, 0, 2]
    assert ir.next_tuple([0, 0, 2], increment_last=False) == [0, 1, 0]
    assert ir.next_tuple([0, 1, 0], increment_last=True) == [0, 1, 1]
    assert ir.next_tuple([0, 1, 1], increment_last=True) == [0, 1, 2]
    assert ir.next_tuple([0, 1, 2], increment_last=False) == [0, 2, 0]
    assert ir.next_tuple([0, 2, 0], increment_last=False) == [1, 0, 0]
    assert ir.next_tuple([1, 0, 0], increment_last=True) == [1, 0, 1]
    assert ir.next_tuple([1, 1, 1], increment_last=False) == [1, 2, 0]
    assert ir.next_tuple([1, 2, 0], increment_last=False) == [2, 0, 0]
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([2, 0, 0], increment_last=False)
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([1, 0], increment_last=False)
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([1], increment_last=False)
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([], increment_last=False)
    with pytest.raises(ir.StopExceptionError):
        ir.next_tuple([], increment_last=True)


def test_convert_value(ome_store: ir.OMEPyramidStore) -> None:
    """Test convert_value function with various input types."""
    # float with units
    assert ir.convert_value(ome_store.getArcPower(0, 0)) == (150.0, "mW")
    # str
    assert (
        ir.convert_value(ome_store.getChannelIlluminationType(13, 2))
        == "Epifluorescence"
    )
    # int
    assert ir.convert_value(ome_store.getLightSourceCount(0)) == 9
    # float
    assert (
        ir.convert_value(ome_store.getChannelLightSourceSettingsAttenuation(13, 2))
        == 0.9
    )


# Common expected values for Metadata.
EXPECTED_VALUES = {
    "getChannelLightSourceSettingsID": (
        2,
        [
            ((0, 0), "LightSource:0:2"),
            ((0, 1), "LightSource:0:4"),
            ((1, 0), "LightSource:1:2"),
            ((1, 1), "LightSource:1:4"),
            ((2, 0), "LightSource:2:2"),
            ((2, 1), "LightSource:2:4"),
            ((3, 0), "LightSource:3:2"),
            ((3, 1), "LightSource:3:4"),
            ((4, 0), "LightSource:4:2"),
            ((4, 1), "LightSource:4:4"),
        ],
    ),
    "getImageCount": (0, [((), 5)]),
    "getPlanePositionZ": (2, [((4, 110), (0.0, "reference frame"))]),
    "getPlanePositionX": (
        2,
        [
            ((0, 122), (0.0434916298968, "reference frame")),
            ((1, 119), (0.0434572221804, "reference frame")),
            ((2, 128), (0.04336546827, "reference frame")),
            ((3, 116), (0.0438492490536, "reference frame")),
            ((4, 110), (0.0439142644374, "reference frame")),
        ],
    ),
    "getPixelsPhysicalSizeY": (1, [((4,), (0.0802453424657534, "µm"))]),
    "getLightPathExcitationFilterRef": (3, None),
    "getExperimentType": (1, None),
    "getChannelExcitationWavelength": (
        2,
        [((4, 0), (488.0, "nm")), ((4, 1), (543.0, "nm")), ((4, 2), None)],
    ),
}


@pytest.mark.parametrize(
    ("key", "expected"),
    EXPECTED_VALUES.items(),
    ids=EXPECTED_VALUES.keys(),
)
def test_get_allvalues_grouped(
    ome_store_lif: ir.OMEPyramidStore,
    key: str,
    expected: tuple[int, ir.FullMDValueType],
) -> None:
    """Test the function to retrieve and group metadata values for a given key."""
    # Test raising exceptions for certain keys
    if key in ["getLightPathExcitationFilterRef", "getExperimentType"]:
        with pytest.raises(Exception, match="java"):
            ir.get_allvalues_grouped(ome_store_lif, key, expected[0])
    else:
        assert ir.get_allvalues_grouped(ome_store_lif, key, expected[0]) == expected[1]


@pytest.mark.parametrize(
    ("key", "expected"),
    EXPECTED_VALUES.items(),
    ids=EXPECTED_VALUES.keys(),
)
def test_full(
    ome_store_lif: ir.OMEPyramidStore,
    key: str,
    expected: tuple[int, ir.FullMDValueType],
) -> None:
    """Generate full and log from ome_store."""
    full, log_miss = ir.get_md_dict(ome_store_lif, Path("llog"))
    if expected[1]:
        assert full[key.replace("get", "")] == expected[1]
        assert log_miss[key] == "Found"
    else:
        assert log_miss[key] == "Jmiss"
