"""Testing module.
It compares:
- showinf
- bioformats
- javabridge access to java classes
- OMEXMLMetadataImpl into image_reader
- [ ] pims
- [ ] jpype
Tests:
- FEI multichannel
- FEI tiled
- OME std multichannel
- LIF
It also tests FEI tiled with a void tile.

"""
import os
import sys

import javabridge  # type: ignore
import nima_io.read as ir  # type: ignore
import pytest


def check_core_md(md, test_md_data_dict):
    """Helper function to compare (read vs. expected) core metadata.

    :param (dict) md: read metadata
    :param (dict) test_md_data_dict: metadata specified in the input data

    :raise: AssertionError
    """
    assert md["SizeS"] == test_md_data_dict["SizeS"]
    assert md["SizeX"] == test_md_data_dict["SizeX"]
    assert md["SizeY"] == test_md_data_dict["SizeY"]
    assert md["SizeC"] == test_md_data_dict["SizeC"]
    assert md["SizeT"] == test_md_data_dict["SizeT"]
    if "SizeZ" in md:
        assert md["SizeZ"] == test_md_data_dict["SizeZ"]
    else:
        for i, v in enumerate(test_md_data_dict["SizeZ"]):  # for LIF file
            assert md["series"][i]["SizeZ"] == v
    assert md["PhysicalSizeX"] == test_md_data_dict["PhysicalSizeX"]


def check_single_md(md, test_md_data_dict, key):
    """Helper function to compare (read vs. expected) single :key: core metadata.

    :param (dict) md: read metadata
    :param (dict) test_md_data_dict: metadata specified in the input data

    :raise: AssertionError
    """
    if key in md:
        assert md[key] == test_md_data_dict[key]
    else:
        for i, v in enumerate(test_md_data_dict[key]):  # e.g. SizeZ in LIF
            assert md["series"][i][key] == v


def check_data(wrapper, data):
    """data is a list of list.... TODO: complete"""
    if len(data) > 0:
        for l in data:
            series = l[0]
            X = l[1]
            Y = l[2]
            channel = l[3]
            time = l[4]
            Z = l[5]
            value = l[6]
            a = wrapper.read(c=channel, t=time, series=series, z=Z, rescale=False)
            # Y then X
            assert a[Y, X] == value


@pytest.mark.skip("to be completed using capsys")
def test_exception() -> None:
    with pytest.raises(Exception):
        ir.read(os.path.join("datafolder", "pippo.tif"))


@pytest.mark.slow
class Test_showinf:
    """Test only metadata retrieve using the shell cmd showinf."""

    def setup_class(cls):
        cls.read = ir.read_inf

    def test_md(self, read_all):
        test_md, md, wr = read_all
        check_core_md(md, test_md)


class TestBioformats:
    """Test metadata retrieve using standard bioformats approach.
    Core metadata seems retrieved correctly only for LIF files.

    """

    reason = "bioformats OMEXML known failure"

    def setup_class(cls):
        cls.read = ir.read_bf
        print("Starting VirtualMachine")
        ir.ensure_VM()

    # @pytest.mark.xfail(
    #     raises=AssertionError, reason="Wrong SizeC,T,PhysicalSizeX")
    @pytest.mark.parametrize(
        "key",
        [
            "SizeS",
            "SizeX",
            "SizeY",
            pytest.param(
                "SizeC", marks=pytest.mark.xfail(raises=AssertionError, reason=reason)
            ),
            pytest.param(
                "SizeT", marks=pytest.mark.xfail(raises=AssertionError, reason=reason)
            ),
            "SizeZ",
            pytest.param(
                "PhysicalSizeX",
                marks=pytest.mark.xfail(raises=AssertionError, reason=reason),
            ),
        ],
    )
    def test_FEI_multichannel(self, read_FEI_multichannel, key):
        md = read_FEI_multichannel[1]
        check_single_md(md, read_FEI_multichannel[0], key)

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param(
                "SizeS", marks=pytest.mark.xfail(raises=AssertionError, reason=reason)
            ),
            "SizeX",
            "SizeY",
            pytest.param(
                "SizeC", marks=pytest.mark.xfail(raises=AssertionError, reason=reason)
            ),
            pytest.param(
                "SizeT", marks=pytest.mark.xfail(raises=AssertionError, reason=reason)
            ),
            "SizeZ",
            pytest.param(
                "PhysicalSizeX",
                marks=pytest.mark.xfail(raises=AssertionError, reason=reason),
            ),
        ],
    )
    def test_FEI_multitile(self, read_FEI_multitile, key):
        md = read_FEI_multitile[1]
        check_single_md(md, read_FEI_multitile[0], key)

    @pytest.mark.parametrize(
        "key",
        [
            "SizeS",
            "SizeX",
            "SizeY",
            pytest.param(
                "SizeC", marks=pytest.mark.xfail(raises=AssertionError, reason=reason)
            ),
            pytest.param(
                "SizeT", marks=pytest.mark.xfail(raises=AssertionError, reason=reason)
            ),
            "SizeZ",
            "PhysicalSizeX",
        ],
    )
    def test_OME_multichannel(self, read_OME_multichannel, key):
        md = read_OME_multichannel[1]
        check_single_md(md, read_OME_multichannel[0], key)

    @pytest.mark.parametrize(
        "key", ["SizeS", "SizeX", "SizeY", "SizeC", "SizeT", "SizeZ", "PhysicalSizeX"]
    )
    def test_LIF(self, read_LIF, key):
        md = read_LIF[1]
        # check_core_md(md, read_LIF[0])
        check_single_md(md, read_LIF[0], key)


class TestJavabridge:
    """Test only metadata retrieve forcing reader check and using OMETiffReader
    class directly thanks to javabridge.

    """

    def setup_class(cls):
        cls.read = ir.read_jb
        print("Starting VirtualMachine")
        ir.ensure_VM()

    def test_TIF_only(self, read_TIF):
        test_md, md, wr = read_TIF
        check_core_md(md, test_md)


class TestMdData:
    """Test both metadata and data with all files, OME and LIF, using
    javabridge OMEXmlMetadata into bioformats image reader.

    """

    def setup_class(cls):
        cls.read = ir.read
        print("Starting VirtualMachine")
        ir.ensure_VM()

    def test_metadata_data(self, read_all):
        test_d, md, wrapper = read_all
        check_core_md(md, test_d)
        check_data(wrapper, test_d["data"])

    def test_tile_stitch(self, read_all):
        if read_all[0]["filename"] == "t4_1.tif":
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

    def test_void_tile_stitch(self, read_void_tile):
        # ir.ensure_VM()
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
    """Test num convertions and raise with printout."""
    assert ir._convert_num(None) is None
    assert ir._convert_num("0.976") == 0.976
    assert ir._convert_num(0.976) == 0.976
    assert ir._convert_num(976) == 976
    assert ir._convert_num("976") == 976


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
    with pytest.raises(ir.stopException):
        ir.next_tuple([2, 0, 0], False)
    with pytest.raises(ir.stopException):
        ir.next_tuple([1, 0], False)
    with pytest.raises(ir.stopException):
        ir.next_tuple([1], False)
    with pytest.raises(ir.stopException):
        ir.next_tuple([], False)
    with pytest.raises(ir.stopException):
        ir.next_tuple([], True)


def test_get_allvalues_grouped():
    # k = 'getLightPathExcitationFilterRef' # npar = 3 can be more tidied up
    # #k = 'getChannelLightSourceSettingsID' # npar = 2
    # #k = 'getPixelsSizeX' # npar = 1
    # #k = 'getExperimentType'
    # #k = 'getImageCount' # npar = 0
    # k = 'getPlanePositionZ'

    # get_allvalues(metadata, k, 2)
    pass


class TestMetadata2:
    def setup_class(cls):
        cls.read = ir.read2
        print("Starting VirtualMachine")
        ir.ensure_VM()

    def teardown_class(cls):
        print("Better not Killing VirtualMachine")
        # javabridge.kill_vm()

    # def test_convert_value(self, filepath, SizeS, SizeX, SizeY, SizeC, SizeT,
    #                        SizeZ, PhysicalSizeX, data):
    #     """Test convertion from java metadata value."""
    #     print(filepath)

    def test_metadata_data2(self, read_all):
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
            md["series"] = [{"SizeZ": l[1]} for l in md2["PixelsSizeZ"]]
        if "PixelsPhysicalSizeX" in md2:
            # this is with unit
            md["PhysicalSizeX"] = round(md2["PixelsPhysicalSizeX"][0][1][0], 6)
        else:
            md["PhysicalSizeX"] = None
        check_core_md(md, test_d)
        check_data(wrapper, test_d["data"])


def teardown_module():
    javabridge.kill_vm()
