#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bioformats
import javabridge
import pytest
import os
import subprocess
import sys

import imgread.read as ir

__author__ = "daniele arosio"
__copyright__ = "daniele arosio"
__license__ = "new-bsd"

datafiles_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data')

img_FEI_multichannel = os.path.join(datafiles_folder, "exp2_2.tif")
img_FEI_tiled = os.path.join(datafiles_folder, "t4_1.tif")
img_FEI_void_tiled = os.path.join(datafiles_folder, "tile6_1.tif")
img_LIF_multiseries = os.path.join(datafiles_folder,
                                   "2015Aug28_TransHXB2_50min+DMSO.lif")
img_ome_multichannel = os.path.join(datafiles_folder,
                                    "multi-channel-time-series.ome.tif")

IN_MD_DD = [
    # img_file, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX
    (
        img_FEI_multichannel,
        1,
        1600,
        1200,
        2,
        81,
        1,
        0.74,
        # series, X, Y, C, time, Z, value; and must be list of list
        # (1 file, 1 md and 1* data).
        [
            [0, 610, 520, 0, 80, 0, 142],  # max = 212
            [0, 610, 520, 1, 80, 0, 132]
        ]  # max = 184
    ),
    (img_FEI_tiled, 15, 512, 256, 4, 3, 1, 0.133333,
     [[14, 509, 231, 0, 2, 0, 14580], [14, 509, 231, 1, 2, 0, 8436],
      [14, 509, 231, 2, 2, 0, 8948], [14, 509, 231, 3, 2, 0, 8041],
      [7, 194, 192, 1, 0, 0, 3783], [7, 194, 192, 1, 1, 0,
                                     3585], [7, 194, 192, 1, 2, 0, 3403]]),
    (img_ome_multichannel, 1, 439, 167, 3, 7, 1, None, []),
    (
        img_LIF_multiseries,
        5,
        512,
        512,
        3,
        1,
        [41, 40, 43, 39, 37],
        0.080245,
        [[4, 256, 128, 2, 0, 21, 2], [4, 285, 65, 2, 0, 21, 16],
         [4, 285, 65, 0, 0, 21, 14]]  # max = 255
    ),
]


def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)


def teardown_module(module):
    """teardown any state that was previously setup with a setup_module method.

    """
    javabridge.kill_vm()


def test_exception():
    with pytest.raises(Exception):
        ir.read(os.path.join(datafiles_folder, "pippo.tif"))


def check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX):
    assert md['SizeS'] == SizeS
    assert md['SizeX'] == SizeX
    assert md['SizeY'] == SizeY
    assert md['SizeC'] == SizeC
    assert md['SizeT'] == SizeT
    if 'SizeZ' in md:
        assert md['SizeZ'] == SizeZ
    else:
        for i, v in enumerate(SizeZ):  # for lif file
            assert md['series'][i]['SizeZ'] == v
    assert md['PhysicalSizeX'] == PhysicalSizeX


# metadata first


# using external showinf (metadata only)
@pytest.mark.parametrize(
    'filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data',
    IN_MD_DD)
def test_metadata_showinf(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ,
                          PhysicalSizeX, data):
    md = ir.read_inf(filepath)
    check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX)


# BF standard from manual (metadata only)
@pytest.mark.parametrize(
    'filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data',
    IN_MD_DD[:3])
def test_metadata_bf(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ,
                     PhysicalSizeX, data):
    md = ir.read_bf(filepath)
    # assert md['SizeS'] == SizeS
    assert md['SizeX'] == SizeX
    assert md['SizeY'] == SizeY
    # assert md['SizeC'] == SizeC  # even the std multichannel OME file fails
    # assert md['SizeT'] == SizeT
    assert md['SizeZ'] == SizeZ
    # assert md['PhysicalSizeX'] == PhysicalSizeX
    # NOT Working well with FEI OME-TIFF


@pytest.mark.parametrize(
    'filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data',
    IN_MD_DD[3:])
def test_metadata_bf2(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ,
                      PhysicalSizeX, data):
    md = ir.read_bf(filepath)
    check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX)


# forcing reader check and using java of (OMETiffReader only) (metadata only)
@pytest.mark.parametrize(
    'filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data',
    IN_MD_DD[:3])
def test_metadata_javabridge(filepath, SizeS, SizeX, SizeY, SizeC, SizeT,
                             SizeZ, PhysicalSizeX, data):
    md = ir.read_jb(filepath)
    check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX)


@pytest.mark.parametrize(
    'filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data',
    IN_MD_DD)
def test_metadata_data(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ,
                       PhysicalSizeX, data):
    md, wrapper = ir.read(filepath)
    check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX)
    if len(data) > 0:
        for l in data:
            series = l[0]
            X = l[1]
            Y = l[2]
            channel = l[3]
            time = l[4]
            Z = l[5]
            value = l[6]
            a = wrapper.read(
                c=channel, t=time, series=series, z=Z, rescale=False)
            # Y then X
            assert a[Y, X] == value


def test_tile_stitch():
    md, wrapper = ir.read(img_FEI_tiled)
    stitched_plane = ir.stitch(md, wrapper)
    # Y then X
    assert stitched_plane[861, 1224] == 7779
    assert stitched_plane[1222, 1416] == 9626
    stitched_plane = ir.stitch(md, wrapper, t=2, c=3)
    assert stitched_plane[1236, 1488] == 6294
    stitched_plane = ir.stitch(md, wrapper, t=1, c=2)
    assert stitched_plane[564, 1044] == 8560


def test_void_tile_stitch():
    md, wrapper = ir.read(img_FEI_void_tiled)
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


#     use capsys and capfd
# https://docs.pytest.org/en/2.8.7/capture.html
class Test_imgdiff:
    def setup_class(self):
        self.fp_a = os.path.join(datafiles_folder, 'im1s1z3c5t_a.ome.tif')
        self.fp_b = os.path.join(datafiles_folder, 'im1s1z3c5t_b.ome.tif')
        self.fp_bmd = os.path.join(datafiles_folder, 'im1s1z2c5t_bmd.ome.tif')
        self.fp_bpix = os.path.join(datafiles_folder,
                                    'im1s1z3c5t_bpix.ome.tif')

    def test_diff(self):
        assert ir.diff(self.fp_a, self.fp_b)
        assert not ir.diff(self.fp_a, self.fp_bmd)
        assert not ir.diff(self.fp_a, self.fp_bpix)

    def test_script(self):
        cmd_line = ['imgdiff', self.fp_a, self.fp_b]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
        assert p.communicate()[0] == b"Files seem equal.\n"
        cmd_line = ['imgdiff', self.fp_a, self.fp_bmd]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
        assert p.communicate()[0] == b"Files differ.\n"
        cmd_line = ['imgdiff', self.fp_a, self.fp_bpix]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
        assert p.communicate()[0] == b"Files differ.\n"


@pytest.mark.skip
def test_read_wrap(capsys):
    print(";pippo")
    fp_a = os.path.join(datafiles_folder, 'im1s1z3c5t_a.ome.tif')
    with capsys.disabled():
        md, wr = ir.read_wrap(fp_a)
    assert True


def test_first_nonzero_reverse():
    assert ir.first_nonzero_reverse([0, 0, 2, 0]) == -2
    assert ir.first_nonzero_reverse([0, 2, 1, 0]) == -2
    assert ir.first_nonzero_reverse([1, 2, 1, 0]) == -2
    assert ir.first_nonzero_reverse([2, 0, 0, 0]) == -4


def test__convert_num(capsys):
    """Test num convertions and raise with printout."""
    assert ir._convert_num(None) is None
    assert ir._convert_num('0.976') == 0.976
    assert ir._convert_num(0.976) == 0.976
    assert ir._convert_num(976) == 976
    assert ir._convert_num('976') == 976
    with pytest.raises(ValueError):
        ir._convert_num('b976')
    out, err = capsys.readouterr()
    sys.stdout.write(out)
    sys.stderr.write(err)
    assert out.startswith("Neither ")


def test_next_tuple():
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


@pytest.mark.skip
def test_convert_value():
    """Test convertion from java metadata value."""
    pass


@pytest.mark.parametrize(
    'filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data',
    IN_MD_DD)
def test_metadata2_data(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ,
                        PhysicalSizeX, data):
    md2, wrapper = ir.read2(filepath)

    md = {
        'SizeS': md2['ImageCount'][0][1],
        'SizeX': md2['PixelsSizeX'][0][1],
        'SizeY': md2['PixelsSizeY'][0][1],
        'SizeC': md2['PixelsSizeC'][0][1],
        'SizeT': md2['PixelsSizeT'][0][1]
    }
    if len(md2['PixelsSizeZ']) == 1:
        md['SizeZ'] = md2['PixelsSizeZ'][0][1]
    elif len(md2['PixelsSizeZ']) > 1:
        md['series'] = [{'SizeZ': l[1]} for l in md2['PixelsSizeZ']]
    if 'PixelsPhysicalSizeX' in md2:
        # this is with unit
        md['PhysicalSizeX'] = round(md2['PixelsPhysicalSizeX'][0][1][0], 6)
    else:
        md['PhysicalSizeX'] = None

    check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX)
    if len(data) > 0:
        for l in data:
            series = l[0]
            X = l[1]
            Y = l[2]
            channel = l[3]
            time = l[4]
            Z = l[5]
            value = l[6]
            a = wrapper.read(
                c=channel, t=time, series=series, z=Z, rescale=False)
            # Y then X
            assert a[Y, X] == value
