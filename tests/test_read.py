#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
# from imgread.skeleton import fib
import imgread.read
import javabridge
import bioformats

__author__ = "daniele arosio"
__copyright__ = "daniele arosio"
__license__ = "new-bsd"

img_FEI_multichannel = "tests/data/exp2_2.tif"
img_FEI_tiled = "tests/data/t4_1.tif"
img_FEI_void_tiled = "tests/data/tile6_1.tif"
img_LIF_multiseries = "tests/data/2015Aug28_TransHXB2_50min+DMSO.lif"
img_ome_multichannel = "tests/data/multi-channel-time-series.ome.tif"


IN_MD_DD = [
     # img_file, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX
    (img_FEI_multichannel, 1, 1600, 1200, 2, 81, 1, 0.74,
     # series, X, Y, C, time, Z, value; and must be list of list (1 file, 1 md and 1* data).
     [[0, 610, 520, 0, 80, 0, 142], # max = 212
      [0, 610, 520, 1, 80, 0, 132]] # max = 184
    ),
    (img_FEI_tiled, 15, 512, 256, 4, 3, 1, 0.133333,
     [[14, 509, 231, 0, 2, 0, 14580],
      [14, 509, 231, 1, 2, 0, 8436],
      [14, 509, 231, 2, 2, 0, 8948],
      [14, 509, 231, 3, 2, 0, 8041],
      [7, 194, 192, 1, 0, 0, 3783],
      [7, 194, 192, 1, 1, 0, 3585],
      [7, 194, 192, 1, 2, 0, 3403]]
    ),
    (img_ome_multichannel, 1, 439, 167, 3, 7, 1, None, []
    ),
    (img_LIF_multiseries, 5, 512, 512, 3, 1, [41, 40, 43, 39, 37], 0.080245,
     [[4, 256, 128, 2, 0, 21, 2],
      [4, 285, 65, 2, 0, 21, 16],
      [4, 285, 65, 0, 0, 21, 14]] # max = 255
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
        imgread.read.read('tests/data/pippo.tif')


def check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX):
    assert md['SizeS'] == SizeS
    assert md['SizeX'] == SizeX
    assert md['SizeY'] == SizeY
    assert md['SizeC'] == SizeC
    assert md['SizeT'] == SizeT
    if 'SizeZ' in md:
        assert md['SizeZ'] == SizeZ
    else:
        for i, v in enumerate(SizeZ):
            assert md['series'][i]['SizeZ'] == v
    assert md['PhysicalSizeX'] == PhysicalSizeX
# metadata first

# using external showinf (metadata only)
@pytest.mark.parametrize('filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data', IN_MD_DD)
def test_metadata_showinf(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data):
    md = imgread.read.read_inf(filepath)
    check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX)


# BF standard from manual (metadata only)
@pytest.mark.parametrize('filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data', IN_MD_DD[:3])
def test_metadata_bf(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data):
    md = imgread.read.read_bf(filepath)
    # assert md['SizeS'] == SizeS
    assert md['SizeX'] == SizeX
    assert md['SizeY'] == SizeY
    # assert md['SizeC'] == SizeC  # also the simple std ome multichannel file fails here
    # assert md['SizeT'] == SizeT
    assert md['SizeZ'] == SizeZ
    # assert md['PhysicalSizeX'] == PhysicalSizeX
    # NOT Working well with FEI OME-TIFF

@pytest.mark.parametrize('filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data', IN_MD_DD[3:])
def test_metadata_bf2(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data):
    md = imgread.read.read_bf(filepath)
    check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX)


# forcing reader check and using java of (OMETiffReader only) (metadata only)
@pytest.mark.parametrize('filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data', IN_MD_DD[:3])
def test_metadata_javabridge(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data):
    md = imgread.read.read_jb(filepath)
    check_md(md, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX)


# @pytest.mark.parametrize('filepath, series, X, Y, channel, time, Z, value, data', IN_MD_DD)
# def test_metadata_data(filepath, series, X, Y, channel, time, Z, value):
@pytest.mark.parametrize('filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data', IN_MD_DD)
def test_metadata_data(filepath, SizeS, SizeX, SizeY, SizeC, SizeT, SizeZ, PhysicalSizeX, data):
    md, wrapper = imgread.read.read(filepath)
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
            a = wrapper.read(c=channel, t=time, series=series, z=Z, rescale=False)
            # Y then X
            assert a[Y, X] == value



def test_tile_stitch():
    md, wrapper = imgread.read.read(img_FEI_tiled)
    stitched_plane = imgread.read.stitch(md, wrapper)
    # Y then X
    assert stitched_plane[861, 1224] == 7779
    assert stitched_plane[1222, 1416] == 9626
    stitched_plane = imgread.read.stitch(md, wrapper, t=2, c=3)
    assert stitched_plane[1236, 1488] == 6294
    stitched_plane = imgread.read.stitch(md, wrapper, t=1, c=2)
    assert stitched_plane[564, 1044] == 8560


def test_void_tile_stitch():
    md, wrapper = imgread.read.read(img_FEI_void_tiled)
    stitched_plane = imgread.read.stitch(md, wrapper, t=0, c=0)
    assert stitched_plane[1179, 882] == 6395
    stitched_plane = imgread.read.stitch(md, wrapper, t=0, c=1)
    assert stitched_plane[1179, 882] == 3386
    stitched_plane = imgread.read.stitch(md, wrapper, t=0, c=2)
    assert stitched_plane[1179, 882] == 1690
    stitched_plane = imgread.read.stitch(md, wrapper, t=1, c=0)
    assert stitched_plane[1179, 882] == 6253
    stitched_plane = imgread.read.stitch(md, wrapper, t=1, c=1)
    assert stitched_plane[1179, 882] == 3499
    stitched_plane = imgread.read.stitch(md, wrapper, t=1, c=2)
    assert stitched_plane[1179, 882] == 1761
    stitched_plane = imgread.read.stitch(md, wrapper, t=2, c=0)
    assert stitched_plane[1179, 882] == 6323
    stitched_plane = imgread.read.stitch(md, wrapper, t=2, c=1)
    assert stitched_plane[1179, 882] == 3354
    stitched_plane = imgread.read.stitch(md, wrapper, t=2, c=2)
    assert stitched_plane[1179, 882] == 1674
    stitched_plane = imgread.read.stitch(md, wrapper, t=3, c=0)
    assert stitched_plane[1179, 882] == 6291
    stitched_plane = imgread.read.stitch(md, wrapper, t=3, c=1)
    assert stitched_plane[1179, 882] == 3373
    stitched_plane = imgread.read.stitch(md, wrapper, t=3, c=2)
    assert stitched_plane[1179, 882] == 1615
    stitched_plane = imgread.read.stitch(md, wrapper, t=3, c=0)
    assert stitched_plane[1213, 1538] == 704
    stitched_plane = imgread.read.stitch(md, wrapper, t=3, c=1)
    assert stitched_plane[1213, 1538] == 422
    stitched_plane = imgread.read.stitch(md, wrapper, t=3, c=2)
    assert stitched_plane[1213, 1538] == 346
    # Void tiles are set to 0
    assert stitched_plane[2400, 2400] == 0
    assert stitched_plane[2400, 200] == 0
