"""Tests for the `read.py` module.

This module tests the core reading functionality:
- ``Metadata`` - structured metadata dataclass
- ``read_image`` - single-scene reader (bioio backend)
- ``stitch_scenes`` - multi-scene tile stitching
- ``diff`` - image comparison

"""

from pathlib import Path

import pytest

import nima_io.read as ir
from nima_io.read import Channel, Metadata, StagePosition, VoxelSize

from .conftest import require_test_data

tpath = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Tests for the Metadata dataclass and its components."""

    def test_metadata_attrs(self) -> None:
        """read_image attaches Metadata in attrs."""
        fp = tpath / "im1s1z3c5t_a.ome.tif"
        require_test_data([fp.name])
        da = ir.read_image(str(fp))
        assert "metadata" in da.attrs
        md = da.attrs["metadata"]
        assert isinstance(md, Metadata)
        assert md.size_s == 1
        assert md.size_c == [3]
        assert md.size_t == [5]

    def test_channel_fields(self) -> None:
        """Channel objects carry wavelength, attenuation, exposure, gain."""
        fp = tpath / "im1s1z3c5t_a.ome.tif"
        require_test_data([fp.name])
        md = ir.read_image(str(fp)).attrs["metadata"]
        ch = md.channels[0][0]
        assert isinstance(ch, Channel)
        assert ch.wavelength > 0
        assert ch.exposure > 0
        assert ch.binning == "1x1"

    def test_voxel_size(self) -> None:
        """VoxelSize is populated from OME pixel sizes."""
        fp = tpath / "im1s1z3c5t_a.ome.tif"
        require_test_data([fp.name])
        md = ir.read_image(str(fp)).attrs["metadata"]
        vs = md.voxel_size[0]
        assert isinstance(vs, VoxelSize)
        assert vs.x is not None
        assert vs.x > 0

    def test_stitch_metadata_dedup(self) -> None:
        """Stitch deduplicates identical per-scene values."""
        fp = tpath / "t4_1.tif"
        require_test_data([fp.name])
        da = ir.stitch_scenes(str(fp))
        md = da.attrs["metadata"]
        assert isinstance(md, Metadata)
        assert md.size_s == 15
        # Channels should be deduplicated (all tiles share same channels)
        assert len(md.channels) == 1
        assert len(md.voxel_size) == 1

    def test_stage_position(self) -> None:
        """Stage positions are parsed from OME planes."""
        fp = tpath / "im1s1z3c5t_a.ome.tif"
        require_test_data([fp.name])
        md = ir.read_image(str(fp)).attrs["metadata"]
        sp = md.stage_position[0]
        assert isinstance(sp, dict)
        pos = next(iter(sp))
        assert isinstance(pos, StagePosition)
        assert pos.x is not None

    def test_tcz_deltat(self) -> None:
        """tcz_deltat lists (T, C, Z, delta_t) per plane."""
        fp = tpath / "im1s1z3c5t_a.ome.tif"
        require_test_data([fp.name])
        md = ir.read_image(str(fp)).attrs["metadata"]
        deltas = md.tcz_deltat[0]
        assert len(deltas) > 0
        t, c, z, dt = deltas[0]
        assert t == 0
        assert c == 0
        assert z == 0
        assert isinstance(dt, float)


# ---------------------------------------------------------------------------
# read_image
# ---------------------------------------------------------------------------


class TestReadImage:
    """Tests for the read_image() function using bioio backend."""

    def test_read_image_basic(self) -> None:
        """Read a single-series OME-TIFF and verify dimensions."""
        fp = tpath / "multi-channel-time-series.ome.tif"
        require_test_data([fp.name])
        da = ir.read_image(str(fp))
        assert da.dims == ("T", "C", "Z", "Y", "X")
        assert da.sizes["C"] >= 1

    def test_read_image_channel_names(self) -> None:
        """Assign custom channel names."""
        fp = tpath / "multi-channel-time-series.ome.tif"
        require_test_data([fp.name])
        da = ir.read_image(str(fp))
        n_c = da.sizes["C"]
        names = [f"ch{i}" for i in range(n_c)]
        da2 = ir.read_image(str(fp), channels=names)
        assert list(da2.coords["C"].values) == names

    def test_read_image_channel_mismatch(self) -> None:
        """Raise ValueError when channel count doesn't match."""
        fp = tpath / "multi-channel-time-series.ome.tif"
        require_test_data([fp.name])
        with pytest.raises(ValueError, match="Channel mismatch"):
            ir.read_image(str(fp), channels=["a"])

    def test_read_image_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            ir.read_image("nonexistent_file.tif")

    def test_read_image_tif(self) -> None:
        """Read a standard TIFF file."""
        fp = tpath / "t4_1.tif"
        require_test_data([fp.name])
        da = ir.read_image(str(fp))
        assert da.dims == ("T", "C", "Z", "Y", "X")
        assert da.sizes["Y"] > 0
        assert da.sizes["X"] > 0


# ---------------------------------------------------------------------------
# stitch_scenes
# ---------------------------------------------------------------------------


class TestStitchScenes:
    """Tests for the stitch_scenes() function using bioio backend."""

    def test_stitch_tile(self) -> None:
        """Stitch a tiled FEI image and verify pixel values."""
        fp = tpath / "t4_1.tif"
        require_test_data([fp.name])
        da = ir.stitch_scenes(str(fp))
        assert da.dims == ("T", "C", "Z", "Y", "X")
        assert da.sizes == {"T": 3, "C": 4, "Z": 1, "Y": 1280, "X": 1536}
        plane = da.isel(T=0, C=0, Z=0).values
        assert plane[861, 1224] == 7779
        assert plane[1222, 1416] == 9626
        plane2 = da.isel(T=2, C=3, Z=0).values
        assert plane2[1236, 1488] == 6294
        plane3 = da.isel(T=1, C=2, Z=0).values
        assert plane3[564, 1044] == 8560

    def test_stitch_void_tile(self) -> None:
        """Stitch a tiled image with void tiles filled with zeros."""
        fp = tpath / "tile6_1.tif"
        require_test_data([fp.name])
        da = ir.stitch_scenes(str(fp))
        assert da.sizes == {"T": 4, "C": 3, "Z": 1, "Y": 2560, "X": 2560}
        plane = da.isel(T=0, C=0, Z=0).values
        assert plane[1179, 882] == 6395
        plane2 = da.isel(T=3, C=0, Z=0).values
        assert plane2[1213, 1538] == 704
        # Void regions are zero
        assert plane2[2400, 2400] == 0
        assert plane2[2400, 200] == 0

    def test_stitch_channel_names(self) -> None:
        """Assign channel names during stitching."""
        fp = tpath / "t4_1.tif"
        require_test_data([fp.name])
        da = ir.stitch_scenes(str(fp), channels=["A", "B", "C", "D"])
        assert list(da.coords["C"].values) == ["A", "B", "C", "D"]

    def test_stitch_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            ir.stitch_scenes("nonexistent.tif")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    """Tests for the diff() function."""

    def test_identical_files(self) -> None:
        """Identical files return True."""
        fp = str(tpath / "multi-channel-time-series.ome.tif")
        require_test_data(["multi-channel-time-series.ome.tif"])
        assert ir.diff(fp, fp) is True

    def test_different_metadata(self) -> None:
        """Files with different metadata return False."""
        fp_a = str(tpath / "im1s1z3c5t_a.ome.tif")
        fp_b = str(tpath / "im1s1z2c5t_bmd.ome.tif")
        require_test_data(["im1s1z3c5t_a.ome.tif", "im1s1z2c5t_bmd.ome.tif"])
        assert ir.diff(fp_a, fp_b) is False

    def test_different_pixels(self) -> None:
        """Files with same metadata but different pixels return False."""
        fp_a = str(tpath / "im1s1z3c5t_a.ome.tif")
        fp_b = str(tpath / "im1s1z3c5t_bpix.ome.tif")
        require_test_data(["im1s1z3c5t_a.ome.tif", "im1s1z3c5t_bpix.ome.tif"])
        assert ir.diff(fp_a, fp_b) is False
