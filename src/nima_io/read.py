"""Microscopy Data Reader for nima_io Library.

This module provides functions to read microscopy data files. The primary
entry point is :func:`read_image`, which returns a lazy-loaded
``xarray.DataArray`` backed by dask with standardized TCZYX dimensions.

Structured metadata is attached as ``data.attrs["metadata"]``
(:class:`Metadata`), consolidating channel settings, voxel sizes,
stage positions, exposure times, and timestamps from the OME model.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import warnings
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from bioio import BioImage
from ome_types import OME
from xarray import DataArray

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Channel:
    """Per-channel illumination and detection settings.

    Attributes
    ----------
    wavelength : int
        Illumination wavelength (nm).
    attenuation : float
        Illumination attenuation.
    exposure : float
        Exposure time (s), from the first plane of this channel.
    gain : float
        Detector gain.
    binning : str
        Detector binning (e.g. ``"1x1"``).
    filters : list[str]
        Excitation filter IDs.
    """

    wavelength: int
    attenuation: float
    exposure: float
    gain: float
    binning: str
    filters: list[str]

    def __repr__(self) -> str:
        """Represent most relevant metadata."""
        return (
            f"Channel(\u03bb={self.wavelength}, att={self.attenuation}, "
            f"exp={self.exposure}, gain={self.gain}, "
            f"binning={self.binning})"
        )


@dataclass(eq=True, frozen=True)
class StagePosition:
    """Stage position in physical units.

    Attributes
    ----------
    x : float | None
        X position.
    y : float | None
        Y position.
    z : float | None
        Z position.
    """

    x: float | None
    y: float | None
    z: float | None

    def __hash__(self) -> int:
        """Generate a hash value for the object based on its attributes."""
        return hash((self.x, self.y, self.z))

    def __repr__(self) -> str:
        """Represent most relevant metadata."""
        return f"XYZ={pformat((self.x, self.y, self.z))}"


@dataclass(eq=True)
class VoxelSize:
    """Voxel size in physical units.

    Attributes
    ----------
    x : float | None
        Pixel size in X.
    y : float | None
        Pixel size in Y.
    z : float | None
        Pixel size in Z.
    """

    x: float | None
    y: float | None
    z: float | None

    def __hash__(self) -> int:
        """Generate a hash value for the object based on its attributes."""
        return hash((self.x, self.y, self.z))


@dataclass
class Metadata:
    """Consolidated metadata from OME, attached as ``data.attrs["metadata"]``.

    Built automatically by :func:`read_image` and :func:`stitch_scenes`.
    Lists are indexed per scene/series; when all scenes share the same
    value the list is collapsed to a single element.

    Attributes
    ----------
    ome : InitVar[OME]
        OME object used to initialise the class (not stored).
    size_s : int
        Number of scenes.
    size_x : list[int]
        X sizes per scene.
    size_y : list[int]
        Y sizes per scene.
    size_z : list[int]
        Z sizes per scene.
    size_c : list[int]
        Channel counts per scene.
    size_t : list[int]
        Time-point counts per scene.
    bits : list[int]
        Significant bits per scene.
    objective : list[str | None]
        Objective ID per scene.
    name : list[str]
        Image ID per scene.
    date : list[str | None]
        Acquisition date per scene.
    stage_position : list[dict[StagePosition, tuple[int, int, int]]]
        ``{StagePosition: (T, C, Z)}`` per scene.
    voxel_size : list[VoxelSize]
        Voxel sizes per scene.
    channels : list[list[Channel]]
        Channel settings per scene.
    tcz_deltat : list[list[tuple[int, int, int, float]]]
        ``(T, C, Z, delta_t)`` from each plane per scene.
    """

    ome: InitVar[OME]
    _ome: OME = field(init=False, repr=False)
    size_s: int = 1
    size_x: list[int] = field(default_factory=list)
    size_y: list[int] = field(default_factory=list)
    size_z: list[int] = field(default_factory=list)
    size_c: list[int] = field(default_factory=list)
    size_t: list[int] = field(default_factory=list)
    bits: list[int] = field(default_factory=list)
    objective: list[str | None] = field(default_factory=list)
    name: list[str] = field(default_factory=list)
    date: list[str | None] = field(default_factory=list)
    stage_position: list[dict[StagePosition, tuple[int, int, int]]] = field(
        default_factory=list
    )
    voxel_size: list[VoxelSize] = field(default_factory=list)
    channels: list[list[Channel]] = field(default_factory=list)
    tcz_deltat: list[list[tuple[int, int, int, float]]] = field(default_factory=list)

    def __repr__(self) -> str:
        """Represent most relevant metadata."""
        return (
            f"Metadata(S={self.size_s}, T={self.size_t}, C={self.size_c}, "
            f"Z={self.size_z}, Y={self.size_y}, X={self.size_x}\n"
            f"  bits={self.bits}, obj={self.objective}\n"
            f"  voxel_size={pformat(self.voxel_size)}\n"
            f"  channels=\n{pformat(self.channels)})"
        )

    def __post_init__(self, ome: OME) -> None:
        """Consolidate all core metadata from OME."""
        self._ome = ome
        self.size_s = len(ome.images)
        for image in ome.images:
            pixels = image.pixels
            self.size_x.append(pixels.size_x)
            self.size_y.append(pixels.size_y)
            self.size_z.append(pixels.size_z)
            self.size_c.append(pixels.size_c)
            self.size_t.append(pixels.size_t)
            self.bits.append(pixels.significant_bits or 0)
            self.name.append(image.id)
            self.objective.append(
                image.objective_settings.id if image.objective_settings else None
            )
            self.date.append(
                str(image.acquisition_date) if image.acquisition_date else None
            )
            self.stage_position.append(self._get_stage_positions(pixels.planes))
            self.voxel_size.append(
                VoxelSize(
                    pixels.physical_size_x,
                    pixels.physical_size_y,
                    pixels.physical_size_z,
                )
            )
            self.channels.append(self._build_channels(pixels.channels, pixels.planes))
            self.tcz_deltat.append(
                [
                    (plane.the_t, plane.the_c, plane.the_z, plane.delta_t or 0.0)
                    for plane in pixels.planes
                ]
            )
        self._deduplicate()

    @staticmethod
    def _build_channels(ch_list: list[Any], planes: list[Any]) -> list[Channel]:
        """Build Channel objects combining channel and plane metadata."""
        result: list[Channel] = []
        for ci, ch in enumerate(ch_list):
            ls = ch.light_source_settings
            ds = ch.detector_settings
            plane = next((p for p in planes if p.the_c == ci), None)
            result.append(
                Channel(
                    wavelength=int(ls.wavelength or 0) if ls else 0,
                    attenuation=ls.attenuation or 0.0 if ls else 0.0,
                    exposure=float(plane.exposure_time or 0.0) if plane else 0.0,
                    gain=float(ds.gain or 0.0) if ds else 0.0,
                    binning=(str(ds.binning.value) if ds and ds.binning else "1x1"),
                    filters=[
                        d.id.replace("Filter:", "")
                        for d in (
                            ch.light_path.excitation_filters if ch.light_path else []
                        )
                    ],
                )
            )
        return result

    def _deduplicate(self) -> None:
        """Collapse per-scene lists when all values are identical."""
        for attr in (
            "size_x",
            "size_y",
            "size_z",
            "size_c",
            "size_t",
            "bits",
            "name",
            "objective",
            "date",
            "voxel_size",
        ):
            vals = getattr(self, attr)
            if len(set(vals)) == 1:
                setattr(self, attr, list(set(vals)))
        if all(ch == self.channels[0] for ch in self.channels[1:]):
            self.channels = [self.channels[0]]

    @staticmethod
    def _get_stage_positions(
        planes: list[Any],
    ) -> dict[StagePosition, tuple[int, int, int]]:
        """Retrieve stage positions from planes."""
        pos_dict: dict[StagePosition, tuple[int, int, int]] = {}
        for plane in planes:
            x, y, z = plane.position_x, plane.position_y, plane.position_z
            pos = StagePosition(
                float(x) if x is not None else None,
                float(y) if y is not None else None,
                float(z) if z is not None else None,
            )
            pos_dict[pos] = (int(plane.the_t), int(plane.the_c), int(plane.the_z))
        return pos_dict


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def _handle_tf8_workaround(fp: Path) -> Path:
    """Create a temporary symlink with .tif extension for .tf8 files.

    This ensures compatibility with bioio plugins that rely on file extensions.
    The temporary directory is cleaned up on process exit.
    """
    tmp_dir = tempfile.mkdtemp(prefix="nima_tf8_")
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    tmp_fp = Path(tmp_dir) / fp.with_suffix(".tif").name
    try:
        tmp_fp.symlink_to(fp.resolve())
    except OSError:
        shutil.copy(fp, tmp_fp)
    warnings.warn(
        f"Renaming .tf8 to .tif in {tmp_fp} for bioio compatibility. "
        "Temporary file will be removed on exit.",
        UserWarning,
        stacklevel=2,
    )
    return tmp_fp


def read_image(fp: str | Path, channels: Sequence[str] | None = None) -> DataArray:
    """Read a microscopy image file into a lazy-loaded xarray DataArray.

    Uses ``bioio`` to auto-dispatch to the best available reader plugin
    (pure-Python readers for TIFF/OME-TIFF/LIF when possible, falling back
    to bioformats via Java for other formats).

    Parameters
    ----------
    fp : str | Path
        Path to the image file.
    channels : Sequence[str] | None
        Optional channel names. When provided these replace the default
        channel coordinate on the ``"C"`` dimension.  The length must match
        the number of channels in the file.

    Returns
    -------
    DataArray
        Lazy-loaded ``xarray.DataArray`` with dimensions ``(T, C, Z, Y, X)``.

        * ``data.attrs["metadata"]`` — :class:`Metadata` with consolidated
          channel settings (wavelength, attenuation, exposure, gain, binning),
          voxel sizes, stage positions, and timestamps.
        * ``data.attrs["ome_metadata"]`` — raw :class:`ome_types.OME` object.

    Raises
    ------
    FileNotFoundError
        If *fp* does not exist.
    ValueError
        If the length of *channels* does not match the number of channels.
    Exception
        If bioio cannot read the file (re-raised from the reader backend).

    Examples
    --------
    >>> da = read_image("tests/data/multi-channel-time-series.ome.tif")
    >>> da.dims
    ('T', 'C', 'Z', 'Y', 'X')
    >>> da.sizes["C"]
    3

    """
    fp = Path(fp)
    if not fp.is_file():
        msg = f"File not found: {fp}"
        raise FileNotFoundError(msg)

    try:
        img = BioImage(fp)
    except Exception:
        if fp.suffix == ".tf8":
            tmp_fp = _handle_tf8_workaround(fp)
            img = BioImage(tmp_fp)
        else:
            raise

    data: DataArray = img.xarray_dask_data

    # Build and attach structured metadata
    try:
        ome = img.ome_metadata
        if isinstance(ome, OME):
            md = Metadata(ome)
            data.attrs["metadata"] = md
            data.attrs["ome_metadata"] = ome
    except Exception:  # noqa: S110
        pass  # Metadata is best-effort; missing OME is not fatal.

    if channels is not None:
        n_channels = data.sizes["C"]
        if len(channels) != n_channels:
            msg = f"Channel mismatch: file has {n_channels}, provided {len(channels)}"
            raise ValueError(msg)

        # Validate C/G/R wavelength ordering when all three are present
        if "ome_metadata" in data.attrs:
            _validate_channel_wavelengths(data.attrs["ome_metadata"], channels)

        data = data.assign_coords(C=list(channels))

    return data


def _validate_channel_wavelengths(ome: OME, channels: Sequence[str]) -> None:
    """Warn if C/G/R channel names don't match expected wavelength order."""
    if not ome.images:
        return
    pixels = ome.images[0].pixels
    if len(pixels.channels) != len(channels):
        return

    name_to_wave: dict[str, float] = {}
    for name, ch_meta in zip(channels, pixels.channels, strict=False):
        if ch_meta.light_source_settings and ch_meta.light_source_settings.wavelength:
            name_to_wave[name] = float(ch_meta.light_source_settings.wavelength)

    if {"C", "G", "R"}.issubset(name_to_wave):
        w_c, w_g, w_r = name_to_wave["C"], name_to_wave["G"], name_to_wave["R"]
        if not (w_c < w_g < w_r):
            msg = (
                f"Channel wavelength validation failed: "
                f"Expected λ_C < λ_G < λ_R. "
                f"Got C={w_c}nm, G={w_g}nm, R={w_r}nm."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)


def _extract_tile_positions(
    ome: OME,
) -> tuple[list[tuple[float, float]], int, int, int, int, int]:
    """Extract stage positions and tile dimensions from OME metadata.

    Parameters
    ----------
    ome : OME
        Parsed OME metadata object.

    Returns
    -------
    positions : list[tuple[float, float]]
        ``(x, y)`` stage position per image/scene.
    tile_y : int
        Tile height in pixels.
    tile_x : int
        Tile width in pixels.
    size_t : int
        Number of time points.
    size_c : int
        Number of channels.
    size_z : int
        Number of z-slices.

    Raises
    ------
    ValueError
        If images are empty, tile sizes differ, or positions are missing.
    """
    if not ome.images:
        msg = "No images found in OME metadata."
        raise ValueError(msg)

    tile_y = ome.images[0].pixels.size_y
    tile_x = ome.images[0].pixels.size_x

    positions: list[tuple[float, float]] = []
    for image in ome.images:
        pix = image.pixels
        if pix.size_y != tile_y or pix.size_x != tile_x:
            msg = "Inconsistent tile sizes across scenes."
            raise ValueError(msg)
        if not pix.planes or pix.planes[0].position_x is None:
            msg = "Stage positions missing from OME metadata."
            raise ValueError(msg)
        p = pix.planes[0]
        positions.append((float(p.position_x), float(p.position_y)))  # type: ignore[arg-type]

    ref = ome.images[0].pixels
    return positions, tile_y, tile_x, ref.size_t, ref.size_c, ref.size_z


def _build_tilemap(
    positions: list[tuple[float, float]],
) -> tuple[npt.NDArray[np.intp], int, int]:
    """Build a 2-D tilemap from stage positions.

    Parameters
    ----------
    positions : list[tuple[float, float]]
        ``(x, y)`` stage position per scene.

    Returns
    -------
    tilemap : npt.NDArray[np.intp]
        2-D array where ``tilemap[row, col]`` is the scene index
        (``-1`` for void tiles).
    n_rows : int
        Number of tile rows.
    n_cols : int
        Number of tile columns.
    """
    unique_x = np.unique([p[0] for p in positions])
    unique_y = np.unique([p[1] for p in positions])
    n_cols, n_rows = len(unique_x), len(unique_y)

    pos_to_scene = {pos: i for i, pos in enumerate(positions)}
    tilemap = np.full((n_rows, n_cols), fill_value=-1, dtype=np.intp)
    for yi, y in enumerate(unique_y):
        for xi, x in enumerate(unique_x):
            idx = pos_to_scene.get((x, y))
            if idx is not None:
                tilemap[yi, xi] = idx
    return tilemap, n_rows, n_cols


def stitch_scenes(
    fp: str | Path,
    channels: Sequence[str] | None = None,
) -> DataArray:
    """Read a tiled/multi-scene microscopy file and stitch into a single DataArray.

    Uses OME metadata stage positions to place each scene (tile) on a grid.
    Missing tiles are filled with zeros.  The result is a lazy
    ``xarray.DataArray`` with dimensions ``(T, C, Z, Y, X)``.

    Parameters
    ----------
    fp : str | Path
        Path to the image file.
    channels : Sequence[str] | None
        Optional channel names (see :func:`read_image`).

    Returns
    -------
    DataArray
        Stitched image with dimensions ``(T, C, Z, Y, X)``.

    Raises
    ------
    FileNotFoundError
        If *fp* does not exist.
    ValueError
        If stage positions are missing or tile sizes are inconsistent.

    Examples
    --------
    >>> da = stitch_scenes("tests/data/t4_1.tif")
    >>> da.dims
    ('T', 'C', 'Z', 'Y', 'X')
    >>> da.sizes["Y"], da.sizes["X"]
    (1280, 1536)

    """
    fp = Path(fp)
    if not fp.is_file():
        msg = f"File not found: {fp}"
        raise FileNotFoundError(msg)

    img = BioImage(fp)
    ome = img.ome_metadata
    positions, tile_y, tile_x, size_t, size_c, size_z = _extract_tile_positions(ome)
    tilemap, n_rows, n_cols = _build_tilemap(positions)

    # Read all scenes and assemble into a stitched dask array.
    import dask.array as da  # noqa: PLC0415

    scene_arrays: dict[int, DataArray] = {}
    for scene_idx in range(len(ome.images)):
        img.set_scene(scene_idx)
        scene_arrays[scene_idx] = img.xarray_dask_data

    ref = scene_arrays[0]
    zero_tile = da.zeros(
        (size_t, size_c, size_z, tile_y, tile_x), dtype=ref.dtype, chunks=-1
    )
    rows_list = [
        [
            scene_arrays[tilemap[yi, xi]].data if tilemap[yi, xi] >= 0 else zero_tile
            for xi in range(n_cols)
        ]
        for yi in range(n_rows)
    ]
    stitched = da.block(rows_list)  # type: ignore[no-untyped-call]

    result = DataArray(stitched, dims=("T", "C", "Z", "Y", "X"))
    result.attrs["ome_metadata"] = ome
    result.attrs["metadata"] = Metadata(ome)
    result.attrs["tilemap"] = tilemap

    if channels is not None:
        n_channels = result.sizes["C"]
        if len(channels) != n_channels:
            msg = f"Channel mismatch: file has {n_channels}, provided {len(channels)}"
            raise ValueError(msg)
        _validate_channel_wavelengths(ome, channels)
        result = result.assign_coords(C=list(channels))

    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def diff(fp_a: str, fp_b: str) -> bool:
    """Compare two microscopy image files for equality.

    Compares metadata (OME) and all pixel data across scenes.

    Parameters
    ----------
    fp_a : str
        File path for the first image.
    fp_b : str
        File path for the second image.

    Returns
    -------
    bool
        True if the two files are equal.

    Examples
    --------
    >>> diff("tests/data/im1s1z3c5t_a.ome.tif", "tests/data/im1s1z3c5t_a.ome.tif")
    True

    """
    img_a = BioImage(fp_a)
    img_b = BioImage(fp_b)

    # Quick scene-count check.
    if len(img_a.scenes) != len(img_b.scenes):
        print("Metadata mismatch: different number of scenes.")
        return False

    # Pixel-level comparison for every scene.
    for scene_idx in range(len(img_a.scenes)):
        img_a.set_scene(scene_idx)
        img_b.set_scene(scene_idx)
        try:
            data_a = img_a.xarray_dask_data
            data_b = img_b.xarray_dask_data
        except Exception:
            print(f"Metadata mismatch in scene {scene_idx}: cannot read data.")
            return False
        if dict(data_a.sizes) != dict(data_b.sizes):
            print(f"Metadata mismatch in scene {scene_idx}: dimensions differ.")
            return False
        if not np.array_equal(data_a.values, data_b.values):
            return False

    return True
