"""Microscopy Data Reader for nima_io Library.

This module provides a set of functions to read microscopy data files,
leveraging the bioformats library and custom processing for metadata and pixel
data.

For detailed function documentation and usage, refer to the Sphinx-generated
documentation.

"""

import collections
import logging
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import jpype  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pims  # type: ignore[import-untyped]
import scyjava  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scyjava import jimport

# Type hint variable, initialized to Any vs. None
Pixels = Any
Image = Any
ChannelSeparator = Any
OMEPyramidStore = Any


def start_loci(
    version: str = "7.3.0", java_memory: str = "4g", debug_level: str = "INFO"
) -> None:
    """Initialize the loci package and associated classes.

    This function starts the Java Virtual Machine (JVM), configures endpoints,
    and initializes global variables for the loci package and related classes.

    Global Variables
    ----------------
    loci: JPackage
        Global variable for the loci package.
    Pixels: ome.xml.model.Pixels
        Global variable for the Pixels class from the ome.xml.model package.
    Image: ome.xml.model.Image
        Global variable for the Image class from the ome.xml.model package.

    Parameters
    ----------
    version : str, optional
        Version of Bioformats to use (default "7.3.0").
    java_memory : str, optional
        Maximum memory for JVM (default "4g").
    debug_level : str, optional
        Logging level for Java process "ERROR", "WARN", "INFO", "DEBUG", "TRACE"
        (default "INFO").

    """
    global Pixels, Image, ChannelSeparator, OMEPyramidStore  # noqa: PLW0603[JVM]
    log_fp = "bf.log"
    scyjava.config.add_option(f"-Xmx{java_memory}")  # Configure memory
    scyjava.config.endpoints.append("org.slf4j:slf4j-reload4j:1.7.36")
    scyjava.config.endpoints.append(f"ome:formats-gpl:{version}")
    # Programmatically configure Log4j to overwrite the log file at startup
    log_manager = scyjava.jimport("org.apache.log4j.LogManager")
    file_appender = scyjava.jimport("org.apache.log4j.FileAppender")
    pattern_layout = scyjava.jimport("org.apache.log4j.PatternLayout")
    log_manager.resetConfiguration()
    # Create a FileAppender set to overwrite existing log file
    appender = file_appender(
        pattern_layout("%-4r [%t] %-5p %c %x - %m%n"), log_fp, False  # noqa: FBT003
    )  # fmt: skip
    log_manager.getRootLogger().addAppender(appender)
    # Get java runtime version
    runtime = scyjava.jimport("java.lang.Runtime")
    runtime_memory = np.round(runtime.getRuntime().maxMemory() / 2**30, 2)
    system = scyjava.jimport("java.lang.System")
    java_version = system.getProperty("java.version")
    print(f"Bioformats-{version} on java-{java_version} ({runtime_memory} GB)")
    # Start JVM
    scyjava.start_jvm()
    # Import and set the logging level
    debug_tools = jimport("loci.common.DebugTools")
    debug_tools.setRootLevel(debug_level)
    # Import the required classes
    ChannelSeparator = jimport("loci.formats.ChannelSeparator")
    OMEPyramidStore = jimport("loci.formats.ome.OMEPyramidStore")
    # Import the required classes from the ome.xml.model package
    Pixels = jimport("ome.xml.model.Pixels")
    Image = jimport("ome.xml.model.Image")


def ensure_jvm() -> None:
    """Start java VM and initialize logger (globally) or Attach running JVM."""
    if not scyjava.jvm_started():
        start_loci()


class JavaFieldUnit(Protocol):
    """Protocol for JavaField's unit representation."""

    def getSymbol(self) -> str:  # noqa: N802[Java]
        """Retrieve the symbol of the unit."""
        ...  # pragma: no cover


class JavaField(Protocol):
    """Protocol for JavaField."""

    def value(self) -> None | str | float | int:
        """Get the value of the JavaField."""
        ...  # pragma: no cover

    def unit(self) -> None | JavaFieldUnit:
        """Get the unit of the JavaField."""
        ...  # pragma: no cover


MDSingleValueType = str | bool | int | float | None
MDValueType = MDSingleValueType | tuple[MDSingleValueType, str]
FullMDValueType = list[tuple[tuple[int, ...], MDValueType]]

MDJavaFieldType = MDSingleValueType | JavaField


@dataclass(eq=True)
class StagePosition:
    """Dataclass representing stage position."""

    #: Position in the X dimension.
    x: float | None
    #: Position in the Y dimension.
    y: float | None
    #: Position in the Z dimension.
    z: float | None

    def __hash__(self) -> int:
        """Generate a hash value for the object based on its attributes."""
        return hash((self.x, self.y, self.z))


@dataclass(eq=True)
class VoxelSize:
    """Dataclass representing voxel size."""

    #: Size in the X dimension.
    x: float | None
    #: Size in the Y dimension.
    y: float | None
    #: Size in the Z dimension.
    z: float | None

    def __hash__(self) -> int:
        """Generate a hash value for the object based on its attributes."""
        return hash((self.x, self.y, self.z))


class MultiplePositionsError(Exception):
    """Exception raised when a series contains multiple stage positions."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class CoreMetadata:
    """Dataclass representing core metadata.

    Parameters
    ----------
    rdr : loci.formats.ChannelSeparator
        ChannelSeparator instance.

    """

    rdr: InitVar[ChannelSeparator]
    #: Number of series.
    size_s: int = field(init=False)
    #:  File format.
    file_format: str = field(init=False)
    #: List of sizes in the X dimension.
    size_x: list[int] = field(default_factory=list)
    #: List of sizes in the Y dimension.
    size_y: list[int] = field(default_factory=list)
    #: List of sizes in the C dimension.
    size_c: list[int] = field(default_factory=list)
    #: List of sizes in the Z dimension.
    size_z: list[int] = field(default_factory=list)
    #: List of sizes in the T dimension.
    size_t: list[int] = field(default_factory=list)
    #: List of bits per pixel.
    bits: list[int] = field(default_factory=list)
    #: List of names.
    name: list[str] = field(default_factory=list)
    #: List of acquisition dates.
    date: list[str | None] = field(default_factory=list)
    #: List of stage positions.
    stage_position: list[StagePosition] = field(default_factory=list)
    #: List of voxel sizes.
    voxel_size: list[VoxelSize] = field(default_factory=list)

    def __post_init__(self, rdr: ChannelSeparator) -> None:
        """Consolidate all core metadata."""
        self.size_s = rdr.getSeriesCount()
        self.file_format = rdr.getFormat()
        root = rdr.getMetadataStoreRoot()
        for i in range(self.size_s):
            image = root.getImage(i)
            pixels = image.getPixels()
            self.size_x.append(int(pixels.getSizeX().getValue()))
            self.size_y.append(int(pixels.getSizeY().getValue()))
            self.size_c.append(int(pixels.getSizeC().getValue()))
            self.size_z.append(int(pixels.getSizeZ().getValue()))
            self.size_t.append(int(pixels.getSizeT().getValue()))
            self.bits.append(int(pixels.getSignificantBits().getValue()))
            self.name.append(image.getName())
            # Date
            self.date.append(self._get_date(image))
            # Stage Positions
            self.stage_position.append(self._get_stage_position(pixels))
            # Voxel: Physical Sizes
            try:
                psx = pixels.getPhysicalSizeX().value()
            except Exception:
                psx = None
            try:
                psy = pixels.getPhysicalSizeY().value()
            except Exception:
                psy = None
            try:
                psz = pixels.getPhysicalSizeZ().value()
            except Exception:
                psz = None
            self.voxel_size.append(
                VoxelSize(
                    self._get_physical_size(psx),
                    self._get_physical_size(psy),
                    self._get_physical_size(psz),
                )
            )
        for attribute in [
            "size_x",
            "size_y",
            "size_c",
            "size_z",
            "size_t",
            "bits",
            "name",
            "date",
            "stage_position",
            "voxel_size",
        ]:
            if len(list(set(getattr(self, attribute)))) == 1:
                setattr(self, attribute, list(set(getattr(self, attribute))))

    def _get_stage_position(self, pixels: Pixels) -> StagePosition:
        """Retrieve the stage positions from the given pixels."""

        def raise_multiple_positions_error(message: str) -> None:
            raise MultiplePositionsError(message)

        try:
            pos = {
                StagePosition(
                    pixels.getPlane(i).getPositionX().value().doubleValue(),
                    pixels.getPlane(i).getPositionY().value().doubleValue(),
                    pixels.getPlane(i).getPositionZ().value().doubleValue(),
                )
                for i in range(pixels.sizeOfPlaneList())
            }
            if len(pos) == 1:
                stage_position = next(iter(pos))
            else:
                raise_multiple_positions_error("Multiple positions within a series.")
        except Exception:
            stage_position = StagePosition(None, None, None)
        return stage_position

    def _get_date(self, image: Image) -> str | None:
        try:
            return cast(str, image.getAcquisitionDate().getValue())
        except Exception:
            return None

    def _get_physical_size(self, value: float) -> float | None:
        try:
            return round(float(value), 6)
        except Exception:
            return None


@dataclass
class Metadata:
    """Dataclass representing all metadata."""

    #: Core metadata.
    core: CoreMetadata
    #: All metadata.
    full: dict[str, FullMDValueType]
    #: Log of missed keys.
    log_miss: dict[str, Any]


@dataclass
class ImageReaderWrapper:
    """Wrapper class for Bioformats image reader.

    Parameters
    ----------
    rdr : ChannelSeparator
        Bioformats image reader.
    """

    #: Bioformats image reader.
    rdr: ChannelSeparator
    #: Data type based on the bit depth of the image.
    dtype: type[np.int8] | type[np.int16] = field(init=False)

    def __post_init__(self) -> None:
        """Define array dtype."""
        self.dtype = self._get_dtype()

    def _get_dtype(self) -> type[np.int8] | type[np.int16]:
        bits_per_pixel = self.rdr.getBitsPerPixel()
        if bits_per_pixel in [8]:
            return np.int8
        elif bits_per_pixel in [12, 16]:
            return np.int16
        else:  # pragma: no cover
            # Handle other bit depths or raise an exception
            msg = f"Unsupported bit depth: {bits_per_pixel} bits per pixel"
            raise ValueError(msg)

    def read(  # noqa: PLR0913[Bioformats]
        self,
        series: int = 0,
        z: int = 0,
        c: int = 0,
        t: int = 0,
        *,
        rescale: bool = False,
    ) -> NDArray[np.generic]:
        """Read image data from the specified series, z-stack, channel, and time point.

        Parameters
        ----------
        series : int, optional
            Index of the image series. Default is 0.
        z : int, optional
            Index of the z-stack. Default is 0.
        c : int, optional
            Index of the channel. Default is 0.
        t : int, optional
            Index of the time point. Default is 0.
        rescale : bool, optional
            Whether to rescale the data. Default is False.

        Returns
        -------
        NDArray[np.generic]
            NumPy array containing the frame data.
        """
        if rescale:
            pass  # pragma: no cover
        # Set the series
        self.rdr.setSeries(series)
        # Get index
        idx = self.rdr.getIndex(z, c, t)
        # Use openBytes to read a specific plane
        java_data = self.rdr.openBytes(idx)
        # Convert the Java byte array to a NumPy array
        np_data = np.frombuffer(jpype.JArray(jpype.JByte)(java_data), dtype=self.dtype)
        # Reshape the NumPy array based on the image dimensions
        np_data = np_data.reshape((self.rdr.getSizeY(), self.rdr.getSizeX()))
        # Add any additional logic or modifications if needed
        return np_data


def read(
    filepath: str,
) -> tuple[Metadata, ImageReaderWrapper]:
    """Read metadata and data using bioformats and scyjava.

    Get all OME metadata. bioformats.formatreader.ImageReader

    rdr as a lot of information e.g rdr.isOriginalMetadataPopulated() (core,
    OME, original metadata)

    Parameters
    ----------
    filepath : str
        The path to the image file.

    Returns
    -------
    md : Metadata
        Tidied metadata.
    wrapper : ImageReaderWrapper
        A wrapper to the Image Reader; to be used for accessing data from disk.

    Raises
    ------
    FileNotFoundError
        If the specified file is not found.

    Examples
    --------
    >>> md, wr = read("tests/data/multi-channel-time-series.ome.tif")
    >>> md.core.file_format
    'OME-TIFF'
    >>> md.core.size_c, md.core.size_t, md.core.size_x, md.core.bits
    ([3], [7], [439], [8])
    >>> a = wr.read(c=2, t=6, series=0, z=0, rescale=False)
    >>> a[20, 200]
    np.int8(-1)
    >>> md, wr = read("tests/data/LC26GFP_1.tf8")
    >>> wr.rdr.getSizeX(), md.core.size_x
    (1600, [1600])
    >>> wr.rdr.getMetadataStore()
    <java object 'loci.formats.ome.OMEPyramidStore'>

    """
    if not Path(filepath).is_file():
        msg = f"File not found: {filepath}"
        raise FileNotFoundError(msg)
    ensure_jvm()
    channel_separator = jimport("loci.formats.ChannelSeparator")
    rdr = channel_separator()  # Ensure each channel is a separate grayscale image
    # Set the metadata store instead of loci.formats.MetadataTools
    metadata_tools = jimport("loci.formats.MetadataTools")
    rdr.setMetadataStore(metadata_tools.createOMEXMLMetadata())
    rdr.setId(filepath)
    core_md = CoreMetadata(rdr)
    ome_store = rdr.getMetadataStore()
    full, log_miss = get_md_dict(ome_store, Path(filepath).with_suffix(".mmdata.log"))
    return Metadata(core_md, full, log_miss), ImageReaderWrapper(rdr)


def read_pims(filepath: str) -> tuple[Metadata, ImageReaderWrapper]:
    """Read metadata and initialize Bioformats reader using the pims library.

    Parameters
    ----------
    filepath : str
        The file path to the Bioformats file.

    Returns
    -------
    md : Metadata
        Tidied metadata.
    wrapper : ImageReaderWrapper
        A wrapper to the Loci image reader; to be used for accessing data from disk.

    Notes
    -----
    The core metadata includes information necessary to understand the basic
    structure of the pixels:

    - Image resolution
    - Number of focal planes
    - Time points (SizeT)
    - Channels (SizeC) and other dimensional axes
    - Byte order
    - Dimension order
    - Color arrangement (RGB, indexed color, or separate channels)
    - Thumbnail resolution

    The series metadata includes information about each series, such as the size
    in X, Y, C, T, and Z dimensions, physical sizes, pixel type, and position in
    XYZ coordinates.

    NB name and date are not core metadata.
    (series)
    (series, plane) where plane combines z, t and c?
    """
    fs = pims.Bioformats(filepath)
    core_md = CoreMetadata(fs.rdr)
    md = Metadata(core_md, {}, {})
    return md, ImageReaderWrapper(fs.rdr)


def stitch(
    core: CoreMetadata, wr: ImageReaderWrapper, c: int = 0, t: int = 0, z: int = 0
) -> npt.NDArray[np.float64]:
    """Stitch image tiles returning a tiled single plane.

    Parameters
    ----------
    core : CoreMetadata
        A dictionary containing information about the series of images, such as
        their positions.
    wr : ImageReaderWrapper
        An object that has a method `read` to read the images.
    c : int, optional
        The index or identifier for the images to be read (default is 0).
    t : int, optional
        The index or identifier for the images to be read (default is 0).
    z : int, optional
        The index or identifier for the images to be read (default is 0).

    Returns
    -------
    npt.NDArray[np.float64]
        The stitched image tiles.

    Raises
    ------
    ValueError
        If one or more series doesn't have a single XYZ position.
    IndexError
        If building tilemap fails in searching xy_position indexes.
    """
    if len({(p.x, p.y) for p in core.stage_position}) != len(core.stage_position):
        msg = "Duplicate position mapping detected."
        raise IndexError(msg)
    unique_x_positions = np.unique([p.x for p in core.stage_position])
    unique_y_positions = np.unique([p.y for p in core.stage_position])
    tile_rows, tile_cols = len(unique_y_positions), len(unique_x_positions)
    tilemap = np.full((tile_rows, tile_cols), fill_value=-1, dtype=int)
    position_to_index = {(p.x, p.y): i for i, p in enumerate(core.stage_position)}
    # Build the tilemap
    for y_index, y in enumerate(unique_y_positions):
        for x_index, x in enumerate(unique_x_positions):
            index = position_to_index.get((x, y))
            if index is not None:  # as some tile is empty
                tilemap[y_index, x_index] = index
    # Place the image tiles into the tiled_plane
    tiled_image_size = (core.size_y[0] * tile_rows, core.size_x[0] * tile_cols)
    tiled_plane = np.zeros(tiled_image_size)
    for y_tile in range(tile_rows):
        for x_tile in range(tile_cols):
            tile_index = tilemap[y_tile, x_tile]
            if tile_index >= 0:
                y_slice = slice(y_tile * core.size_y[0], (y_tile + 1) * core.size_y[0])
                x_slice = slice(x_tile * core.size_x[0], (x_tile + 1) * core.size_x[0])
                tiled_plane[y_slice, x_slice] = wr.read(
                    c=c, t=t, z=z, series=tile_index, rescale=False
                )
    return tiled_plane


def diff(fp_a: str, fp_b: str) -> bool:
    """Diff for two image data.

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
    """
    md_a, wr_a = read(fp_a)
    md_b, wr_b = read(fp_b)
    are_equal: bool = True
    # Check if metadata is equal
    are_equal = are_equal and (md_a.core == md_b.core)
    # MAYBE: print(md_b) maybe return md_a and different md_b
    if not are_equal:
        print("Metadata mismatch:")
        print("md_a:", md_a.core)
        print("md_b:", md_b.core)
    # Check pixel data equality
    are_equal = all(
        np.array_equal(
            wr_a.read(series=s, t=t, c=c, z=z, rescale=False),
            wr_b.read(series=s, t=t, c=c, z=z, rescale=False),
        )
        for s in range(md_a.core.size_s)
        for t in range(md_a.core.size_t[0])
        for c in range(md_a.core.size_c[0])
        for z in range(md_a.core.size_z[0])
    )
    return are_equal


def first_nonzero_reverse(llist: list[int]) -> None | int:
    """Return the index of the last nonzero element in a list.

    Parameters
    ----------
    llist : list[int]
        The input list of integers.

    Returns
    -------
    None | int
        The index of the last nonzero element. Returns None if all elements are zero.

    Examples
    --------
    >>> first_nonzero_reverse([0, 2, 0, 0])
    -3
    >>> first_nonzero_reverse([0, 0, 0])
    None

    """
    for i in range(-1, -len(llist) - 1, -1):
        if llist[i] != 0:
            return i
    return None


def get_md_dict(
    ome_store: OMEPyramidStore,
    log_fp: None | Path = None,
) -> tuple[dict[str, FullMDValueType], dict[str, str]]:
    """Parse xml_md and return parsed md dictionary and md status dictionary.

    Parameters
    ----------
    ome_store: OMEPyramidStore
        The metadata java object.
    log_fp: None | Path
        The filepath, used for logging JavaExceptions (default=None).

    Returns
    -------
    md: dict[str, FullMDValueType]
        Parsed metadata dictionary excluding None values.
    mdd: dict[str, str]
        Metadata status dictionary indicating if a value was found ('Found'),
        is None ('None'), or if there was a JavaException ('Jmiss').

    """
    # Configure logging.
    logging.basicConfig(
        filename=log_fp,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    # Status constants.
    found = "Found"
    none_value = "None"
    error = "Jmiss"

    def process_key(
        ome_store: OMEPyramidStore, key: str, n_max_pars: int
    ) -> tuple[FullMDValueType | None, str]:
        """Invoke the method named 'key' with a tuple of length up to 'n_max_pars'."""
        for npar in range(n_max_pars + 1):
            method = getattr(ome_store, key)
            try:
                value = method(*(0,) * npar)
            except (TypeError, RuntimeError):
                continue
            except Exception:
                logging.exception(f"Error processing {key}: {npar}")
                return None, error
            if value is not None:
                return get_allvalues_grouped(ome_store, key, npar), found
        return None, none_value

    def process_all_keys(
        ome_store: OMEPyramidStore,
        key_prefix: str,
        n_max_pars: int,
        excluded: set[str],
    ) -> tuple[dict[str, FullMDValueType], dict[str, str]]:
        """Process ome_store methods starting with key_prefix and not excluded."""
        keys = [
            m for m in dir(ome_store) if m.startswith(key_prefix) and m not in excluded
        ]
        full = {}
        log_miss = {}
        for key in keys:
            value, status = process_key(ome_store, key, n_max_pars)
            if value is not None:
                full[key[3:]] = value
            log_miss[key] = status
        return full, log_miss

    # Assuming 'xml_md' is a predefined XML metadata object
    key_prefix = "get"
    n_max_pars = 3
    excluded = {
        "getRoot",
        "getClass",
        "getXMLAnnotationValue",
        "getPixelsBinDataBigEndian",
    }
    return process_all_keys(ome_store, key_prefix, n_max_pars, excluded)


def convert_field(field: JavaField | float | str | None) -> MDValueType:
    """Convert a JavaField to a Python data type, optionally including its unit symbol.

    This function handles cases where the JavaField could be None, which is
    possible for composite metadata, which may contain None e.g.,
    [(4, 1), (543.0, 'nm')] might be followed by [(4, 2), None].

    Parameters
    ----------
    field : JavaField | float | str | None
        A field from Java, potentially holding a numeric value and a unit.

    Returns
    -------
    MDValueType
        The converted metadata value as a Python primitive type (int, float,
        str, or bool), or None, or a tuple of the value and the unit symbol (as
        a string) if a unit is associated with the value.

    """
    # Directly return if value is already one of the basic Python types
    if isinstance(field, bool | int | float) or field is None:  # float, str
        return field
    # Handle case if field is a Java object with unit and value attributes
    if hasattr(field, "value") and hasattr(field, "unit"):
        # Recursively call convert_value to unwrap the 'field' attribute
        value = convert_field(field.value())
        unwrapped_value = value[0] if isinstance(value, tuple) else value
        unit_obj = field.unit()
        unit_symbol = unit_obj.getSymbol() if unit_obj is not None else ""
        return unwrapped_value, unit_symbol
    # To address potential floating-point inaccuracies such as those that may
    # arise from calling getDouble(), which could convert 0.9 to 0.8999.
    snum = str(field)
    try:
        return int(snum)
    except ValueError:
        try:
            return float(snum)
        except ValueError:
            return snum


class StopExceptionError(Exception):
    """Exception raised when need to stop."""

    pass


def next_tuple(llist: list[int], *, increment_last: bool) -> list[int]:
    """Generate the next tuple in lexicographical order.

    This function generates the next tuple in lexicographical order based on
    the input list `llist`. The lexicographical order is defined as follows:

    - If the `s` flag is True, the last element of the tuple is incremented.
    - If the `s` flag is False, the function finds the rightmost non-zero
      element and increments the element to its left, setting the rightmost
      non-zero element to 0.

    Parameters
    ----------
    llist : list[int]
        The input list representing a tuple.
    increment_last : bool
        A flag indicating whether to increment the last element or not.

    Returns
    -------
    list[int]
        The next tuple in lexicographical order.

    Raises
    ------
    StopExceptionError:
        If the input tuple is empty or if the generation needs to stop.

    Examples
    --------
    >>> next_tuple([0, 0, 0], increment_last=True)
    [0, 0, 1]
    >>> next_tuple([0, 0, 1], increment_last=True)
    [0, 0, 2]
    >>> next_tuple([0, 0, 2], increment_last=False)
    [0, 1, 0]
    >>> next_tuple([0, 1, 2], increment_last=False)
    [0, 2, 0]
    >>> next_tuple([2, 0, 0], increment_last=False)
    Traceback (most recent call last):
    ...
    nima_io.read.StopExceptionError

    """
    if not llist:  # Next item never exists for an empty tuple.
        raise StopExceptionError
    if increment_last:
        llist[-1] += 1
        return llist
    idx = first_nonzero_reverse(llist)
    if idx == -len(llist):
        raise StopExceptionError
    if idx is not None:
        llist[idx] = 0
        llist[idx - 1] += 1
    return llist


def retrieve_values(ome_store: OMEPyramidStore, key: str, npar: int) -> FullMDValueType:
    """Retrieve values for the given key and number of parameters from the OMEStore."""

    def append_converted_value(tuple_list: list[int]) -> None:
        tuple_pars = tuple(tuple_list)
        value = convert_field(getattr(ome_store, key)(*tuple_pars))
        res.append((tuple_pars, value))

    res: FullMDValueType = []
    tuple_list = [0] * npar
    # Initial value retrieval
    append_converted_value(tuple_list)
    increment_last = True
    while True:
        try:
            tuple_list = next_tuple(tuple_list, increment_last=increment_last)
            # Subsequent value retries
            append_converted_value(tuple_list)
            increment_last = True
        except StopExceptionError:
            break
        except Exception:
            increment_last = False
    return res


def group_metadata(res: FullMDValueType) -> FullMDValueType:
    """Tidy up by grouping common metadata."""
    length_md_with_units = 2
    if len(res) > 1:
        values_list = [e[1] for e in res]
        if values_list.count(values_list[0]) == len(res):
            res = [res[-1]]
        elif len(res[0][0]) >= length_md_with_units:
            # first group the list of tuples by (tuple_idx=0)
            grouped_res = collections.defaultdict(list)
            for tuple_pars, value in res:
                grouped_res[tuple_pars[0]].append(value)
            max_key = max(grouped_res.keys())  # or: res[-1][0][0]
            # now check for single common value within a group
            new_res: FullMDValueType = []
            for k, val in grouped_res.items():
                if val.count(val[0]) == len(val):
                    new_res.append(((k, len(val) - 1), val[-1]))
            if new_res:
                res = new_res
            # now check for the same group repeated
            for _, val in grouped_res.items():
                if val != grouped_res[max_key]:
                    break
            else:
                # This block executes if the loop completes without a 'break'
                res = res[-len(val) :]
    return res


def get_allvalues_grouped(
    ome_store: OMEPyramidStore, key: str, npar: int
) -> FullMDValueType:
    """Retrieve and group metadata values for a given key.

    Assume that all the OMEStore methods have a certain number of parameters. Group
    common values into a list without knowledge of parameters meaning.

    Parameters
    ----------
    ome_store: OMEPyramidStore
        The metadata java object.
    key : str
        The key for which values are retrieved.
    npar : int
        The number of parameters for the key.

    Returns
    -------
    FullMDValueType
        A list of tuples containing the tuple configuration and corresponding values.

    """
    res = retrieve_values(ome_store, key, npar)
    res = group_metadata(res)
    return res
