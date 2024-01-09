"""Microscopy Data Reader for nima_io Library.

This module provides a set of functions to read microscopy data files,
leveraging the bioformats library and custom processing for metadata and pixel
data.

For detailed function documentation and usage, refer to the Sphinx-generated
documentation.

"""

import collections
import hashlib
import logging
import urllib.request
import warnings
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urljoin

import jpype  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pims  # type: ignore[import-untyped]
import scyjava  # type: ignore[import-untyped]
from numpy.typing import NDArray

# Type hint variable, initialized to Any vs. None
Pixels = Any
Image = Any
loci = Any
Memoizer = Any
OMEPyramidStore = Any


def start_loci() -> None:
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

    """
    global loci, Pixels, Image, Memoizer, OMEPyramidStore  # [JVM]
    scyjava.config.endpoints.append("ome:formats-gpl:6.10.1")
    scyjava.start_jvm()
    loci = jpype.JPackage("loci")
    loci.common.DebugTools.setRootLevel("ERROR")
    model_jar = jpype.JPackage("ome.xml.model")
    Pixels = model_jar.Pixels
    Image = model_jar.Image
    formats_jar = jpype.JPackage("loci.formats")
    Memoizer = formats_jar.Memoizer
    OMEPyramidStore = formats_jar.ome.OMEPyramidStore


# # if not jpype.isJVMStarted():
# if not scyjava.jvm_started():
#     start_loci()


class JavaField(Protocol):
    """Define a Protocol for JavaField."""

    def value(self) -> None | str | float | int:
        """Get the value of the JavaField.

        Returns
        -------
        None | str | float | int
            The value of the JavaField, which can be None or one of the specified types.
        """
        ...


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
    rdr : loci.formats.Memoizer
        Memoizer instance.

    """

    rdr: InitVar[Memoizer]
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

    def __post_init__(self, rdr: Memoizer) -> None:
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


class ImageReaderWrapper:
    """Wrapper class for Bioformats image reader.

    Parameters
    ----------
    rdr : Memoizer
        Bioformats image reader.

    Attributes
    ----------
    rdr : loci.formats.Memoizer
        Bioformats image reader.
    dtype : type[np.int8]| type[np.int16]
        Data type based on the bit depth of the image.
    """

    def __init__(self, rdr: Memoizer) -> None:
        self.rdr = rdr
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

    def read(
        self, series: int = 0, z: int = 0, c: int = 0, t: int = 0, rescale: bool = False
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
            pass
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
    """Read a data using bioformats, scyjava and jpype.

    Get all OME metadata. bioformats.formatreader.ImageReader

    Parameters
    ----------
    filepath : str
        The path to the data file.

    Returns
    -------
    md : Metadata
        Tidied metadata.
    wrapper : ImageReaderWrapper
        A wrapper to the Loci image reader; to be used for accessing data from disk.

    Raises
    ------
    FileNotFoundError
        If the specified file is not found.

    Examples
    --------
    >>> md, wr = read('tests/data/multi-channel-time-series.ome.tif')
    >>> md.core.file_format
    'OME-TIFF'
    >>> md.core.size_c, md.core.size_t, md.core.size_x, md.core.bits
    ([3], [7], [439], [8])
    >>> a = wr.read(c=2, t=6, series=0, z=0, rescale=False)
    >>> a[20,200]
    -1
    >>> md, wr = read("tests/data/LC26GFP_1.tf8")
    >>> wr.rdr.getSizeX(), md.core.size_x
    (1600, [1600])
    >>> wr.rdr.getMetadataStore()
    <java object 'loci.formats.ome.OMEPyramidStore'>

    """
    if not Path(filepath).is_file():
        msg = f"File not found: {filepath}"
        raise FileNotFoundError(msg)
    if not scyjava.jvm_started():
        start_loci()
    # rdr = loci.formats.ImageReader()
    rdr = loci.formats.Memoizer()  # 32 vs 102 ms
    # rdr.setMetadataStore(loci.formats.MetadataTools.createOMEXMLMetadata())
    rdr.setId(filepath)
    core_md = CoreMetadata(rdr)
    # Create a wrapper around the ImageReader
    wrapper = ImageReaderWrapper(rdr)
    full_md, log_miss = get_md_dict(
        rdr.getMetadataStore(), Path(filepath).with_suffix(".mmdata.log")
    )
    md = Metadata(core_md, full_md, log_miss)
    return md, wrapper


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
    md: CoreMetadata, wrapper: ImageReaderWrapper, c: int = 0, t: int = 0, z: int = 0
) -> npt.NDArray[np.float64]:
    """Stitch image tiles returning a tiled single plane.

    Parameters
    ----------
    md : CoreMetadata
        A dictionary containing information about the series of images, such as
        their positions.
    wrapper : ImageReaderWrapper
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
    xyz_list_of_sets = [{(p.x, p.y, p.z)} for p in md.stage_position]
    if not all(len(p) == 1 for p in xyz_list_of_sets):
        msg = "One or more series doesn't have a single XYZ position."
        raise ValueError(msg)
    xy_positions = [next(iter(p))[:2] for p in xyz_list_of_sets]
    unique_x = np.sort(list({xy[0] for xy in xy_positions}))
    unique_y = np.sort(list({xy[1] for xy in xy_positions}))
    tiley = len(unique_y)
    tilex = len(unique_x)
    # tilemap only for complete tiles without None tile
    tilemap = np.zeros(shape=(tiley, tilex), dtype=int)
    for yi, y in enumerate(unique_y):
        for xi, x in enumerate(unique_x):
            indexes = [i for i, v in enumerate(xy_positions) if v == (x, y)]
            li = len(indexes)
            if li == 0:
                tilemap[yi, xi] = -1
            elif li == 1:
                tilemap[yi, xi] = indexes[0]
            else:
                msg = "Building tilemap failed in searching xy_position indexes."
                raise IndexError(msg)
    tiled_plane = np.zeros((md.size_y[0] * tiley, md.size_x[0] * tilex))
    for yt in range(tiley):
        for xt in range(tilex):
            if tilemap[yt, xt] >= 0:
                tiled_plane[
                    yt * md.size_y[0] : (yt + 1) * md.size_y[0],
                    xt * md.size_x[0] : (xt + 1) * md.size_x[0],
                ] = wrapper.read(c=c, t=t, z=z, series=tilemap[yt, xt], rescale=False)
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


def download_loci_jar() -> None:
    """Download loci."""
    version = "6.8.0"
    base_url = "http://downloads.openmicroscopy.org/bio-formats/"
    jar_name = "loci_tools.jar"
    url = urljoin(base_url, f"{version}/artifacts/{jar_name}")
    path = Path(jar_name)

    loci_tools_content = urllib.request.urlopen(url).read()  # noqa: S310
    sha1_url = url + ".sha1"
    sha1_checksum = (
        urllib.request.urlopen(sha1_url).read().split(b" ")[0].decode()  # noqa: S310
    )
    downloaded_sha1 = hashlib.sha1(loci_tools_content).hexdigest()  # noqa: S324[256 np]
    if downloaded_sha1 != sha1_checksum:
        msg = "Downloaded loci_tools.jar has an invalid checksum. Please try again."
        raise OSError(msg)
    path.write_bytes(loci_tools_content)


def start_jpype(java_memory: str = "512m") -> None:
    """Start the JPype JVM with the specified Java memory.

    Parameters
    ----------
    java_memory : str, optional
        The amount of Java memory to allocate, e.g., "512m" (default is "512m").

    """
    # loci_path = _find_jar()  # Uncomment or adjust as needed
    loci_path = Path("loci_tools.jar")
    if not loci_path.exists():
        print("Downloading loci_tools.jar...")
        download_loci_jar()
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        "-ea",
        f"-Djava.class.path={loci_path}",
        f"-Xmx{java_memory}",
    )
    log4j = jpype.JPackage("org.apache.log4j")
    log4j.BasicConfigurator.configure()
    log4j_logger = log4j.Logger.getRootLogger()
    log4j_logger.setLevel(log4j.Level.ERROR)


def read_jpype(
    filepath: str, java_memory: str = "512m"
) -> tuple[Metadata, ImageReaderWrapper]:
    """Read metadata and data from an image file using JPype.

    Get all OME metadata.

    rdr as a lot of information e.g rdr.isOriginalMetadataPopulated() (core,
    OME, original metadata)

    This function uses JPype to read metadata and data from an image file. It
    returns a dictionary containing tidied metadata and a tuple containing
    JPype objects for the ImageReader, data type, and additional metadata.

    Parameters
    ----------
    filepath : str
        The path to the image file.
    java_memory : str, optional
        The amount of Java memory to allocate (default is "512m").

    Returns
    -------
    md : Metadata
        Tidied metadata.
    wrapper : ImageReaderWrapper
        A wrapper to the Loci image reader; to be used for accessing data from disk.

    Examples
    --------
    We can not start JVM
    >> metadata, jpype_objects = read_jpype("tests/data/LC26GFP_1.tf8")
    >> metadata["SizeX"]
    1600
    >> jpype_objects[1]
    'u2'

    """
    # Start java VM and initialize logger (globally)
    if not jpype.isJVMStarted():
        start_jpype(java_memory)
    if not jpype.isThreadAttachedToJVM():
        jpype.attachThreadToJVM()

    loci = jpype.JPackage("loci")
    # rdr = loci.formats.ChannelSeparator(loci.formats.ChannelFiller())
    rdr = loci.formats.ImageReader()
    rdr.setMetadataStore(loci.formats.MetadataTools.createOMEXMLMetadata())
    rdr.setId(filepath)
    xml_md = rdr.getMetadataStore()
    # sr = image_reader.getSeriesCount()
    md, mdd = get_md_dict(xml_md, Path(filepath).with_suffix(".mmdata.log"))
    core_md = CoreMetadata(rdr)
    return Metadata(core_md, md, mdd), ImageReaderWrapper(rdr)


class FoundMetadataError(Exception):
    """Exception raised when metadata is found during a specific condition."""

    pass


def get_md_dict(
    xml_md: OMEPyramidStore,
    log_fp: None | Path = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse xml_md and return parsed md dictionary and md status dictionary.

    Parameters
    ----------
    xml_md: OMEPyramidStore
        The xml metadata to parse.
    log_fp: None | Path
        The filepath, used for logging JavaExceptions (default=None).

    Returns
    -------
    md: dict[str, Any]
        Parsed metadata dictionary excluding None values.
    mdd: dict[str, str]
        Metadata status dictionary indicating if a value was found ('Found'),
        is None ('None'), or if there was a JavaException ('Jmiss').

    Raises
    ------
    FoundMetadataError:
        If metadata is found during a specific condition.

    """
    key_prefix = "get"
    excluded_methods = {
        "getRoot",
        "getClass",
        "getXMLAnnotationValue",
        "getPixelsBinDataBigEndian",
    }
    keys = [
        m for m in dir(xml_md) if m.startswith(key_prefix) and m not in excluded_methods
    ]
    logging.basicConfig(
        filename=log_fp,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    md = {}
    mdd = {}

    for k in keys:
        try:
            for npar in range(5):
                try:
                    t = (0,) * npar
                    v = getattr(xml_md, k)(*t)
                    raise FoundMetadataError()
                except (TypeError, RuntimeError):
                    continue
        except FoundMetadataError:
            if v is not None:
                # md[k] = [(npar, conversion(v))] # to get only the first value
                md[k[3:]] = get_allvalues_grouped(xml_md, k, npar)
                mdd[k] = "Found"
            else:
                # md[k[3:]] = None
                # md[k[3:]] = get_allvalues_grouped(xml_md, k, npar)
                mdd[k] = "None"
            # keys.remove(k)
        except Exception:
            logging.exception(f"Error processing {k}: {npar}")
            mdd[k] = "Jmiss"
            continue
    return md, mdd


def convert_java_numeric_field(java_field: MDJavaFieldType) -> MDSingleValueType:
    """Convert numeric fields from Java.

    The input `java_field` can be None. It can happen for a list of values that
    doesn't start with None, e.g., (.., ((4, 1), (543.0, 'nm')), ((4, 2), None).

    Parameters
    ----------
    java_field: MDJavaFieldType
        A numeric field from Java.

    Returns
    -------
    MDSingleValueType
        The converted number as int or float types, or None.

    Notes
    -----
    This unconventional conversion is required due to the unreliability of
    `getDouble` and `getFloat`. These methods may introduce precision issues,
    causing values like '0.9' to be inaccurately represented as 0.89999.

    """
    if java_field is None:
        return None
    snum = str(java_field)
    try:
        int(snum)
    except ValueError:
        pass
    else:
        return int(snum)
    try:
        return float(snum)
    except ValueError:
        return snum


def convert_value(
    value: JavaField,
) -> MDSingleValueType | tuple[MDSingleValueType, str]:
    """Convert value from Instance of loci.formats.ome.OMEXMLMetadataImpl."""
    if isinstance(value, str | bool | int):
        return value
    if hasattr(value, "unit") and hasattr(value, "value"):
        return convert_java_numeric_field(value.value()), value.unit().getSymbol()
    try:
        return convert_java_numeric_field(value)
    except Exception as e:
        warnings.warn(
            f"EXCEPTION: {type(e).__name__}: {e}", category=UserWarning, stacklevel=2
        )
        # Re-raise the exception for further analysis
        raise


class StopExceptionError(Exception):
    """Exception raised when need to stop."""

    pass


def next_tuple(llist: list[int], increment_last: bool) -> list[int]:
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
    >>> next_tuple([0, 0, 0], True)
    [0, 0, 1]
    >>> next_tuple([0, 0, 1], True)
    [0, 0, 2]
    >>> next_tuple([0, 0, 2], False)
    [0, 1, 0]
    >>> next_tuple([0, 1, 2], False)
    [0, 2, 0]
    >>> next_tuple([2, 0, 0], False)
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
    # def is_homogenous(lst: list[Any]) -> bool:
    #     return lst.count(lst[0]) == len(lst)

    res: FullMDValueType = []
    tuple_list = [0] * npar
    tuple_pars = tuple(tuple_list)
    value = convert_value(getattr(ome_store, key)(*tuple_pars))
    res.append((tuple_pars, value))
    increment_last = True
    while True:
        try:
            tuple_list = next_tuple(tuple_list, increment_last)
            tuple_pars = tuple(tuple_list)
            value = convert_value(getattr(ome_store, key)(*tuple_pars))
            res.append((tuple_pars, value))
            increment_last = True
        except StopExceptionError:
            break
        except Exception:
            increment_last = False
    # tidy up common metadata
    # TODO Separate into a function to be tested on sample metadata pr what?
    if len(res) > 1:
        values_list = [e[1] for e in res]
        if values_list.count(values_list[0]) == len(res):
            res = [res[-1]]
        elif len(res[0][0]) >= 2:
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
