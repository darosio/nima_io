# -*- coding: utf-8 -*-
"""This is the main module of the imgread library to read my microscopy data.

DOC:

exploiting getattr(metadata, key)(\*t)
first try t = () -> process the value and STOP
on TypeError try (0) -> process the value and STOP
on TypeError try (0,0) -> process the value and STOP
on TypeError try (0,0,0) -> process the value and STOP
loop until (0,0,0,0,0).

Values are processed .... MOVE TO THE FUNCTION.

tidy up metadata, group common values makes use of a next funtion
that depends on (tuple, bool).
0,0,0 True
0,0,1 True
0,0,2 False
0,1,0 True
0,1,1 True
0,1,2 False
0,2,0 False
1,0,0 True
...
2,0,0 False -> Raise stopException

what a strange math obj like a set of vector in N^2 + order of creation which
actually depends of a condition defined in the whole space. (and not
necessarily predefined, or yes? -- it should be the same as long as the
condition space is arbirarily defined.)

ricorda che 2500 value con unit, ma alcuni cambiano per lo stesso md key
488 nm 543 nm None
"""
import subprocess

import bioformats
import javabridge
import lxml.etree as etree
import numpy as np

import io
import os
import sys
import tempfile
from contextlib import contextmanager
import collections

__author__ = "daniele arosio"
__copyright__ = "daniele arosio"
__license__ = "new-bsd"

# javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
# javabridge.kill_vm()

# /home/dan/4bioformats/python-microscopy/PYME/IO/DataSources/BioformatsDataSource.py
numVMRefs = 0


def ensure_VM():
    """Start javabridge VM."""
    global numVMRefs
    if numVMRefs < 1:
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        numVMRefs += 1


def release_VM():
    """Kill javabridge VM."""
    global numVMRefs
    numVMRefs -= 1
    if numVMRefs < 1:
        javabridge.kill_vm()


@contextmanager
def stdout_redirector(stream):
    """Context manager to capure fd-level stdout.

    Taken from:
    https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    Alternatively see also the following, but did not work
    https://stackoverflow.com/questions/24277488/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable

    """
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        # Fixed for python3 read() returns byte and not string.
        stream.write(str(tfile.read(), 'utf8'))
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


def init_metadata(series_count, file_format):
    """return an initialized metadata dict.
    Any data file has one file format and contains one or more series.
    Each serie can have different metadata (channels, Z, SizeX, etc.).

    Parameters
    ----------
    series_count : int
        Number of series (stacks, images, ...).
    file_format : str
        File format as a string.

    Returns
    -------
    md : dict
        The key "series" is a list of dictionaries; one for each serie
        (to be filled).

    """
    md = {'SizeS': series_count, 'Format': file_format, 'series': []}
    return md


def fill_metadata(md, sr, root):
    """Works when using (java) root metadata.
    For each serie return a dict with metadata like SizeX, SizeT, etc.

    Parameters
    ----------
    md : dict
        Initialized dict for metadata.
    sr : int
        Number of series (stacks, images, ...).
    root : ome.xml.meta.OMEXMLMetadataRoot
        OME metadata root.

    Returns
    -------
    md : dict
        The key "series" is a list of dictionaries; one for each serie
        (now filled).

    """
    for i in range(sr):
        image = root.getImage(i)
        pixels = image.getPixels()
        try:
            psX = round(float(pixels.getPhysicalSizeX().value()), 6)
        except Exception:
            psX = None
        try:
            psY = round(float(pixels.getPhysicalSizeY().value()), 6)
        except Exception:
            psY = None
        try:
            psZ = round(float(pixels.getPhysicalSizeZ().value()), 6)
        except Exception:
            psZ = None
        try:
            date = image.getAcquisitionDate().getValue()
        except Exception:
            date = None
        try:
            pos = set(
                [(pixels.getPlane(i).getPositionX().value().doubleValue(),
                  pixels.getPlane(i).getPositionY().value().doubleValue(),
                  pixels.getPlane(i).getPositionZ().value().doubleValue())
                 for i in range(pixels.sizeOfPlaneList())])
        except Exception:
            pos = None
        md['series'].append({
            'PhysicalSizeX':
            psX,
            'PhysicalSizeY':
            psY,
            'PhysicalSizeZ':
            psZ,
            'SizeX':
            int(pixels.getSizeX().getValue()),
            'SizeY':
            int(pixels.getSizeY().getValue()),
            'SizeC':
            int(pixels.getSizeC().getValue()),
            'SizeZ':
            int(pixels.getSizeZ().getValue()),
            'SizeT':
            int(pixels.getSizeT().getValue()),
            'Bits':
            int(pixels.getSignificantBits().getValue()),
            'Name':
            image.getName(),
            'Date':
            date,
            'PositionXYZ':
            pos,
        })


def tidy_metadata(md):
    """Moves metadata common to all series into principal keys of the metadata
    dict.

    Parameters
    ----------
    md : dict
        Dict for metadata with all series filled.

    Returns
    -------
    md : dict
        The key "series" is a list of dictionaries containing only metadata
        that are not common among all series. Common metadata are accessible
        as first level keys.

    """
    if len(md['series']) == 1:
        d = md['series'][0]
        while len(d):
            k, v = d.popitem()
            md[k] = v
        md.pop('series')
    else:
        assert len(md['series']) > 1
        keys_samevalue = []
        for k in md['series'][0].keys():
            ll = [d[k] for d in md['series']]
            if ll.count(ll[0]) == len(ll):
                keys_samevalue.append(k)
        for k in keys_samevalue:
            for d in md['series']:
                val = d.pop(k)
            md[k] = val


def read_inf(filepath):
    """ Using external showinf.

    Parameters
    ----------
    filepath : path
        File to be parsed.

    Returns
    -------
    md : dict
        Tidied metadata.

    Notes
    -----
    10-40 times slower than all others

    References
    ----------
    http://bioimage-analysis.stanford.edu/guides/3-Loading_microscopy_images/

    """
    # first run to get number of images (i.e. series)
    inf0 = ['showinf', '-nopix', filepath]
    p0 = subprocess.Popen(inf0, stdout=subprocess.PIPE)
    a0 = subprocess.check_output(
        ('grep', '-E', 'Series count|file format'), stdin=p0.stdout)
    for l in a0.decode('utf8').splitlines():
        if 'file format' in l:
            ff = l.rstrip(']').split('[')[1]
        if 'Series count' in l:
            sr = int(l.split('=')[1])
    md = init_metadata(sr, ff)
    # second run for xml metadata
    inf = ['showinf', "-nopix", "-omexml-only", filepath]
    p = subprocess.Popen(inf, stdout=subprocess.PIPE)
    stdout = p.communicate()[0]
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(stdout, parser)
    for child in tree:
        if child.tag.endswith('Image'):
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    att = grandchild.attrib
                    try:
                        psX = round(float(att['PhysicalSizeX']), 6)
                    except Exception:
                        psX = None
                    try:
                        psY = round(float(att['PhysicalSizeY']), 6)
                    except Exception:
                        psY = None
                    try:
                        psZ = round(float(att['PhysicalSizeZ']), 6)
                    except Exception:
                        psZ = None
                    try:
                        psXu = att['PhysicalSizeXUnit']
                    except Exception:
                        psXu = None
                    try:
                        psYu = att['PhysicalSizeYUnit']
                    except Exception:
                        psYu = None
                    try:
                        psZu = att['PhysicalSizeZUnit']
                    except Exception:
                        psZu = None
                    md['series'].append({
                        'PhysicalSizeX': psX,
                        'PhysicalSizeY': psY,
                        'PhysicalSizeZ': psZ,
                        'PhysicalSizeXUnit': psXu,
                        'PhysicalSizeYUnit': psYu,
                        'PhysicalSizeZUnit': psZu,
                        'SizeX': int(att['SizeX']),
                        'SizeY': int(att['SizeY']),
                        'SizeC': int(att['SizeC']),
                        'SizeZ': int(att['SizeZ']),
                        'SizeT': int(att['SizeT']),
                        'Bits': int(att['SignificantBits'])
                    })
        elif child.tag.endswith('Instrument'):
            for grandchild in child:
                if grandchild.tag.endswith('Objective'):
                    att = grandchild.attrib
                    Obj = att['Model']
    tidy_metadata(md)
    if 'Obj' in locals():
        md['Obj'] = Obj
    return md


def read_bf(filepath):
    """Using standard bioformats instruction; fails with FEITiff.

    Parameters
    ----------
    filepath : path
        File to be parsed.

    Returns
    -------
    md : dict
        Tidied metadata.

    Notes
    -----
    In this approach the reader reports the last Pixels read (e.g. z=37),
    dimensionorder ...

    """
    omexmlstr = bioformats.get_omexml_metadata(filepath)
    o = bioformats.omexml.OMEXML(omexmlstr)
    sr = o.get_image_count()
    md = init_metadata(sr, "ff")
    for i in range(sr):
        md['series'].append({
            'PhysicalSizeX':
            round(o.image(i).Pixels.PhysicalSizeX, 6)
            if o.image(i).Pixels.PhysicalSizeX else None,
            'PhysicalSizeY':
            round(o.image(i).Pixels.PhysicalSizeY, 6)
            if o.image(i).Pixels.PhysicalSizeY else None,
            'SizeX':
            o.image(i).Pixels.SizeX,
            'SizeY':
            o.image(i).Pixels.SizeY,
            'SizeC':
            o.image(i).Pixels.SizeC,
            'SizeZ':
            o.image(i).Pixels.SizeZ,
            'SizeT':
            o.image(i).Pixels.SizeT,
        })
    tidy_metadata(md)
    return md


def read_jb(filepath):
    """Using java directly to access metadata.

    Parameters
    ----------
    filepath : path
        File to be parsed.

    Returns
    -------
    md : dict
        Tidied metadata.

    References
    ----------
    Following suggestions at:
    https://github.com/CellProfiler/python-bioformats/issues/23

    """
    rdr = javabridge.JClassWrapper('loci.formats.in.OMETiffReader')()
    rdr.setOriginalMetadataPopulated(True)
    clsOMEXMLService = javabridge.JClassWrapper(
        'loci.formats.services.OMEXMLService')
    serviceFactory = javabridge.JClassWrapper(
        'loci.common.services.ServiceFactory')()
    service = serviceFactory.getInstance(clsOMEXMLService.klass)
    metadata = service.createOMEXMLMetadata()
    rdr.setMetadataStore(metadata)
    rdr.setId(filepath)
    sr = rdr.getSeriesCount()
    root = metadata.getRoot()
    md = init_metadata(sr, rdr.getFormat())
    fill_metadata(md, sr, root)
    tidy_metadata(md)
    return md


def read(filepath):
    """Read a data file picking the correct Format and metadata (e.g. channels,
    time points, ...).

    It uses java directly to access metadata, but the reader is picked by
    loci.formats.ImageReader.

    Parameters
    ----------
    filepath : path
        File to be parsed.

    Returns
    -------
    md : dict
        Tidied metadata.
    wrapper : bioformats.formatreader.ImageReader
        A wrapper to the Loci image reader; to be used for accessing data from
        disk.

    Examples
    --------
    >>> javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
    >>> md, wr = read('../tests/data/multi-channel-time-series.ome.tif')
    >>> md['SizeC'], md['SizeT'], md['SizeX'], md['Format'], md['Bits']
    (3, 7, 439, 'OME-TIFF', 8)
    >>> a = wr.read(c=2, t=6, series=0, z=0, rescale=False)
    >>> a[20,200]
    -1

    """
    image_reader = bioformats.formatreader.make_image_reader_class()()
    image_reader.allowOpenToCheckType(True)
    # metadata a la java
    clsOMEXMLService = javabridge.JClassWrapper(
        'loci.formats.services.OMEXMLService')
    serviceFactory = javabridge.JClassWrapper(
        'loci.common.services.ServiceFactory')()
    service = serviceFactory.getInstance(clsOMEXMLService.klass)
    metadata = service.createOMEXMLMetadata()
    image_reader.setMetadataStore(metadata)
    image_reader.setId(filepath)
    sr = image_reader.getSeriesCount()
    # n_t = image_reader.getSizeT() remember it refers to pixs of first serie
    root = metadata.getRoot()
    md = init_metadata(sr, image_reader.getFormat())
    fill_metadata(md, sr, root)
    tidy_metadata(md)
    # Make a fake ImageReader and install the one above inside it
    wrapper = bioformats.formatreader.ImageReader(
        path=filepath, perform_init=False)
    wrapper.rdr = image_reader
    return md, wrapper


def read_wrap(filepath, logpath="bioformats.log"):
    """wrap for read function; capture standard output.
    """
    f = io.StringIO()
    with stdout_redirector(f):
        md, wr = read(filepath)
    out = f.getvalue()
    with open(logpath, 'a') as f:
        f.write("\n\nreading " + filepath + "\n")
        f.write(out)
    return md, wr


def stitch(md, wrapper, c=0, t=0, z=0):
    """Stitch image tiles returning a tiled single plane."""
    xyz_list_of_sets = [p['PositionXYZ'] for p in md['series']]
    try:
        assert all([len(p) == 1 for p in xyz_list_of_sets])
    except Exception:
        raise Exception(
            "One or more series doesn't have a single XYZ position.")
    xy_positions = [list(p)[0][:2] for p in xyz_list_of_sets]
    unique_x = np.sort(list(set([xy[0] for xy in xy_positions])))
    unique_y = np.sort(list(set([xy[1] for xy in xy_positions])))
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
                raise IndexError(
                    "Building tilemap failed in searching xy_position indexes."
                )
    F = np.zeros((md['SizeY'] * tiley, md['SizeX'] * tilex))
    for yt in range(tiley):
        for xt in range(tilex):
            if tilemap[yt, xt] >= 0:
                F[yt * md['SizeY']: (yt+1) * md['SizeY'],
                  xt * md['SizeX']: (xt+1) * md['SizeX']] = \
                                               wrapper.read(c=c,
                                                            t=t,
                                                            z=z,
                                                            series=tilemap[yt,
                                                                           xt],
                                                            rescale=False)
    return F


def diff(fp_a, fp_b):
    """Diff for two image data.

    Returns
    -------
    Bool: True if the two files are equal

    """
    md_a, wr_a = read(fp_a)
    md_b, wr_b = read(fp_b)
    are_equal = True
    are_equal = are_equal & (md_a == md_b)
    # print(md_b) maybe return md_a and different md_b TODO
    if are_equal:
        for s in range(md_a['SizeS']):
            for t in range(md_a['SizeT']):
                for c in range(md_a['SizeC']):
                    for z in range(md_a['SizeZ']):
                        are_equal = are_equal & np.array_equal(
                            wr_a.read(series=s, t=t, c=c, z=z, rescale=False),
                            wr_b.read(series=s, t=t, c=c, z=z, rescale=False))
    return are_equal


def first_nonzero_reverse(l):
    """Return the index of the first nonzero element of a list from the last
    element and moving backward.

    Examples
    --------
    >>> first_nonzero_reverse([0, 2, 0, 0])
    >>> -3

    """
    for i in range(-1, -len(l) - 1, -1):
        if l[i] != 0:
            return i


def img_reader(filepath):

    image_reader = bioformats.formatreader.make_image_reader_class()()
    image_reader.allowOpenToCheckType(True)
    # metadata a la java
    clsOMEXMLService = javabridge.JClassWrapper(
        'loci.formats.services.OMEXMLService')
    serviceFactory = javabridge.JClassWrapper(
        'loci.common.services.ServiceFactory')()
    service = serviceFactory.getInstance(clsOMEXMLService.klass)
    xml_md = service.createOMEXMLMetadata()
    image_reader.setMetadataStore(xml_md)
    image_reader.setId(filepath)
    return image_reader, xml_md


def read2(filepath, mdd_wanted=False):
    """Read a data file picking the correct Format and metadata (e.g. channels,
    time points, ...).

    It uses java directly to access metadata, but the reader is picked by
    loci.formats.ImageReader.

    Parameters
    ----------
    filepath : path
        File to be parsed.

    Returns
    -------
    md : dict
        Tidied metadata.
    wrapper : bioformats.formatreader.ImageReader
        A wrapper to the Loci image reader; to be used for accessing data from
        disk.

    Examples
    --------
    >>> javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
    >>> md, wr = read('../tests/data/multi-channel-time-series.ome.tif')
    >>> md['SizeC'], md['SizeT'], md['SizeX'], md['Format'], md['Bits']
    (3, 7, 439, 'OME-TIFF', 8)
    >>> a = wr.read(c=2, t=6, series=0, z=0, rescale=False)
    >>> a[20,200]
    -1

    """

    image_reader, xml_md = img_reader(filepath)
    # sr = image_reader.getSeriesCount()
    md, mdd = get_md_dict(xml_md, filepath)
    md['Format'] = image_reader.getFormat()
    # Make a fake ImageReader and install the one above inside it
    wrapper = bioformats.formatreader.ImageReader(
        path=filepath, perform_init=False)
    wrapper.rdr = image_reader
    if mdd_wanted:
        return md, wrapper, mdd
    else:
        return md, wrapper


class FoundMetadata(Exception):
    pass


def get_md_dict(xml_md, filepath=None):
    """Parse xml_md and return md{} and list of missing and None keys.
    keys list return only missing (JavaException) values.
    md dict keys exclune None values.

    """
    keys = [
        m for m in xml_md.methods if m[:3] == 'get' and not (
            m == 'getRoot' or m == 'getClass' or m == 'getXMLAnnotationValue')
    ]
    md = {}
    mdd = {}
    if filepath:
        javaexception_logfile = open(filepath+".mmdata.log", "w")
    for k in keys:
        try:
            for npar in range(5):
                try:
                    t = (0, ) * npar
                    v = getattr(xml_md, k)(*t)
                    raise FoundMetadata()
                except TypeError:
                    continue
        except FoundMetadata:
            if v is not None:
                # md[k] = [(npar, convertion(v))] # to get only the first value
                md[k[3:]] = get_allvalues_grouped(xml_md, k, npar)
                mdd[k] = "Found"
            else:
                # md[k[3:]] = None
                # md[k[3:]] = get_allvalues_grouped(xml_md, k, npar)
                mdd[k] = "None"
            #keys.remove(k)
        except Exception as e:
            if filepath:
                javaexception_logfile.write(
                    str((k, type(e), e, "--", npar)) + '\n')
            mdd[k] = "Jmiss"
            continue
    if filepath:
        javaexception_logfile.close()
    return md, mdd


def _convert_num(num):
    """Convert numeric fields.

    num can also be None. It can happen for a list of values that doesn't start
    with None e.g. (.., ((4, 1), (543.0, 'nm')), ((4, 2), None)

    Param
    -----
    num a numeric field from java

    Return
    ------
    number as int ot float types or None.

    Raise
    -----
    on non numeric input

    number -> str -> int or float

    This is necessary because getDouble, getFloat are not
    reliable ('0.9' become 0.89999).

    """
    if num is None:
        return
    snum = str(num)
    try:
        return int(snum)
    except ValueError:
        try:
            return float(snum)
        except ValueError as e:
            print("Neither int nor float value to convert {}.".format(num))
            raise e


def convert_value(v, debug=False):
    """Convert value from Instance of loci.formats.ome.OMEXMLMetadataImpl."""
    if type(v) in [str, bool, int]:
        md2 = v, type(v), "v"
    elif hasattr(v, "getValue"):
        vv = v.getValue()
        if type(vv) in [str, bool, int, float]:
            md2 = vv, type(vv), "gV"
        else:
            vv = _convert_num(vv)
            md2 = vv, type(vv), "gVc"
    elif hasattr(v, "unit"):
        # this conversion is better than using stringIO
        vv = _convert_num(v.value()), v.unit().getSymbol()
        md2 = vv, type(vv), "unit"
    else:
        try:
            vv = _convert_num(v)
            md2 = vv, type(vv), "c"
        except ValueError:
            # print(k, v, 'unknown type') TODO: use a warn
            md2 = v, 'unknown', "un"
        except Exception as e:
            print("EXCEPTION ", e)  # should never happen
            raise Exception
    if debug:
        return md2
    else:
        return md2[0]


class stopException(Exception):
    pass


def next_tuple(l, s):
    # next item never exists for empty tuple.
    if len(l) == 0:
        raise stopException
    if s:
        l[-1] += 1
    else:
        idx = first_nonzero_reverse(l)
        if idx == -len(l):
            raise stopException
        else:
            l[idx] = 0
            l[idx - 1] += 1
    return l


def get_allvalues_grouped(metadata, k, npar):
    res = []
    ll = [0] * npar
    t = tuple(ll)
    v = convert_value(getattr(metadata, k)(*t))
    res.append((t, v))
    s = True
    while True:
        try:
            ll = next_tuple(ll, s)
            t = tuple(ll)
            v = convert_value(getattr(metadata, k)(*t))
            res.append((t, v))
            s = True
        except stopException:
            break
        except Exception:
            s = False

    # tidy up common metadata
    # TODO Separate into a function to be tested on sample metadata pr what?
    if len(res) > 1:
        ll = [e[1] for e in res]
        if ll.count(ll[0]) == len(res):
            res = [res[-1]]
        elif len(res[0][0]) >= 2:
            # first group the list of tuples by (tuple_idx=0)
            grouped_res = collections.defaultdict(list)
            for t, v in res:
                grouped_res[t[0]].append(v)
            max_key = max(grouped_res.keys())  # or: res[-1][0][0]
            # now check for single common value within a group
            new_res = []
            for k, v in grouped_res.items():
                if v.count(v[0]) == len(v):
                    new_res.append(((k, len(v) - 1), v[-1]))
            if new_res:
                res = new_res
            # now check for the same group repeated
            try:
                for k, v in grouped_res.items():
                    assert v == grouped_res[max_key]
                res = res[-len(v):]
            except Exception:
                pass

    return res
