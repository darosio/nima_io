# -*- coding: utf-8 -*-
"""
This is the main module of the imgread library to read my microscopy data.

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


import ctypes
# FIXME libc
libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

__author__ = "daniele arosio"
__copyright__ = "daniele arosio"
__license__ = "new-bsd"

# javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
# javabridge.kill_vm()

# /home/dan/4bioformats/python-microscopy/PYME/IO/DataSources/BioformatsDataSource.py
numVMRefs = 0


def ensure_VM():
    global numVMRefs
    if numVMRefs < 1:
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        numVMRefs += 1


def release_VM():
    global numVMRefs
    numVMRefs -= 1
    if numVMRefs < 1:
        javabridge.kill_vm()


# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
# alternatively see also the following, but did not work
# https://stackoverflow.com/questions/24277488/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable
@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        # libc.fflush(c_stdout)  # FIXME maybe I do not need
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
    """ Using external showinf; 10-40 times slower than all others
    http://bioimage-analysis.stanford.edu/guides/3-Loading_microscopy_images/

    Parameters
    ----------
    filepath : path
        File to be parsed.

    Returns
    -------
    md : dict
        Tidied metadata.

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
    "Stitch a tiled image. Return a single plane"
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
