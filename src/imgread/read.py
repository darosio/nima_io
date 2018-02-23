# -*- coding: utf-8 -*-
"""
This is the main module of the imgread library to read my microscopy data.

"""
from imgread import __version__

import subprocess
import lxml.etree as etree
import numpy as np

import javabridge
import bioformats

__author__ = "daniele arosio"
__copyright__ = "daniele arosio"
__license__ = "new-bsd"

javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
# javabridge.kill_vm()


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
        The key "series" is a list of dictionaries; one for each serie (to be filled).

    """
    md = {'SizeS' : series_count,
          'Format' : file_format,
          'series' : []}
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
        The key "series" is a list of dictionaries; one for each serie (now filled).

    """
    for i in range(sr) :
        image = root.getImage(i)
        pixels = image.getPixels()
        try:
            psX = round(float(pixels.getPhysicalSizeX().value()), 6)
        except:
            psX = None
        try:
            psY = round(float(pixels.getPhysicalSizeY().value()), 6)
        except:
            psY = None
        try:
            psZ = round(float(pixels.getPhysicalSizeZ().value()), 6)
        except:
            psZ = None
        try:
            date = image.getAcquisitionDate().getValue()
        except:
            date = None
        md['series'].append(
            {'PhysicalSizeX' : psX,
             'PhysicalSizeY' : psY,
             'PhysicalSizeZ' : psZ,
             'SizeX' : int(pixels.getSizeX().getValue()),
             'SizeY' : int(pixels.getSizeY().getValue()),
             'SizeC' : int(pixels.getSizeC().getValue()),
             'SizeZ' : int(pixels.getSizeZ().getValue()),
             'SizeT' : int(pixels.getSizeT().getValue()),
             'Bits' : int(pixels.getSignificantBits().getValue()),
             'Name' : image.getName(),
             'Date' : date,
             'PositionXYZ' : set([(pixels.getPlane(i).getPositionX().value().doubleValue(),
                                   pixels.getPlane(i).getPositionY().value().doubleValue(),
                                   pixels.getPlane(i).getPositionZ().value().doubleValue()) for i in range(pixels.sizeOfPlaneList())]),
            }
        )


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
        The key "series" is a list of dictionaries containing only metadata that are not
        common among all series. Common metadata are accessible as first level keys.

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
    a0 = subprocess.check_output(('grep', '-E', 'Series count|file format'), stdin=p0.stdout)
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
    tree   = etree.fromstring(stdout, parser)
    for child in tree:
        if child.tag.endswith('Image'):
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    att = grandchild.attrib
                    try:
                        psX = round(float(att['PhysicalSizeX']), 6)
                    except:
                        psX = None
                    try:
                        psY = round(float(att['PhysicalSizeY']), 6)
                    except:
                        psY = None
                    try:
                        psZ = round(float(att['PhysicalSizeZ']), 6)
                    except:
                        psZ = None
                    try:
                        psXu = att['PhysicalSizeXUnit']
                    except:
                        psXu = None
                    try:
                        psYu = att['PhysicalSizeYUnit']
                    except:
                        psYu = None
                    try:
                        psZu = att['PhysicalSizeZUnit']
                    except:
                        psZu = None
                    md['series'].append(
                        {'PhysicalSizeX' : psX,
                         'PhysicalSizeY' : psY,
                         'PhysicalSizeZ' : psZ,
                         'PhysicalSizeXUnit' : psXu,
                         'PhysicalSizeYUnit' : psYu,
                         'PhysicalSizeZUnit' : psZu,
                         'SizeX' : int(att['SizeX']),
                         'SizeY' : int(att['SizeY']),
                         'SizeC' : int(att['SizeC']),
                         'SizeZ' : int(att['SizeZ']),
                         'SizeT' : int(att['SizeT']),
                         'Bits' : int(att['SignificantBits'])
                        }
                    )
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
        md['series'].append(
            {'PhysicalSizeX' : round(o.image(i).Pixels.PhysicalSizeX, 6) if o.image(i).Pixels.PhysicalSizeX else None,
             'PhysicalSizeY' : round(o.image(i).Pixels.PhysicalSizeY, 6) if o.image(i).Pixels.PhysicalSizeY else None,
             'SizeX' : o.image(i).Pixels.SizeX,
             'SizeY' : o.image(i).Pixels.SizeY,
             'SizeC' : o.image(i).Pixels.SizeC,
             'SizeZ' : o.image(i).Pixels.SizeZ,
             'SizeT' : o.image(i).Pixels.SizeT,
            }
        )
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
    clsOMEXMLService = javabridge.JClassWrapper('loci.formats.services.OMEXMLService')
    serviceFactory = javabridge.JClassWrapper('loci.common.services.ServiceFactory')()
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
        A wrapper to the Loci image reader; to be used for accessing data from disk.

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
    clsOMEXMLService = javabridge.JClassWrapper('loci.formats.services.OMEXMLService')
    serviceFactory = javabridge.JClassWrapper('loci.common.services.ServiceFactory')()
    service = serviceFactory.getInstance(clsOMEXMLService.klass)
    metadata = service.createOMEXMLMetadata()
    image_reader.setMetadataStore(metadata)
    image_reader.setId(filepath)
    sr = image_reader.getSeriesCount()
    # n_t = image_reader.getSizeT() to remember: it refers to first serie pixels
    root = metadata.getRoot()
    md = init_metadata(sr, image_reader.getFormat())
    fill_metadata(md, sr, root)
    tidy_metadata(md)
    # Make a fake ImageReader and install the one above inside it
    wrapper = bioformats.formatreader.ImageReader(path=filepath, perform_init=False)
    wrapper.rdr = image_reader
    return md, wrapper



def stitch(md, wrapper, c=0, t=0, z=0):
    "Stitch a tiled image. Return a single plane"
    xyz_list_of_sets = [p['PositionXYZ'] for p in md['series']]
    try:
        assert all([len(p) == 1 for p in xyz_list_of_sets])
    except:
        raise Exception("One or more series doesn't have a single XYZ position.")
    xy_positions = [list(p)[0][:2] for p in xyz_list_of_sets]
    unique_x = np.sort(list(set([xy[0] for xy in xy_positions])))
    unique_y = np.sort(list(set([xy[1] for xy in xy_positions])))
    tiley = len(unique_y)
    tilex = len(unique_x)
    # tilemap only for complete tiles without None tile
    tilemap = np.zeros(shape=(tiley, tilex), dtype=int)
    for yi, y in enumerate(unique_y):
        for xi, x in enumerate(unique_x):
            indexes = [i for i,v in enumerate(xy_positions) if v == (x, y)]
            li = len(indexes)
            if li == 0: 
                tilemap[yi, xi] = -1
            elif li == 1:
                tilemap[yi, xi] = indexes[0]
            else:
                raise IndexError("Building tilemap failed in searching xy_positions indexes.")
    F = np.zeros((md['SizeY'] * tiley, md['SizeX'] * tilex))
    for yt in range(tiley):
        for xt in range(tilex):
            if tilemap[yt, xt] >= 0:
                F[yt * md['SizeY'] : (yt+1) * md['SizeY'], 
                  xt * md['SizeX'] : (xt+1) * md['SizeX']] = \
                                               wrapper.read(c=c,
                                                            t=t,
                                                            z=z,
                                                            series=tilemap[yt, xt],
                                                            rescale=False)
    return F

