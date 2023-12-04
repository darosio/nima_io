nima_io Module
==============

This module is the main component of the nima_io library, designed for reading
microscopy data. It provides functionality to explore metadata, process values,
and extract information from bioformats core metadata.

Metadata Exploration
--------------------

The exploration of metadata is facilitated using the ``getattr(metadata,
key)(*t)`` approach:

- First, try ``t=()`` to process the value and stop.
- On TypeError, try ``(0)`` to process the value and stop.
- On TypeError, try ``(0,0)`` to process the value and stop.
- On TypeError, try ``(0,0,0)`` to process the value and stop.
- Continue looping until ``(0,0,0,0,0)``. Raise RuntimeError for using jpype.

Tidying Metadata
----------------

The metadata tidying process involves grouping common values and utilizes a next function dependent on ``(tuple, bool)``.

Bioformats Core Metadata
------------------------

The module extracts essential information from bioformats core metadata, including:
- ``SizeS: rdr.getSeriesCount()`` - may vary for each series.
- ``SizeX: rdr.getSizeX()``
- ``SizeY: rdr.getSizeY()``
- ``SizeZ: rdr.getSizeZ()``
- ``SizeT: rdr.getSizeT()``
- ``SizeC: rdr.getSizeC()``
- ... (additional core metadata)

Additional Information
----------------------

In addition to core metadata, the module provides access to the following information:
- ``Format``: File format of the opened file.
- ``Date``: Date information.
- ``Series Name``: Name of the series.

Physical Metadata
-----------------

For each series, the module extracts physical metadata:
- ``PositionXYZ``: Physical position (x_um, y_um, and z_um).
- ``PhysicalSizeX``: Physical size in the X dimension [PhysicalSizeXUnit].
- ``PhysicalSizeY``: Physical size in the Y dimension [PhysicalSizeYUnit].
- ``PhysicalSizeZ``: Physical size in the Z dimension [PhysicalSizeZUnit].
- ``t_s``: Time information.

Note: Ensure that the provided information is adjusted based on the specific implementation details of the module.


This is the main module of the nima_io library to read my microscopy data.

DOC:

exploiting getattr(metadata, key)(\\*t)
first try t = () -> process the value and STOP
on TypeError try (0) -> process the value and STOP
on TypeError try (0,0) -> process the value and STOP
on TypeError try (0,0,0) -> process the value and STOP
loop until (0,0,0,0,0). RuntimeError for using jpype.

Values are processed .... MOVE TO THE FUNCTION.

tidy up metadata, group common values makes use of a next function
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
condition space is arbitrarily defined.)

ricorda che 2500 value con unit, ma alcuni cambiano per lo stesso md key
488 nm 543 nm None

Model:
A file contains:
- 1* series
- Pixels
- Planes

cfr. FrameSequences of Pims where a frame is nDim and each Frame contains 1*
frame==plane

Bioformats core metadata:
- SizeS; rdr.getSeriesCount() -- could be different for each series --

  - ; rdr.getImageCount()
  - SizeX; rdr.getSizeX()
  - SizeY; rdr.getSizeY()
  - SizeZ; rdr.getSizeZ()
  - SizeT; rdr.getSizeT()
  - SizeC; rdr.getSizeC()
  - ; rdr.getDimensionOrder()
  - ; rdr.getRGBChannelCount()
  - ; rdr.isRGB()
  - ; rdr.isInterleaved()
  - ; rdr.getPixelType()

I would add:
- Format (for the file opened)
- date
- series name
and most importantly physical metadata for each series:
- PositionXYZ (x_um, y_um and z_um)
- PhysicalSizeX [PhysicalSizeXUnit]
- PhysicalSizeY [PhysicalSizeYUnit]
- PhysicalSizeZ [PhysicalSizeZUnit]

- t_s

I would also add objective: NA, Xmag and immersion
as well as PlaneExposure
when reading a plane (a la memmap) can check TheC, TheT, TheZ ....

Probably a good choice can be a vector, but TODO: think to tiles, lif, ...
