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

  - ; rdr.getImageCount()
  - ; rdr.getDimensionOrder()
  - ; rdr.getRGBChannelCount()
  - ; rdr.isRGB()
  - ; rdr.isInterleaved()
  - ; rdr.getPixelType()

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

TODO
----
I would also add:

- objective: NA, Xmag, and immersion;
- PlaneExposure.

DOC:

Keep in mind that there are 2500 values with units, and some may change for the
same metadata key (e.g., 488 nm, 543 nm, None).

Model:
A file comprises:

- 1* series
- Pixels
- Planes

Refer to FrameSequences of Pims, where a frame is nDim, and each frame contains
1* plane (frame == plane).

When reading a plane (similar to memmap), it is possible to check TheC, TheT,
TheZ.

It might be beneficial to consider using a vector in **Dask**, but further
exploration is needed, especially regarding tiles, lif, and other formats.
