=======
imgread
=======


This is an helper library to read microscopy data supported by bioformats using
python.


Description
===========

Bioformats reading in python is not perfect despite the tremendous
python-bioformats package.

In order to compare correct reading and performance, I collected few test input
files from real working data and setup various approaches for reading them:
1. using external "showinf" and parsing generated xml metadata
2. using out-of-the-box python-bioformats
3. using bioformats thought java API
4. using python-bioformats mixed to java for metadata
http://downloads.openmicroscopy.org/bio-formats/5.9.2/artifacts/

Solution n.4 seems the best at this moment.



FEI files are not 100% OME compliant. 

OME metadata are not easy to understand:
metadata.getXXX is sometime equivalent to metadata.getRoot.getImage(i),getPixels().getPlane(index)

Using parametrized tests improves clarity and consistency.

By returning a wrapper to a bioformats reader, it works a la memmap.

Notebooks to help development and illustrate usage are in the examples folder.

Tried to explore TileStitch java class, but decided to implement TileStitcher in python.

Improvements can be implemented in the code for the multichannel ome standard example,
which does not have obj or resolutionX metadata. Many other instrument, experiment or
plate metadata can be supported in the future.



Note
====

This project has been set up using PyScaffold 3.0. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
