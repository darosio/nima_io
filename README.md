# NImA-io

[![PyPI](https://img.shields.io/pypi/v/nima_io.svg)](https://pypi.org/project/nima_io/)
[![CI](https://github.com/darosio/nima_io/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/nima_io/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/darosio/nima_io/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/nima_io)
[![RtD](https://readthedocs.org/projects/nima_io/badge/?version=latest)](https://nima-io.readthedocs.io/en/latest/?badge=latest)

<!-- [![RtD](https://readthedocs.org/projects/nima_io/badge/)](https://nima_io.readthedocs.io/) -->

This is a helper library designed for reading microscopy data supported by
[Bioformats](https://www.openmicroscopy.org/bio-formats/) using Python. The
package also includes a command-line interface for assessing differences between
images.

- Version: "0.3.4"

## Installation

You can get the library directly from [PyPI](https://pypi.org/project/nima_io/)
using `pip`:

    pip install nima_io

Alternatively, you can use [pipx](https://pypa.github.io/pipx/) to install it in
an isolated environment:

    pipx install nima_io

To enable auto completion for the `cli` command, follow these steps:

1.  Generate the completion script by running the following command:

        _IMGDIFF_COMPLETE=bash_source imgdiff > ~/.local/bin/imgdiff-complete.bash

2.  Source the generated completion script to enable auto completion:

        source ~/.local/bin/imgdiff-complete.bash

## Usage

You can check out the documentation on <https://darosio.github.io/nima_io> for
up to date usage information and examples.

### CLI

ii provides several command line interface tools for …

    imgdiff --help

### Python

ii can be imported and used as a Python package. The following modules are
available:

    nima_io.read - TODO DESCRIBE

To use nima_io in your python:

    from nima_io import read

## Features / Description

Despite the comprehensive python-bioformats package, Bioformats reading in
Python is not flawless. To assess correct reading and performance, I gathered a
set of test input files from real working data and established various
approaches for reading them:

1. Utilizing the external "showinf" and parsing the generated XML metadata.
2. Employing out-of-the-box python-bioformats.
3. Leveraging bioformats through the Java API.
4. Combining python-bioformats with Java for metadata (Download link: bio-formats 5.9.2).

At present, Solution No. 4 appears to be the most effective.

It's important to note that FEI files are not 100% OME compliant, and
understanding OME metadata can be challenging. For instance, metadata.getXXX is
sometimes equivalent to
metadata.getRoot().getImage(i).getPixels().getPlane(index).

The use of parametrized tests enhances clarity and consistency. The approach of
returning a wrapper to a Bioformats reader enables memory-mapped (a la memmap)
operations.

Notebooks are included in the documentation tutorials to aid development and
illustrate usage. Although there was an initial exploration of the TileStitch
Java class, the decision was made to implement TileStitcher in Python.

Future improvements can be implemented in the code, particularly for the
multichannel OME standard example, which currently lacks obj or resolutionX
metadata. Additionally, support for various instrument, experiment, or plate
metadata can be considered in future updates.

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

All code is licensed under the terms of the [revised BSD license](LICENSE.txt).

## Contributing

If you are interested in contributing to the project, please read our
[contributing](https://darosio.github.io/nima_io/references/contributing.html)
and
[development environment](https://darosio.github.io/nima_io/references/development.html)
guides, which outline the guidelines and conventions that we follow for
contributing code, documentation, and other resources.

### Development

To begin development, follow these steps:

Create an .envrc file with the command:

    echo "layout hatch" > .envrc
    direnv allow

Update and initialize submodules:

    git submodule update --init --recursive

Navigate to the tests/data/ directory:

    cd tests/data/
    git co master

Configure Git Annex for SSH caching:

    git config annex.sshcaching true

Pull the necessary files using Git Annex:

    git annex pull

These commands set up the development environment and fetch the required data for testing.

Modify tests/data.filenames.txt and tests/data.filenames.md5 as needed and run:

    cd tests
    ./data.filenames.sh

### Note

This project was initialized using the [Cookiecutter Python
template](https://github.com/darosio/cookiecutter-python).
