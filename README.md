# NImA-io

[![PyPI](https://img.shields.io/pypi/v/nima_io.svg)](https://pypi.org/project/nima_io/)
[![CI](https://github.com/darosio/nima_io/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/nima_io/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/darosio/nima_io/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/nima_io)
[![](https://img.shields.io/badge/Pages-blue?logo=github)](https://darosio.github.io/nima_io/)

This is a helper library designed for reading microscopy data supported by
[Bioformats](https://www.openmicroscopy.org/bio-formats/) using Python. The
package also includes a command-line interface for assessing differences between
images.

## Features / Description

Despite the comprehensive python-bioformats package, Bioformats reading in
Python is not flawless. To assess correct reading and performance, I gathered a
set of test input files from real working data and established various
approaches for reading them:

1. Utilizing the external "showinf" and parsing the generated XML metadata.
1. Employing out-of-the-box python-bioformats.
1. Leveraging bioformats through the Java API.
1. Combining python-bioformats with Java for metadata (Download link: bio-formats 5.9.2).

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

## Installation

System requirements:

- maven

### From PyPI

Using pip:

```
pip install nima_io
```

### Recommended: Using pipx

For isolated installation (recommended):

```
pipx install nima_io
```

### Shell Completion

#### Bash

```bash
_IMGDIFF_COMPLETE=bash_source imgdiff > ~/.local/bin/imgdiff-complete.bash
source ~/.local/bin/imgdiff-complete.bash
# Add to your ~/.bashrc to make it permanent:
echo 'source ~/.local/bin/ingdiff-complete.bash' >> ~/.bashrc
```

#### Fish:

```bash
_IMGDIFF_COMPLETE=fish_source imgdiff | source
# Add to fish config to make it permanent:
_IMGDIFF_COMPLETE=fish_source imgdiff > ~/.config/fish/completions/imgdiff.fish

```

## Usage

Docs: https://{{ cookiecutter.project_slug }}.readthedocs.io/

### CLI

```bash
imgdiff --help
```

### Python

```python
from nima_io import read
```

## Development

Requires Python `uv`.

With uv:

```bash
# one-time
pre-commit install
# dev tools and deps
uv sync --group dev
# lint/test
uv run ruff check .  (or: make lint)
uv run pytest -q  (or: make test)
```

### Update and initialize submodules

```
git submodule update --init --recursive
```

Navigate to the tests/data/ directory:

```
cd tests/data/
git co master
```

Configure Git Annex for SSH caching:

```
git config annex.sshcaching true
```

Pull the necessary files using Git Annex:

```
git annex pull
```

These commands set up the development environment and fetch the required data for testing.

Modify tests/data.filenames.txt and tests/data.filenames.md5 as needed and run:

```
cd tests
./data.filenames.sh
```

We use Renovate to keep dependencies current.

## Dependency updates (Renovate)

Enable Renovate:

1. Install the GitHub App: https://github.com/apps/renovate (Settings → Integrations → GitHub Apps → Configure → select this repo/org).
1. This repo includes a `renovate.json` policy. Renovate will open a “Dependency Dashboard” issue and PRs accordingly.

Notes:

- Commit style: `build(deps): bump <dep> from <old> to <new>`
- Pre-commit hooks are grouped and labeled; Python version bumps in `pyproject.toml` are disabled by policy.

Migrating from Dependabot:

- You may keep “Dependabot alerts” ON for vulnerability visibility, but disable Dependabot security PRs.

## Template updates (Cruft)

This project is linked to its Cookiecutter template with Cruft.

- Check for updates: `cruft check`
- Apply updates: `cruft update -y` (resolve conflicts, then commit)

CI runs a weekly job to open a PR when template updates are available.

First-time setup if you didn’t generate with Cruft:

```bash
pipx install cruft  # or: pip install --user cruft
cruft link --checkout main https://github.com/darosio/cookiecutter-python.git
```

Notes:

- The CI workflow skips if `.cruft.json` is absent.
- If you maintain a stable template branch (e.g., `v1`), link with `--checkout v1`. You can also update within that line using `cruft update -y --checkout v1`.

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
