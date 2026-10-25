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
- Pre-commit hooks are grouped and labeled.

Python upgrades are **not** managed by Renovate. Bumping `.python-version`
on its own would make local dev and the release workflow use an interpreter
that CI never tests. Upgrade Python in a single manual PR that updates
`.python-version`, the `requires-python`/classifier/ruff settings in
`pyproject.toml`, the CI matrix in `.github/workflows/ci.yml` and the version
in `.readthedocs.yml`, then refresh `uv.lock`.

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

### Enabling the CI push (GitHub App, one-time per repo)

The `cruft-update` workflow needs write access to push its update branch and
open a PR, and `github.token` alone cannot push changes under
`.github/workflows/*`. Rather than a personal access token (which expires and
must be rotated), this template uses a GitHub App installation token via
[`actions/create-github-app-token`](https://github.com/actions/create-github-app-token).
The workflow falls back to `github.token` automatically if the App isn't
configured, but that fallback fails whenever Cruft touches workflow files.

One-time setup (shared across all your repos once the App exists):

1. Create the App (skip if you already have one for this purpose):
   [github.com/settings/apps/new](https://github.com/settings/apps/new) —
   Repository permissions: `Contents: Read & write`,
   `Pull requests: Read & write`, `Workflows: Read & write`. Webhook: inactive.

1. Generate a private key on the App's settings page and download the `.pem`.

1. Install the App on this repository (or "All repositories" on your account
   to cover future repos too).

1. Set two **repo-level** secrets (GitHub Apps on personal accounts don't
   support org-wide shared secrets, so this step is repeated per repo):

   ```bash
   gh secret set APP_ID --repo <owner>/<repo> --body "<app-id>"
   gh secret set APP_PRIVATE_KEY --repo <owner>/<repo> < /path/to/key.pem
   ```

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
