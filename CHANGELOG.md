<!-- markdownlint-disable MD024 -->
<!-- vale write-good.TooWordy = NO -->

# Changelog

## v0.2.0 (2023-12-13)

### Feat

- Download loci_tools.jar; fix tests.py3.10

### Docs

- RTD
- Updating tutorials
- Add docstrings and some type annotation

### Style

- Linted with pre-commit

### Test

- Separate module run with -k
- Refactor TestDataItem for correct typing

### Build

- **deps-dev**: update pre-commit requirement from <=3.5.0 to <=3.6.0 (#21)
- **deps**: bump actions/setup-python from 4 to 5 (#20)
- **deps-dev**: bump ruff from 0.1.6 to 0.1.7 (#19)
- **deps**: bump actions/deploy-pages from 2 to 3 (#18)
- **deps**: bump actions/configure-pages from 3 to 4 (#17)
- Switch from darglint to pydoclint
- **deps-dev**: update commitizen requirement from <=3.12.0 to <=3.13.0 (#15)
- **deps-dev**: bump pre-commit from 3.3.3 to 3.5.0 (#14)
- **deps**: bump actions/cache from 2 to 3 (#13)
- **deps**: bump actions/setup-java from 3 to 4 (#12)

### CI/CD

- Change caches and tests
- Try to fix lint

### Refactor

- Add more type annotations
- Add some typing and click test
- Reformat code using ruff for improved consistency
- ruff and precommit
- Drop docopt in favor of click for `imgdiff`

### chore

- Refactor few variable names

## v0.1.0 (2023-11-30)

### Feat

- Add jpype and pims; pytest markers for slow and jpype; blacken
- Add read2 using new metadata (bit [0]\*npar)

### Build

- Refactor from setup.py to pyproject.toml with hatch

### Refactor

- Renamed nima_io; Update up to py-3.10; Update deps
- data test; jpype 30x faster md reading

## v0.0.1 (2023-07-27)

- Transferred from bitbucket.
- Read all metadata from various data files

Available in [TestPyPI](https://test.pypi.org/project/imgread/0.0.1/):

    pyenv virtualenv 3.8.18 test
    pyenv activate test
    pip install setuptools
    pip install lxml==4.2.3
    pip install javabridge==1.0.17
    pip install python-bioformats==1.4.0
    pip install -i https://test.pypi.org/simple/ imgread

### Added

- Project transferred from [Bitbucket](https://bitbucket.org/darosio/imgread/).
- Implemented functionality to read all metadata from various data files.

### Changed

This release marks the initial transfer of the project and introduces metadata reading capabilities for diverse data files.
