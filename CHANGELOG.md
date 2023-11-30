<!-- markdownlint-disable MD024 -->
<!-- vale write-good.TooWordy = NO -->

# Changelog

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
