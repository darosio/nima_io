<!-- markdownlint-disable MD024 -->
<!-- vale write-good.TooWordy = NO -->

# Changelog

## v0.3.0 (2024-01-06)

### Feat

- Adopt proper logging
- Add parametrized fixture tests for the transition to CoreMetadata
- Add CoreMetadata

### Fix

- Attenuation != 0.9 simplifying covert_value()
- RtD missing maven

### Docs

- Add missing docstrings
- Adopt github-pages-deploy-action

### Build

- **deps-dev**: update autodocsumm requirement (#44)
- **deps-dev**: update pydata-sphinx-theme requirement (#43)
- **deps-dev**: update numpy requirement from <=1.26.2 to <=1.26.3 (#42)
- **deps-dev**: update ruff requirement from <=0.1.9 to <=0.1.11 (#41)
- **deps**: update numpy requirement in /.github/workflows (#40)
- Drop py3.8 and add py3.12
- Remove the ci group
- **deps-dev**: update lxml requirement from <=4.9.4 to <=5.0.0 (#39)
- **deps-dev**: update pytest requirement from <=7.4.3 to <=7.4.4 (#38)
- **deps-dev**: update coverage[toml] requirement (#37)
- **deps-dev**: update jpype1 requirement from <=1.4.1 to <=1.5.0 (#36)
- **deps**: bump hatch from 1.9.0 to 1.9.1 in /.github/workflows (#35)
- **deps-dev**: update mypy requirement from <=1.7.1 to <=1.8.0 (#34)
- **deps-dev**: update ruff requirement from <=0.1.8 to <=0.1.9 (#33)
- **deps-dev**: update coverage[toml] requirement (#32)
- **deps-dev**: update lxml requirement from <=4.9.3 to <=4.9.4 (#31)
- **deps**: bump hatch from 1.8.1 to 1.9.0 in /.github/workflows (#30)
- **deps**: bump pip from 23.3.1 to 23.3.2 in /.github/workflows (#27)
- **deps-dev**: update coverage[toml] requirement (#26)
- **deps**: bump hatch from 1.8.0 to 1.8.1 in /.github/workflows (#24)
- **deps-dev**: update ruff requirement from <=0.1.7 to <=0.1.8 (#23)
- **deps**: bump hatch from 1.7.0 to 1.8.0 in /.github/workflows (#22)

### CI/CD

- Docs out of gh-pages branch

### Refactor

- Add mypy and missing tests
- Drop py3.9
- Adopt pathlib
- Remove outdated read functions
- Transition from python-bioformats with javabridge to scyjava
- Simplify jpype testing; update loci to 6.8.0
- Drop javabridge `read_jb` and bioformats `read_bf`
- Drop bftools showinf

### chore

- Again tutorials updating2
- Again tutorials updating
- Update Tutorials
- Set max-complexity to 13
- Fix few type annotations and pytest style
- Clean up workflows

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
