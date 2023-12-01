"""
Dummy conftest.py for nima_io.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
https://pytest.org/latest/plugins.html
"""

import os

import pytest

void_tile = {"filename": "tile6_1.tif"}

# data must be list of (list) [series, X, Y, C, time, Z, value]
test_md_data_dict = [
    {
        "filename": "exp2_2.tif",  # FEI_multichannel
        "SizeS": 1,
        "SizeX": 1600,
        "SizeY": 1200,
        "SizeC": 2,
        "SizeT": 81,
        "SizeZ": 1,
        "PhysicalSizeX": 0.74,
        "data": [
            [0, 610, 520, 0, 80, 0, 142],  # max = 212
            [0, 610, 520, 1, 80, 0, 132],  # max = 184
        ],
    },
    {
        "filename": "t4_1.tif",  # FEI_tiled
        "SizeS": 15,
        "SizeX": 512,
        "SizeY": 256,
        "SizeC": 4,
        "SizeT": 3,
        "SizeZ": 1,
        "PhysicalSizeX": 0.133333,
        "data": [
            [14, 509, 231, 0, 2, 0, 14580],
            [14, 509, 231, 1, 2, 0, 8436],
            [14, 509, 231, 2, 2, 0, 8948],
            [14, 509, 231, 3, 2, 0, 8041],
            [7, 194, 192, 1, 0, 0, 3783],
            [7, 194, 192, 1, 1, 0, 3585],
            [7, 194, 192, 1, 2, 0, 3403],
        ],
    },
    {
        "filename": "multi-channel-time-series.ome.tif",  # ome_multichannel
        "SizeS": 1,
        "SizeX": 439,
        "SizeY": 167,
        "SizeC": 3,
        "SizeT": 7,
        "SizeZ": 1,
        "PhysicalSizeX": None,
        "data": [],
    },
    {
        "filename": "2015Aug28_TransHXB2_50min+DMSO.lif",  # LIF_multiseries
        "SizeS": 5,
        "SizeX": 512,
        "SizeY": 512,
        "SizeC": 3,
        "SizeT": 1,
        "SizeZ": [41, 40, 43, 39, 37],
        "PhysicalSizeX": 0.080245,
        "data": [
            [4, 256, 128, 2, 0, 21, 2],
            [4, 285, 65, 2, 0, 21, 16],
            [4, 285, 65, 0, 0, 21, 14],
        ],  # max = 255
    },
]


def read_fixture(request, test_d):
    """Helper function to produce read fixtures."""
    read = request.cls.read
    filepath = os.path.join(os.path.dirname(request.fspath), "data", test_d["filename"])
    md, wr = read(filepath)
    return test_d, md, wr


@pytest.fixture(
    scope="class",
    params=test_md_data_dict,
    ids=["FEI multichannel", "FEI multitiles", "OME std test", "Leica LIF"],
)
def read_all(request):
    """Fixture to read all test files."""
    yield read_fixture(request, request.param)
    print("closing fixture: " + str(request.cls.read))


@pytest.fixture(
    scope="class",
    params=test_md_data_dict[:3],
    ids=["FEI multichannel", "FEI multitiles", "OME std test"],
)
def read_tif(request):
    """Fixture to read all TIF test files."""
    yield read_fixture(request, request.param)
    print("closing fixture: " + str(request.cls.read))


@pytest.fixture(scope="class")
def read_fei_multichannel(request):
    """Fixture to read a single test files."""
    yield read_fixture(request, test_md_data_dict[0])
    print("closing fixture: " + str(request.cls.read))


@pytest.fixture(scope="class")
def read_fei_multitile(request):
    """Fixture to read the FEI TIF multitile test file."""
    yield read_fixture(request, test_md_data_dict[1])
    print("closing fixture: " + str(request.cls.read))


@pytest.fixture(scope="class")
def read_ome_multichannel(request):
    """Fixture to read the OME TIF multichannel test file."""
    yield read_fixture(request, test_md_data_dict[2])
    print("closing fixture: " + str(request.cls.read))


@pytest.fixture(scope="class")
def read_lif(request):
    """Fixture to read the Leica LIF test file."""
    yield read_fixture(request, test_md_data_dict[3])
    print("closing fixture: " + str(request.cls.read))


@pytest.fixture(scope="class")
def read_void_tile(request):
    """Fixture to read the multitile test file with a missing tile."""
    yield read_fixture(request, void_tile)
    print("closing fixture: " + str(request.cls.read))
