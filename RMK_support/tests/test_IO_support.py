import json
import numpy as np
import pytest

from RMK_support import Grid, Variable, VariableContainer
from RMK_support import IO_support as io


@pytest.fixture
def grid():
    return Grid(
        np.geomspace(5.0, 0.2, 8),
        np.geomspace(0.01, 0.8, 6),
        1,
        0,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )


def test_write_dict_to_json_truncates_shorter_update(tmp_path):
    filepath = tmp_path / "config.json"
    io.writeDictToJSON(
        {"key": "a_very_long_initial_value_" + "x" * 100}, filepath=str(filepath)
    )
    io.writeDictToJSON({"key": "short"}, filepath=str(filepath))

    with open(filepath) as f:
        loaded = json.load(f)
    assert loaded == {"key": "short"}


def test_hdf5_roundtrip_scalar_fluid_dist(tmp_path, grid):
    paths = []
    for i, t_val in enumerate([0.0, 1.5, 3.0]):
        vc = VariableContainer(grid, autoAddDuals=False)
        vc["time"].values = np.array([t_val])
        vc.add(
            Variable("s", grid, data=np.array([10.0 + i]), isScalar=True),
            Variable("n", grid, data=np.full(grid.numX, 2.0 + i)),
            Variable(
                "f",
                grid,
                data=np.full((grid.numX, grid.numH, grid.numV), 0.5 + i),
                isDistribution=True,
            ),
        )
        fp = tmp_path / f"ReMKiT1DVarOutput_{i}.h5"
        io.writeRMKHDF5(vc, filepath=str(fp))
        paths.append(str(fp))

    assert io.getOutputFilenames(str(tmp_path)) == [
        "ReMKiT1DVarOutput_0.h5",
        "ReMKiT1DVarOutput_1.h5",
        "ReMKiT1DVarOutput_2.h5",
    ]

    # Single-file scalar & fluid loading
    s_template = Variable("s", grid, isScalar=True)
    n_template = Variable("n", grid)
    f_template = Variable("f", grid, isDistribution=True)

    s_single = io.loadVariableFromHDF5(s_template, [paths[1]])
    assert s_single.values.shape == (1,)
    assert np.allclose(s_single.values, [11.0])

    # Multi-file scalar, fluid, and distribution loading (NumPy 2.0+ safe)
    loaded_vc = io.loadVarContFromHDF5(
        s_template, n_template, f_template, filepaths=paths
    )
    assert np.allclose(loaded_vc["s"].values, [10.0, 11.0, 12.0])
    assert loaded_vc["n"].values.shape == (3, grid.numX)
    assert np.allclose(loaded_vc["n"].values[2], 4.0)
    assert loaded_vc["f"].values.shape == (3, grid.numX, grid.numH, grid.numV)

    # Dummy loading without prior variable metadata
    dummy_vc = io.loadFromHDF5(grid, ["time", "s", "n", "f"], filepaths=paths)
    assert dummy_vc["s"].isScalar
    assert np.allclose(dummy_vc["s"].values, [10.0, 11.0, 12.0])
    assert np.allclose(dummy_vc["time"].values, [0.0, 1.5, 3.0])
    assert dummy_vc["f"].isDistribution
    assert dummy_vc["f"].values.shape == (3, grid.numX, grid.numH, grid.numV)
