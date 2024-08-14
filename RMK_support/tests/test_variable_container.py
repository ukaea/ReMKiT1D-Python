from RMK_support.grid import Grid
import numpy as np
import pytest
import xarray as xr
from RMK_support.variable_container import VariableContainer


@pytest.fixture
def grid():
    return Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        1,
        0,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )


@pytest.fixture
def vCont(grid):
    return VariableContainer(grid)


def test_vs_init(grid, vCont):
    assert all(vCont.dataset.coords["x"].data == grid.xGrid)
    assert all(vCont.dataset.coords["h"].data == np.array([0, 1]))
    assert all(vCont.dataset.coords["v"].data == grid.vGrid)


def test_add_fluid_zeros(grid, vCont):
    testCont = vCont

    testCont.setVariable("var")

    assert all(testCont.dataset["var"] == 0)
    assert testCont.getVarAttrs("var") == {
        "isDerived": False,
        "isDistribution": False,
        "units": "normalized units",
        "isStationary": False,
        "isScalar": False,
        "isOnDualGrid": False,
        "priority": 0,
        "derivationRule": "none",
        "normSI": 1.0,
        "unitSI": "",
    }


def test_add_fluid_nonzeros(grid, vCont):
    testCont = vCont

    with pytest.warns(
        UserWarning,
        match="Variable on dual grid var has been initialised with non-zero data. Make sure that the rightmost cell is zeroed out or intentionally left as non-zero.",
    ):
        testCont.setVariable(
            "var",
            np.ones(grid.numX()),
            isDerived=True,
            units="arb",
            isOnDualGrid=True,
            normSI=2.0,
            unitSI="a",
        )

    assert all(testCont.dataset["var"] == np.ones(grid.numX()))
    assert testCont.dataset["var"].attrs == {
        "isDerived": True,
        "isDistribution": False,
        "units": "arb",
        "isStationary": False,
        "isScalar": False,
        "isOnDualGrid": True,
        "priority": 0,
        "derivationRule": "none",
        "normSI": 2.0,
        "unitSI": "a",
    }


def test_add_dist_nonzeros(grid, vCont):
    testCont = vCont

    testCont.setVariable(
        "var",
        np.ones((grid.numX(), grid.numH(), grid.numV())),
        isDistribution=True,
        isDerived=True,
        units="arb",
    )

    assert np.all(
        testCont.dataset["var"].data == np.ones((grid.numX(), grid.numH(), grid.numV()))
    )
    assert testCont.dataset["var"].attrs == {
        "isDerived": True,
        "isDistribution": True,
        "units": "arb",
        "isStationary": False,
        "isScalar": False,
        "isOnDualGrid": False,
        "priority": 0,
        "derivationRule": "none",
        "normSI": 1.0,
        "unitSI": "",
    }


def test_add_scalar(grid, vCont):
    testCont = vCont

    testCont.setVariable("var", isScalar=True, isDerived=True, units="arb")

    assert np.shape(testCont.dataset["var"].data == [1])

    assert testCont.dataset["var"].attrs == {
        "isDerived": True,
        "isDistribution": False,
        "units": "arb",
        "isStationary": False,
        "isScalar": True,
        "isOnDualGrid": False,
        "priority": 0,
        "derivationRule": "none",
        "normSI": 1.0,
        "unitSI": "",
    }


def test_json_dump(grid, vCont):
    testCont = vCont

    testCont.setVariable("var0")
    testCont.setVariable("var1", np.ones(grid.numX()), isDerived=True, units="arb")
    testCont.setVariable(
        "var2",
        np.ones((grid.numX(), grid.numH(), grid.numV())),
        isDistribution=True,
        units="arb",
    )
    testCont.setVariable(
        "var3",
        isScalar=True,
        isDerived=True,
        units="arb",
        derivationRule={"testRule": True},
    )

    expectedOutput = {
        "variables": {
            "implicitVariables": {
                "names": ["var0", "var2"],
                "var0": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.zeros(grid.numX()).tolist(),
                },
                "var2": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.ones(
                        grid.numX() * grid.numH() * grid.numV()
                    ).tolist(),
                },
            },
            "derivedVariables": {
                "names": ["var1", "var3"],
                "var1": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.ones(grid.numX()).tolist(),
                },
                "var3": {
                    "isDistribution": False,
                    "isScalar": True,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.zeros(1).tolist(),
                    "derivationRule": {"testRule": True},
                },
            },
        }
    }

    assert expectedOutput == testCont.dict()
