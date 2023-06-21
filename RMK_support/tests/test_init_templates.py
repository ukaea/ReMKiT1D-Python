import RMK_support.init_templates as it
from RMK_support.grid import Grid
import numpy as np
import pytest
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


def test_add_var_and_dual_fluid(grid, vCont):
    testCont = vCont

    it.addVarAndDual(testCont, "var", primaryOnDualGrid=True)

    it.addVarAndDual(testCont, "var2")

    expectedOutput = {
        "variables": {
            "implicitVariables": {
                "names": ["var_dual", "var2"],
                "var_dual": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": True,
                    "priority": 0,
                    "initVals": np.zeros(grid.numX()).tolist(),
                },
                "var2": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.zeros(grid.numX()).tolist(),
                },
            },
            "derivedVariables": {
                "names": ["var", "var2_dual"],
                "var": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.zeros(grid.numX()).tolist(),
                    "derivationRule": {
                        "ruleName": "dualToGrid",
                        "requiredVarNames": ["var_dual"],
                    },
                },
                "var2_dual": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": True,
                    "priority": 0,
                    "initVals": np.zeros(grid.numX()).tolist(),
                    "derivationRule": {
                        "ruleName": "gridToDual",
                        "requiredVarNames": ["var2"],
                    },
                },
            },
        }
    }
    assert expectedOutput == testCont.dict()


def test_add_var_and_dual_dist(grid, vCont):
    testCont = vCont

    it.addVarAndDual(testCont, "var", isDistribution=True)

    expectedOutput = {
        "variables": {
            "implicitVariables": {
                "names": ["var"],
                "var": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": True,
                    "priority": 0,
                    "initVals": np.zeros(
                        grid.numX() * grid.numH() * grid.numV()
                    ).tolist(),
                },
            },
            "derivedVariables": {
                "names": ["var_dual"],
                "var_dual": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": True,
                    "priority": 0,
                    "initVals": np.zeros(
                        grid.numX() * grid.numH() * grid.numV()
                    ).tolist(),
                    "derivationRule": {
                        "ruleName": "distributionInterp",
                        "requiredVarNames": ["var"],
                    },
                },
            },
        }
    }
    assert expectedOutput == testCont.dict()
