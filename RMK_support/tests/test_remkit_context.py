import numpy as np
from RMK_support.remkit_context import RMKContext, IOContext, MPIContext
from RMK_support.variable_container import VariableContainer
from RMK_support.grid import Grid
from RMK_support import derivations as dv
import RMK_support.sk_normalization as skn

import pytest


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


def test_wrapper_init(grid: Grid):
    rk = RMKContext()

    rk.grid = grid
    assert rk.grid.dict() == grid.dict()

    assert rk.normDensity == 1e19
    assert rk.normTemperature == 10
    assert rk.normZ == 1
    assert rk.norms == skn.calculateNorms(
        Te=rk.normTemperature, ne=rk.normDensity, Z=rk.normZ
    )

    assert rk.textbook.dict() == dv.Textbook(grid).dict()

    assert rk.species.dict() == dv.SpeciesContainer().dict()

    assert rk.variables.dict() == VariableContainer(grid).dict()

    assert rk.mpiContext.dict(varCont=rk.variables) == MPIContext(1).dict(
        varCont=rk.variables
    )

    assert rk.IOContext.dict() == IOContext().dict()

    assert rk.optionsPETSc == {
        "active": True,
        "solverOptions": {
            "solverToleranceRel": 0.1e-16,
            "solverToleranceAbs": 1.0e-20,
            "solverToleranceDiv": 0.1e8,
            "maxSolverIters": 10000,
            "kspSolverType": "bcgs",
            "hyprePCType": "",
            "PETScCommandLineOpts": "-pc_type bjacobi -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1",
        },
        "objGroups": 1,
    }

    assert rk.models.dict() == {"tags": []}

    assert rk.manipulators.dict() == {"tags": []}

    assert rk.integrationScheme == None


def test_set_norm():
    rk = RMKContext()

    # Get default norm values
    oldTemp = rk.normTemperature
    oldDensity = rk.normDensity
    oldZ = rk.normZ

    newTemp = 99
    rk.normTemperature = newTemp
    assert rk.normTemperature == newTemp

    newDensity = 9.9e19
    rk.normDensity = newDensity
    assert rk.normDensity == newDensity

    newZ = 9
    rk.normZ = newZ
    assert rk.normZ == newZ

    # Reset values to defaults and check values again
    rk = RMKContext()
    assert rk.normTemperature == oldTemp
    assert rk.normDensity == oldDensity
    assert rk.normZ == oldZ
