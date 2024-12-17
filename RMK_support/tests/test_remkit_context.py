import numpy as np
from RMK_support.remkit_context import RMKContext, IOContext, MPIContext, Variable
from RMK_support.grid import Grid
import RMK_support.derivations as dv
import RMK_support.variable_container as vc
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

    assert rk.variables.dict() == vc.VariableContainer(grid).dict()

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


def test_add_var(grid: Grid):

    rk = RMKContext()

    rk.grid = grid

    numProcsX = 10
    numProcsH = 2
    xHaloWidth = 1
    rk.mpiContext = MPIContext(numProcsX, numProcsH, xHaloWidth)

    # Add implicit variables
    a = Variable("a", rk.grid, isCommunicated=True)
    b = Variable("b", rk.grid, isDerived=True, isScalar=True, isCommunicated=True)

    # Add derived variables
    var, var_dual = vc.varAndDual(
        "var",
        rk.grid,
        primaryOnDualGrid=True,
        isDerived=True,
        derivation=dv.NodeDerivation("testDerivation", node=vc.node(a) + vc.node(b)),
    )

    # Add additional properties to a variable, e.g. scalarHostProcess
    b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]
    assert b.scalarHostProcess == (numProcsX - 1) * numProcsH

    # Add the same variables to another variable container and compare with the rk.variables container
    rk.variables.add(a, b, var, var_dual)

    compVariableContainer = vc.VariableContainer(grid)
    compVariableContainer.add(a, b, var, var_dual)

    assert rk.variables.dict() == compVariableContainer.dict()


def test_io(grid):

    rk = RMKContext()

    rk.grid = grid

    jsonFilepath = "./testing.json"
    rk.IOContext.jsonFilepath = jsonFilepath
    assert rk.IOContext.jsonFilepath == jsonFilepath

    HDF5Dir = "./testingDir/"
    rk.IOContext.HDF5Dir = HDF5Dir
    assert rk.IOContext.HDF5Dir == HDF5Dir

    inputFile = "./input_file.nc"
    inputVars = [Variable("a", rk.grid), Variable("b", rk.grid)]
    rk.IOContext.setHDF5InputOptions(inputFile, inputVars)
    assert rk.IOContext.dict()["timeloop"]["loadInitValsFromHDF5"] == True
    assert rk.IOContext.dict()["timeloop"]["initValFilename"] == inputFile

    restartOptions = {
        "save": False,
        "load": True,
        "frequency": 9999,
        "resetTime": False,
        "initialOutputIndex": 100,
    }
    rk.IOContext.setRestartOptions(**restartOptions)
    assert rk.IOContext.dict()["timeloop"]["restart"] == restartOptions
