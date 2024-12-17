import numpy as np
from RMK_support.remkit_context import (
    Manipulator,
    ManipulatorCollection,
    RMKContext,
    IOContext,
    MPIContext,
    Variable,
)
from RMK_support.grid import Grid
import RMK_support.remkit_context as rmk
import RMK_support.derivations as dv
import RMK_support.model_construction as mc
import RMK_support.sk_normalization as skn
import RMK_support.variable_container as vc

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

    # Test the empty variable container
    assert rk.variables.dict() == vc.VariableContainer(grid).dict()

    # Now define variables to add to the variable container
    a = Variable("a", rk.grid, isCommunicated=True)
    b = Variable("b", rk.grid, isDerived=True, isScalar=True, isCommunicated=True)

    var, var_dual = vc.varAndDual(
        "var",
        rk.grid,
        primaryOnDualGrid=True,
        isDerived=True,
        derivation=dv.NodeDerivation("testDerivation", node=vc.node(a) + vc.node(b)),
    )

    # Set additional properties of a variable, e.g. scalarHostProcess
    b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]
    assert b.scalarHostProcess == (numProcsX - 1) * numProcsH

    # Add the variables to another variable container and set the rk.variables container
    varCont = vc.VariableContainer(grid)
    varCont.add(a, b, var, var_dual)

    rk.variables = varCont

    assert rk.variables.dict() == varCont.dict()


def test_io(grid: Grid):

    rk = RMKContext()

    rk.grid = grid

    rk.mpiContext = MPIContext(1)

    jsonFilepath = "./testing.json"
    HDF5Dir = "./testingDir/"

    ioCont = IOContext(jsonFilepath, HDF5Dir)

    # Setters for json file, HDF5 directory and output file
    ioCont.jsonFilepath = jsonFilepath
    assert ioCont.jsonFilepath == jsonFilepath

    ioCont.HDF5Dir = HDF5Dir
    assert ioCont.HDF5Dir == HDF5Dir

    inputFile = "./input_file.nc"
    inputVars = [Variable("in1", rk.grid), Variable("in2", rk.grid)]
    ioCont.setHDF5InputOptions(inputFile, inputVars)
    assert ioCont.dict()["timeloop"]["loadInitValsFromHDF5"] == True
    assert ioCont.dict()["timeloop"]["initValFilename"] == inputFile

    # Add restart settings
    restartOptions = {
        "save": False,
        "load": True,
        "frequency": 9999,
        "resetTime": False,
        "initialOutputIndex": 100,
    }
    ioCont.setRestartOptions(**restartOptions)
    assert ioCont.dict()["timeloop"]["restart"] == restartOptions

    # Add output variables
    outputVar1 = Variable("out1", rk.grid)
    outputVar2 = Variable("out2", rk.grid)
    rk.variables.add(outputVar1, outputVar2)
    ioCont.populateOutputVars(rk.variables)
    # Check if the output variables were added to IOContext. Output variable 0 is always "time"
    assert ioCont.__outputVars__[1].__dict__ == outputVar1.__dict__
    assert ioCont.__outputVars__[2].__dict__ == outputVar2.__dict__

    # Setting rk.IOContext with a pre-built IOContext
    rk.IOContext = ioCont
    assert rk.IOContext.dict() == ioCont.dict()


def test_models_and_manipulators(grid):
    rk = RMKContext()

    rk.grid = grid

    # Adding a Model (collection)
    assert rk.models.dict() == mc.ModelCollection().dict()

    model = mc.Model("newModel")

    a, b, c, d = (Variable(name, rk.grid) for name in "abcd")

    model.ddt[a] += mc.DiagonalStencil()(a).rename("a")
    model.addTerm("c", -mc.DiagonalStencil()(c).withEvolvedVar(a))
    model.ddt[b] += -model.ddt[a].withSuffix("_b")
    model.ddt[c] += mc.DiagonalStencil()(d).rename("d")

    modelCollection = mc.ModelCollection()
    modelCollection.add(model)

    rk.models = modelCollection

    assert rk.models.dict() == modelCollection.dict()

    # Adding a Manipulator (collection)

    assert rk.manipulators.dict() == ManipulatorCollection().dict()

    # manipulatorCollection = ManipulatorCollection()
    # manipulatorCollection.add(
    #     rmk.GroupEvaluator(
    #         "groupEval",
    #         model,
    #         termGroup=1,
    #         resultVar=Variable("groupEvalResult", rk.grid),
    #     )
    # )


def test_set_petsc():
    rk = RMKContext()

    rk.setPETScOptions(
        relTol=1e-14, absTol=1e-15, divTol=1e6, maxIters=2000, kspSolverType="gmres"
    )

    assert rk.optionsPETSc == {
        "active": True,
        "solverOptions": {
            "solverToleranceRel": 1.0e-14,
            "solverToleranceAbs": 1.0e-15,
            "solverToleranceDiv": 0.1e7,
            "maxSolverIters": 2000,
            "kspSolverType": "gmres",
            "hyprePCType": "",
            "PETScCommandLineOpts": "-pc_type bjacobi -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1",
        },
        "objGroups": 1,
    }


def test_species(grid: Grid):
    rk = RMKContext()

    rk.grid = grid

    # Initialize species container
    speciesCont = dv.SpeciesContainer()
    assert rk.species.dict() == speciesCont.dict()

    # Add species to the container
    na = Variable("na", rk.grid, isCommunicated=True)
    rk.variables.add(na)

    speciesA = dv.Species(
        name="a", speciesID=-1, atomicA=1, charge=+1, associatedVars=[na]
    )
    speciesCont.add(speciesA)

    speciesB = dv.Species("b", 0)
    speciesCont.add(speciesB)

    rk.species = speciesCont

    assert rk.species.dict()["names"] == ["a", "b"]

    assert rk.species.dict()["a"] == speciesA.dict()
    assert rk.species.dict()["a"]["associatedVars"] == [na.name]

    assert rk.species.dict()["b"] == speciesB.dict()
