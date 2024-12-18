import numpy as np
from RMK_support.remkit_context import (
    Manipulator,
    ManipulatorCollection,
    MBDataExtractor,
    RMKContext,
    IOContext,
    MPIContext,
    Variable,
    TermEvaluator,
)
from RMK_support.grid import Grid
import RMK_support.remkit_context as rmk
import RMK_support.derivations as dv
import RMK_support.integrators as it
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

    assert rk.models.dict() == mc.ModelCollection().dict()

    assert rk.manipulators.dict() == ManipulatorCollection().dict()

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


def test_models_manipulators_terms(grid: Grid):
    """Testing minimum features of RMKContext, including:

    - Model with evolved variables, derived variables and terms
    - modelCollection
    - manipulatorCollection with groupEvaluator
    - integrationScheme
    - textbook, set with a custom derivation
    - term diagnostics via termEvaluator
    - MBDataExtractor
    """
    rk = RMKContext()

    rk.grid = grid

    rk.mpiContext = MPIContext(1)

    a, b = (Variable(name, rk.grid) for name in "ab")

    cDeriv = dv.NodeDerivation("cDeriv", node=vc.node(a) + vc.node(b))
    c = Variable("c", rk.grid, isDerived=True, derivation=cDeriv)

    rk.variables.add(a, b, c)

    # Model with terms and modelbound data

    model = mc.Model("newModel")

    model.ddt[a] += mc.DiagonalStencil()(a).rename("a")
    model.ddt[b] += -model.ddt[a].withSuffix("_b")

    mbData = mc.VarlikeModelboundData()
    dModelbound = Variable("dModelbound", rk.grid, isDerived=True, derivation=cDeriv)
    mbData.addVar(dModelbound)
    model.setModelboundData(mbData)

    # Model Collection

    modelCollection = mc.ModelCollection()

    assert modelCollection.dict() == {"tags": []}

    modelCollection.add(model)

    assert modelCollection.numGroups() == (1, 1)
    implicitGroups = modelCollection.numGroups()[1]

    rk.models = modelCollection

    assert rk.models.dict() == modelCollection.dict()

    # Manipulator Collection

    manipulatorCollection = ManipulatorCollection()

    assert manipulatorCollection.dict() == {"tags": []}

    resultVar = Variable("groupEvalResult", rk.grid, isDerived=True)

    groupEvaluator = rmk.GroupEvaluator(
        "groupEval",
        model,
        termGroup=99,
        resultVar=resultVar,
        priority=1,
    )

    assert groupEvaluator.dict() == {
        "type": "groupEvaluator",
        "modelTag": model.name,
        "evaluatedTermGroup": 99,
        "resultVarName": resultVar.name,
        "priority": 1,
    }

    manipulatorCollection.add(groupEvaluator)

    assert manipulatorCollection.dict() == {
        "tags": ["groupEval"],
        groupEvaluator.name: groupEvaluator.dict(),
    }

    # Test ManipulatorCollection getter and setter (by checking, removing and re-adding groupEvaluator)

    assert (
        manipulatorCollection.__getitem__(groupEvaluator.name).dict()
        == groupEvaluator.dict()
    )

    manipulatorCollection.__delitem__(groupEvaluator.name)
    assert groupEvaluator.name not in manipulatorCollection.manipNames

    manipulatorCollection.__setitem__(groupEvaluator.name, groupEvaluator)
    assert manipulatorCollection.dict()[groupEvaluator.name] == groupEvaluator.dict()

    # Set the RMKContext ManipulatorCollection

    rk.manipulators = manipulatorCollection

    assert rk.manipulators.dict() == manipulatorCollection.dict()

    # Integration Scheme

    # The RMKContext integrationScheme is initially empty, so raises an error
    with pytest.raises(AssertionError) as e_info:
        rk.dict()
    assert e_info.value.args[0] == "IntegrationScheme not set"

    integrationScheme = it.IntegrationScheme(
        dt=0.1,
        steps=it.IntegrationStep(
            "BDEStep",
            it.BDEIntegrator(
                "BDE",
                nonlinTol=1e-12,
                absTol=10.0,
                convergenceVars=[Variable("a", rk.grid)],
            ),
        ),
    )

    rk.integrationScheme = integrationScheme

    assert rk.integrationScheme.dict(
        implicitGroups, mpiComm=rk.mpiContext.dict(rk.variables)
    ) == integrationScheme.dict(
        implicitGroups, mpiComm=rk.mpiContext.dict(rk.variables)
    )

    # Textbook

    # Test the textbook setter by registering the derivation in another textbook, then setting RMKContext.textbook
    tb = dv.Textbook(grid)

    rk.variables.registerDerivs(tb)
    rk.models.registerDerivs(tb)

    assert tb.dict()["customDerivations"]["tags"] == [cDeriv.name]
    assert tb.dict()["customDerivations"][cDeriv.name] == cDeriv.dict()

    rk.textbook = tb

    assert rk.textbook.dict() == tb.dict()

    # Term diagnostics

    # Currently the only manipulator should be the "groupEval" manipulator added earlier
    # Now add term diagnostic manipulators for the model terms

    # Terms with evolved variables can have term diagnostics
    rk.addTermDiagnostics(*[a, b])

    # Adding a term diagnostic for a non-evolved term should raise a warning
    with pytest.warns(
        UserWarning,
        match=(
            "addTermDiagnostics called when variable "
            + c.name
            + " has no terms that evolve it"
        ),
    ):
        rk.addTermDiagnostics(*[c])

    # Get the list of term tags for all models in the RMKContext
    termTagsGrouped = [
        [
            "_".join([model, term])
            for model, term in rk.models.getTermsThatEvolveVar(var)
        ]
        for var in [a, b, c]
    ]
    termTags = [item for pair in termTagsGrouped for item in pair]

    # RMKContext should now contain the existing evaluator manipulator plus the newly added term diagnostics
    assert rk.manipulators.dict()["tags"] == [groupEvaluator.name] + termTags

    for term in rk.models[model.name].dict()["termTags"]:
        tag = "_".join([model.name, term])
        termEvaluator = TermEvaluator(
            tag, [(model.name, term)], resultVar=rk.variables[tag]
        )

        assert termEvaluator.dict() == {
            "type": "termEvaluator",
            "evaluatedModelNames": [model.name],
            "evaluatedTermNames": [term],
            "resultVarName": tag,
            "priority": 4,
            "update": False,
            "accumulate": False,
        }

        assert rk.manipulators.dict()[tag] == termEvaluator.dict()

        # Modelbound data extractor

        mbExtract = MBDataExtractor("mbExtract", model, dModelbound, priority=1).dict()

        assert mbExtract == {
            "type": "modelboundDataExtractor",
            "modelTag": model.name,
            "modelboundDataName": dModelbound.name,
            "resultVarName": dModelbound.name,
            "priority": 1,
        }

        # Check final config output

        cfg = rk.dict()

        assert cfg["normalization"] == skn.calculateNorms(
            Te=rk.normTemperature, ne=rk.normDensity, Z=rk.normZ
        )

        assert cfg["species"] == rk.species.dict()

        assert cfg["MPI"] == rk.mpiContext.dict(rk.variables)

        assert cfg["PETSc"] == rk.optionsPETSc

        assert cfg["models"] == rk.models.dict()

        assert cfg["manipulators"] == rk.manipulators.dict()

        # assert cfg["xGrid"] == rk.grid.xGrid
        # assert cfg["vGrid"] == rk.grid.vGrid

        assert cfg["variables"] == rk.variables.dict()["variables"]

        assert cfg["HDF5"] == rk.IOContext.dict()["HDF5"]

        # TODO: "timeloop", "integrator", "standardTextbook", "customDerivations"


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
