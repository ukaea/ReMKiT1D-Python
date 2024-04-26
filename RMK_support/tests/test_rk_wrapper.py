import numpy as np
import pytest
from RMK_support import RKWrapper, VariableContainer, Grid
from RMK_support.simple_containers import (
    Species,
    CustomModel,
    TermGenerator,
    GeneralMatrixTerm,
)
import RMK_support.init_templates as it
import warnings


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


def test_wrapper_init(grid):
    rk = RKWrapper()

    rk.grid = grid

    assert rk.customDerivs == {"tags": []}

    assert rk.grid.dict() == grid.dict()

    assert rk.normalization["eVTemperature"] == 10.0
    assert rk.normalization["density"] == 1e19
    assert rk.normalization["referenceIonZ"] == 1.0
    assert rk.standardTextbook == {
        "temperatureDerivSpeciesIDs": [],
        "electronPolytropicCoeff": 1.0,
        "ionPolytropicCoeff": 1.0,
        "electronSheathGammaIonSpeciesID": -1,
    }

    assert rk.speciesData == {"names": []}
    assert rk.mpiData == {
        "numProcsX": 1,
        "numProcsH": 1,
        "xHaloWidth": 1,
        "commData": {
            "varsToBroadcast": [],
            "haloExchangeVars": [],
            "scalarVarsToBroadcast": [],
            "scalarBroadcastRoots": [],
        },
    }

    assert rk.modelData == {"tags": []}
    assert rk.jsonFilepath == "./config.json"

    assert rk.integratorData == {
        "stepTags": [],
        "integratorTags": [],
        "timestepController": {"active": False},
        "initialTimestep": 0.1,
        "numImplicitGroups": 1,
        "numGeneralGroups": 1,
    }
    assert rk.timeloopData == {
        "mode": "fixedNumSteps",
        "numTimesteps": 1,
        "timeValueTarget": 1.0,
        "outputMode": "fixedNumSteps",
        "fixedSaveInterval": 1,
        "minimumSaveInterval": 0.1,
        "restart": {"save": False, "load": False, "frequency": 1, "resetTime": False},
        "loadInitValsFromHDF5": False,
        "initValFilename": "ReMKiT1DVarInput",
        "outputPoints":[]
    }

    assert rk.manipulatorData == {"tags": []}

    assert rk.hdf5Filepath == "./RMKOutput/"

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

    rk.jsonFilepath = "newPath"
    assert rk.jsonFilepath == "newPath"


def test_set_norm():
    rk = RKWrapper()

    oldNorm = rk.normalization.copy()
    rk.setNormDensity(1e18)
    assert rk.normalization["density"] == 1e18
    rk.setNormTemperature(1)
    assert rk.normalization["eVTemperature"] == 1
    rk.setNormRefZ(2.0)
    assert rk.normalization["referenceIonZ"] == 2.0

    rk.normalization = oldNorm

    assert rk.normalization["eVTemperature"] == 10.0
    assert rk.normalization["density"] == 1e19
    assert rk.normalization["referenceIonZ"] == 1.0


def test_add_var(grid):
    rk = RKWrapper()

    rk.grid = grid

    rk.setMPIData(10, 2, 1)

    rk.addVar("a", isCommunicated=True)
    rk.addVar(
        "b", isDerived=True, isScalar=True, isCommunicated=True, hostScalarProcess=1
    )
    rk.addVarAndDual("var", isCommunicated=True, primaryOnDualGrid=True)

    compVarCont = VariableContainer(grid)
    compVarCont.setVariable("time", isDerived=True, isScalar=True)
    compVarCont.setVariable("a")
    compVarCont.setVariable("b", isDerived=True, isScalar=True)
    it.addVarAndDual(compVarCont, "var", primaryOnDualGrid=True)

    assert rk.varCont.dict() == compVarCont.dict()

    assert rk.mpiData["numProcsX"] == 10
    assert rk.mpiData["numProcsH"] == 2

    assert rk.mpiData["commData"] == {
        "varsToBroadcast": ["a", "var_dual", "var"],
        "haloExchangeVars": ["a", "var_dual", "var"],
        "scalarVarsToBroadcast": ["b"],
        "scalarBroadcastRoots": [1],
    }

    assert rk.varList() == ["time", "a", "b", "var_dual", "var"]
    assert rk.varsInOutput() == ["time", "a", "b", "var", "var_dual"]

    rk.varCont = VariableContainer(grid)

    assert rk.varCont.dict() == VariableContainer(grid).dict()


def test_add_var_auto_derivation(grid):
    rk = RKWrapper()

    rk.grid = grid

    rk.addVar(
        "a",
        isDerived=True,
        derivationRule={"ruleName": "deriv1"},
        derivOptions={"options": 2},
    )
    rk.addVarAndDual(
        "b",
        isDerived=True,
        derivationRule={"ruleName": "deriv2"},
        derivOptions={"options": 3},
    )

    assert rk.customDerivs == {
        "tags": ["deriv1", "deriv2"],
        "deriv1": {"options": 2},
        "deriv2": {"options": 3},
    }


def test_add_dist_var(grid):
    rk = RKWrapper(False)

    rk.grid = grid

    rk.addVarAndDual("a", isDistribution=True, isCommunicated=True)

    compVarCont = VariableContainer(grid)
    it.addVarAndDual(compVarCont, "a", isDistribution=True)

    assert rk.varCont.dict() == compVarCont.dict()

    assert rk.mpiData["commData"] == {
        "varsToBroadcast": ["a", "a_dual"],
        "haloExchangeVars": ["a", "a_dual"],
        "scalarVarsToBroadcast": [],
        "scalarBroadcastRoots": [],
    }


def test_set_options():
    rk = RKWrapper()

    rk.setPETScOptions(
        relTol=1e-14, absTol=1e-15, divTol=1e6, maxIters=2000, kspSolverType="gmres"
    )

    assert rk.optionsPETSc == {
        "active": True,
        "solverOptions": {
            "solverToleranceRel": 0.1e-13,
            "solverToleranceAbs": 1.0e-15,
            "solverToleranceDiv": 0.1e7,
            "maxSolverIters": 2000,
            "kspSolverType": "gmres",
            "hyprePCType": "",
            "PETScCommandLineOpts": "-pc_type bjacobi -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1",
        },
        "objGroups": 1,
    }

    rk.setHDF5Path("./path")
    assert rk.hdf5Filepath == "./path"

    rk.setStandardTextbookOptions([-2], 2.0, 2.0, -2)

    assert rk.standardTextbook == {
        "temperatureDerivSpeciesIDs": [-2],
        "electronPolytropicCoeff": 2.0,
        "ionPolytropicCoeff": 2.0,
        "electronSheathGammaIonSpeciesID": -2,
    }


def test_set_integrators():
    rk = RKWrapper()

    with pytest.warns(UserWarning) as warnings:
        rk.setIntegratorGlobalData(2, 2, 0.5)

    assert (
        warnings[0].message.args[0]
        == "Explicitly setting number of implicit groups in models. This is deprecated and provided only for legacy scripts. Useful checks are disabled. Use at own risk."
    )

    assert (
        warnings[1].message.args[0]
        == "Explicitly setting number of general groups in models. This is deprecated and provided only for legacy scripts. Useful checks are disabled. Use at own risk."
    )

    rk.setTimestepController({"property": True})

    rk.addIntegrator("integ", {"property": True})
    rk.addIntegrationStep("step", {"property": True})

    assert rk.integratorData == {
        "stepTags": ["step"],
        "integratorTags": ["integ"],
        "integ": {"property": True},
        "step": {"property": True, "commData": rk.mpiData["commData"]},
        "timestepController": {"active": True, "property": True},
        "initialTimestep": 0.5,
        "numImplicitGroups": 2,
        "numGeneralGroups": 2,
    }


def test_timeloop_options():
    rk = RKWrapper()

    rk.setFixedNumTimesteps(20)
    rk.setFixedStepOutput(10)

    assert rk.timeloopData["mode"] == "fixedNumSteps"
    assert rk.timeloopData["numTimesteps"] == 20
    assert rk.timeloopData["outputMode"] == "fixedNumSteps"
    assert rk.timeloopData["fixedSaveInterval"] == 10

    rk.setTimeTargetTimestepping(1000.0)
    assert rk.timeloopData["mode"] == "normalizedTimeTarget"
    assert rk.timeloopData["timeValueTarget"] == 1000.0

    rk.setTimeTargetTimestepping(1000.0, True)
    assert rk.timeloopData["mode"] == "realTimeTarget"

    rk.setMinimumIntervalOutput(10.0)
    assert rk.timeloopData["outputMode"] == "minimumSaveInterval"
    assert rk.timeloopData["minimumSaveInterval"] == 10.0

    rk.setOutputDrivenTimesteps([0.1, 0.2])
    assert rk.timeloopData["mode"] == "outputDriven"
    assert rk.timeloopData["outputPoints"] == [0.1, 0.2]
    rk.setRestartOptions(True, True, 100, True)

    assert rk.timeloopData["restart"] == {
        "save": True,
        "load": True,
        "frequency": 100,
        "resetTime": True,
    }

    rk.setHDF5FileInitialData(["var", "var2"], filename="hdf5file")

    assert rk.__optionsHDF5__["inputVars"] == ["var", "var2"]

    assert rk.timeloopData["loadInitValsFromHDF5"]
    assert rk.timeloopData["initValFilename"] == "hdf5file"


def test_species():
    rk = RKWrapper()

    rk.addSpecies("a", -1, 1, 1, ["var_a"])
    rk.addSpecies("e", 0)

    assert rk.getSpecies("a").name == "a"
    assert rk.getSpecies("a").speciesID == -1
    assert rk.getSpecies("a").atomicA == 1
    assert rk.getSpecies("a").charge == 1
    assert rk.getSpecies("a").associatedVars == ["var_a"]

    assert rk.getSpecies("e").dict() == Species("e", 0).dict()


def test_add_derivations():
    rk = RKWrapper()

    rk.addCustomDerivation("deriv1", {"options": 1})
    rk.addDerivationCollection({"deriv2": {"options": 2}, "deriv3": {"options": 3}})

    assert rk.customDerivs == {
        "tags": ["deriv1", "deriv2", "deriv3"],
        "deriv1": {"options": 1},
        "deriv2": {"options": 2},
        "deriv3": {"options": 3},
    }


def test_add_models_and_maniplators(grid):
    rk = RKWrapper()

    rk.grid = grid

    dummyModel1 = {
        "termTags": ["term1", "term2"],
        "term1": {"evolvedVar": "a"},
        "term2": {"evolvedVar": "b"},
    }

    dummyModel2 = {"termTags": ["term1"], "term1": {"evolvedVar": "a"}}
    with pytest.warns(
        UserWarning,
        match="Adding model as dictionary. This is deprecated and provided only for legacy scripts. Useful checks are disabled. Use at own risk.",
    ):
        rk.addModel({"model1": dummyModel1})
        rk.addModel({"model2": dummyModel2})

    assert rk.modelTags() == ["model1", "model2"]

    rk.addManipulator("manip", {"properties": True})

    assert rk.manipulatorData == {"tags": ["manip"], "manip": {"properties": True}}

    rk.addTermDiagnosisForVars(["a"])

    assert rk.manipulatorData == {
        "tags": ["manip", "model1term1", "model2term1"],
        "manip": {"properties": True},
        "model1term1": {
            "type": "termEvaluator",
            "accumulate": False,
            "evaluatedModelNames": ["model1"],
            "evaluatedTermNames": ["term1"],
            "resultVarName": "model1term1",
            "priority": 4,
            "update": False,
        },
        "model2term1": {
            "type": "termEvaluator",
            "accumulate": False,
            "evaluatedModelNames": ["model2"],
            "evaluatedTermNames": ["term1"],
            "resultVarName": "model2term1",
            "priority": 4,
            "update": False,
        },
    }


def test_add_diagnosis_terms_warning():

    rk = RKWrapper()

    with pytest.warns(
        UserWarning,
        match="addTermDiagnosisForVars called when variable n has no terms that evolve it",
    ):
        rk.addTermDiagnosisForVars(["n"])

    with pytest.warns(
        UserWarning,
        match="addTermDiagnosisForDistVars called when variable n has no terms that evolve it",
    ):
        rk.addTermDiagnosisForDistVars(["n"])


def test_add_model_as_obj_error(grid):

    rk = RKWrapper()

    rk.grid = grid
    rk.addVar("evo1")

    testModel = CustomModel("test")

    testModel.addTerm("term1", GeneralMatrixTerm("evo2", implicitGroups=[1, 2]))

    with pytest.raises(AssertionError) as e_info:
        rk.addModel(testModel)

    assert (
        e_info.value.args[0]
        == "Evolved variable evo2 not registered in used variable container"
    )


def test_add_model_as_obj(grid):

    rk = RKWrapper()

    rk.grid = grid
    rk.addVar("evo1")

    testModel = CustomModel("test")

    testModel.addTerm("term1", GeneralMatrixTerm("evo1", implicitGroups=[1, 2]))
    testModel.addTerm(
        "term2", GeneralMatrixTerm("evo1", implicitGroups=[3], generalGroups=[1, 2])
    )

    rk.addModel(testModel)

    assert rk.modelTags() == ["test"]

    assert rk.activeImplicitGroups["test"] == [1, 2, 3]
    assert rk.activeGeneralGroups["test"] == [1, 2]

    rk.setIntegratorGlobalData()

    assert rk.__integratorData__["numImplicitGroups"] == 3
    assert rk.__integratorData__["numGeneralGroups"] == 2


def test_add_term_gen_as_object():

    rk = RKWrapper()

    testModel = CustomModel("test")

    testModel.addTermGenerator("test1", TermGenerator([1, 2], [1, 2, 3], {"opt1": 1}))

    assert testModel.dict()["test"]["termGenerators"]["test1"] == {
        "implicitGroups": [1, 2],
        "generalGroups": [1, 2, 3],
        "opt1": 1,
    }
    rk.addModel(testModel)

    assert rk.modelTags() == ["test"]

    assert rk.activeImplicitGroups["test"] == [1, 2]
    assert rk.activeGeneralGroups["test"] == [1, 2, 3]
