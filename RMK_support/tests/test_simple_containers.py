import RMK_support.simple_containers as sc
import pytest
import numpy as np


@pytest.fixture
def mbData():
    data = sc.VarlikeModelboundData()

    data.addVariable("var1", {"deriv": True})
    data.addVariable("var2", {"deriv": True}, True, isDerivedFromOtherData=True)
    data.addVariable("var3", {"deriv": True}, isSingleHarmonic=True, priority=4)

    return data


def test_general_matrix_term_simple():
    term = sc.GeneralMatrixTerm(
        "evo", "impl", varData=sc.VarData(["var"]), customNormConst=5.0
    )

    assert term.dict() == {
        "evolvedVar": "evo",
        "implicitVar": "impl",
        "spatialProfile": [],
        "harmonicProfile": [],
        "velocityProfile": [],
        "evaluatedTermGroup": 0,
        "implicitGroups": [1],
        "generalGroups": [1],
        "customNormConst": {"multConst": 5.0, "normNames": [], "normPowers": []},
        "timeSignalData": sc.TimeSignalData().dict(),
        "varData": sc.VarData(["var"]).dict(),
        "stencilData": {},
        "skipPattern": False,
        "fixedMatrix": False,
    }


def test_general_matrix_term_profiles():
    term = sc.GeneralMatrixTerm(
        "evo",
        "impl",
        customNormConst=sc.CustomNormConst(2.0, normNames=["norm1"]),
        spatialProfile=np.ones(15).tolist(),
        harmonicProfile=np.zeros(4).tolist(),
        velocityProfile=2 * np.ones(10).tolist(),
        timeSignalData=sc.TimeSignalData(signalType="box", period=5),
    )

    assert term.dict() == {
        "evolvedVar": "evo",
        "implicitVar": "impl",
        "spatialProfile": np.ones(15).tolist(),
        "harmonicProfile": np.zeros(4).tolist(),
        "velocityProfile": 2 * np.ones(10).tolist(),
        "evaluatedTermGroup": 0,
        "implicitGroups": [1],
        "generalGroups": [1],
        "customNormConst": {
            "multConst": 2.0,
            "normNames": ["norm1"],
            "normPowers": [1.0],
        },
        "timeSignalData": sc.TimeSignalData(signalType="box", period=5).dict(),
        "varData": sc.VarData().dict(),
        "stencilData": {},
        "skipPattern": False,
        "fixedMatrix": False,
    }


def test_custom_model_add_term():
    testModel = sc.CustomModel("model")

    testModel.addTerm("term1", sc.GeneralMatrixTerm("var"))
    testModel.addTerm("term2", sc.GeneralMatrixTerm("var2"))

    assert testModel.dict() == {
        "model": {
            "type": "customModel",
            "termTags": ["term1", "term2"],
            "term1": sc.GeneralMatrixTerm("var").dict(),
            "term2": sc.GeneralMatrixTerm("var2").dict(),
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_varlike_mb_data(mbData):
    assert mbData.dict() == {
        "modelboundDataType": "varlikeData",
        "dataNames": ["var1", "var2", "var3"],
        "var1": {
            "isDistribution": False,
            "isScalar": False,
            "isSingleHarmonic": False,
            "isDerivedFromOtherData": False,
            "derivationPriority": 0,
            "deriv": True,
        },
        "var2": {
            "isDistribution": True,
            "isScalar": False,
            "isSingleHarmonic": False,
            "isDerivedFromOtherData": True,
            "derivationPriority": 0,
            "deriv": True,
        },
        "var3": {
            "isDistribution": False,
            "isScalar": False,
            "isSingleHarmonic": True,
            "isDerivedFromOtherData": False,
            "derivationPriority": 4,
            "deriv": True,
        },
    }


def test_add_mb_data(mbData):
    testModel = sc.CustomModel("model")

    testModel.setModelboundData(mbData.dict())

    assert testModel.dict() == {
        "model": {
            "type": "customModel",
            "termTags": [],
            "modelboundData": mbData.dict(),
            "termGenerators": {"tags": []},
        }
    }


def test_integration_step():
    testStep = sc.IntegrationStep(
        "int",
        globalStepFraction=0.5,
        allowTimeEvolution=False,
        defaultEvaluateGroups=[1, 2, 3],
        defaultUpdateModelData=True,
        defaultUpdateGroups=[1, 2, 3],
    )

    testStep.addModel("model")
    testStep.addModel("model2", updateGroups=[1])

    assert not testStep.allowTimeEvolution
    assert testStep.defaultUpdateModelData
    assert testStep.defaultEvaluateGroups == [1, 2, 3]
    assert testStep.defaultUpdateGroups == [1, 2, 3]
    assert testStep.evolvedModels == ["model", "model2"]
    assert testStep.evolvedModelProperties == {
        "model": {
            "groupIndices": [1, 2, 3],
            "internallyUpdatedGroups": [1, 2, 3],
            "internallyUpdateModelData": True,
        },
        "model2": {
            "groupIndices": [1, 2, 3],
            "internallyUpdatedGroups": [1],
            "internallyUpdateModelData": True,
        },
    }

    assert testStep.globalStepFraction == 0.5
    assert testStep.integratorTag == "int"
    assert not testStep.useInitialInput
    assert testStep.dict() == {
        "integratorTag": "int",
        "evolvedModels": ["model", "model2"],
        "globalStepFraction": 0.5,
        "allowTimeEvolution": False,
        "useInitialInput": False,
        "model": {
            "groupIndices": [1, 2, 3],
            "internallyUpdatedGroups": [1, 2, 3],
            "internallyUpdateModelData": True,
        },
        "model2": {
            "groupIndices": [1, 2, 3],
            "internallyUpdatedGroups": [1],
            "internallyUpdateModelData": True,
        },
    }


def test_additive_derivartion():
    deriv1 = sc.additiveDerivation(["deriv1"], 1, [[1]])

    assert deriv1 == {
        "deriv1": {"derivationIndices": [1]},
        "derivationTags": ["deriv1"],
        "linearCoefficients": [],
        "resultPower": 1,
        "type": "additiveDerivation",
    }

    with pytest.raises(AssertionError) as excinfo:
        sc.additiveDerivation(["deriv2"], 1, [[1]], [3.14, 1e-5])
    assert (
        str(excinfo.value)
        == "derivTags and linCoeffs in additiveDerivation must be of same size"
    )
