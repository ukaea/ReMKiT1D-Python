import RMK_support.integrators as it
import RMK_support.model_construction as mc
from RMK_support import Variable, Grid
import numpy as np
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
        isLengthInMeters=True,
    )


def test_bde_integrator(grid):

    integrator = it.BDEIntegrator(
        "BDE", nonlinTol=1e-12, absTol=10.0, convergenceVars=[Variable("a", grid)]
    )

    assert integrator.dict() == {
        "type": "BDE",
        "maxNonlinIters": 100,
        "nonlinTol": 1e-12,
        "absTol": 10.0,
        "convergenceVars": ["a"],
        "associatedPETScGroup": 1,
        "use2Norm": False,
        "relaxationWeight": 1.0,
        "internalStepControl": {
            "active": False,
            "startingNumSteps": 1,
            "stepMultiplier": 2,
            "stepDecrament": 1,
            "minNumNonlinIters": 5,
            "maxBDERestarts": 3,
            "BDEConsolidationInterval": 50,
        },
    }


def test_steps(grid):

    integrator = it.BDEIntegrator(
        "BDE", nonlinTol=1e-12, absTol=10.0, convergenceVars=[Variable("a", grid)]
    )

    step1 = it.IntegrationStep("step1", integrator)

    model = mc.Model("mod")
    model.ddt[Variable("a", grid)] += mc.DiagonalStencil()(Variable("a", grid))
    model2 = mc.Model("mod2")  # to test if this model is removed
    model2.ddt[Variable("a", grid)] += mc.DiagonalStencil()(Variable("a", grid))
    step1.add(model, model2)

    scheme = it.IntegrationScheme(10.0)

    scheme.steps = (
        step1(0.5).enableTimeEvo()
        * step1.rename("step2")(1.0).startFromZero()
        * step1(1.0).disableTimeEvo()
    )

    scheme.setFixedNumTimesteps(100, 10)
    with pytest.warns(
        UserWarning,
        match=(
            "Model mod2 excluded from integration rules - not present in filtering models"
        ),
    ):
        assert scheme.dict(1, {}, models=mc.ModelCollection(model)) == {
            "integrator": {
                "initialTimestep": 10.0,
                "timestepController": {
                    "active": False,
                    "rescaleTimestep": True,
                    "requiredVarNames": [],
                    "requiredVarPowers": [],
                    "multConst": 1.0,
                    "useMaxVal": False,
                },
                "stepTags": ["step12", "step20", "step11"],
                "integratorTags": ["BDE"],
                "BDE": integrator.dict(),
                "step12": {
                    "integratorTag": "BDE",
                    "evolvedModels": ["mod"],
                    "globalStepFraction": 1.0,
                    "useInitialInput": False,
                    "allowTimeEvolution": False,
                    "commData": {},
                    "mod": {
                        "groupIndices": [1],
                        "internallyUpdatedGroups": [1],
                        "internallyUpdateModelData": True,
                    },
                },
                "step11": {
                    "integratorTag": "BDE",
                    "evolvedModels": ["mod"],
                    "globalStepFraction": 0.5,
                    "useInitialInput": False,
                    "allowTimeEvolution": True,
                    "commData": {},
                    "mod": {
                        "groupIndices": [1],
                        "internallyUpdatedGroups": [1],
                        "internallyUpdateModelData": True,
                    },
                },
                "step20": {
                    "integratorTag": "BDE",
                    "evolvedModels": ["mod"],
                    "globalStepFraction": 1.0,
                    "useInitialInput": True,
                    "allowTimeEvolution": True,
                    "commData": {},
                    "mod": {
                        "groupIndices": [1],
                        "internallyUpdatedGroups": [1],
                        "internallyUpdateModelData": True,
                    },
                },
            },
            "timeloop": {
                "mode": "fixedNumSteps",
                "numTimesteps": 100,
                "fixedSaveInterval": 10,
                "outputPoints": [],
            },
        }

    scheme.setOutputPoints([10, 20])

    assert scheme.__outputPoints__ == [10, 20]


def test_rk_integrator():

    rk = it.RKIntegrator("rk", 3)

    assert rk.dict() == {"type": "RK", "order": 3}


def test_CVODE_integrator():

    cvode = it.CVODEIntegrator(
        "CVODE",
        relTol=1e-3,
        absTol=1e-9,
        maxGMRESRestarts=1,
        CVODEBBDPreParams=(1, 1, 1, 1),
        useAdamsMoulton=True,
        useStabLimitDet=True,
        maxOrder=3,
        maxInternalStep=100,
        minTimestep=0.1,
        maxTimestep=0.2,
        initTimestep=0.15,
    )

    assert cvode.dict() == {
        "type": "CVODE",
        "relTol": 1e-3,
        "absTol": 1e-9,
        "maxRestarts": 1,
        "CVODEPreBBDParams": [1, 1, 1, 1],
        "CVODEUseAdamsMoulton": True,
        "CVODEUseStabLimDet": True,
        "CVODEMaxOrder": 3,
        "CVODEMaxInternalSteps": 100,
        "CVODEMaxStepSize": 0.2,
        "CVODEMinStepSize": 0.1,
        "CVODEInitStepSize": 0.15,
    }


def test_timestep(grid):

    dt = it.Timestep(0.1 * Variable("a", grid))

    assert not dt.usingMaxVal

    dt = dt.max()
    assert dt.usingMaxVal

    dt = dt.min()
    assert not dt.usingMaxVal

    assert dt.dict() == {
        "initialTimestep": 0.1,
        "timestepController": {
            "active": True,
            "rescaleTimestep": True,
            "requiredVarNames": ["a"],
            "requiredVarPowers": [1.0],
            "multConst": 1.0,
            "useMaxVal": False,
        },
    }
