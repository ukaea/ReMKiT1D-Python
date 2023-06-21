import RMK_support.common_models as cm
import pytest
import numpy as np
from RMK_support import Grid, VariableContainer, RKWrapper
import RMK_support.simple_containers as sc


def test_collocated_advection():
    newModel = cm.collocatedAdvection(
        "adv",
        "var",
        "var_flux",
        centralDiff=True,
        lowerBoundVar="bound",
        leftOutflow=True,
        rightOutflow=True,
    )

    assert newModel.dict() == {
        "adv": {
            "type": "customModel",
            "termTags": ["divFlux", "leftBC", "rightBC"],
            "divFlux": {
                "evolvedVar": "var",
                "implicitVar": "var",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["speed", "time", "length"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "centralDifferenceInterpolated",
                    "interpolatedVarName": "var_flux",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "leftBC": {
                "evolvedVar": "var",
                "implicitVar": "var",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["speed", "time", "length"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "boundaryStencil",
                    "fluxJacVar": "var_flux",
                    "leftBoundary": True,
                    "lowerBoundVar": "bound",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "rightBC": {
                "evolvedVar": "var",
                "implicitVar": "var",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["speed", "time", "length"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "boundaryStencil",
                    "fluxJacVar": "var_flux",
                    "leftBoundary": False,
                    "lowerBoundVar": "bound",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_collocated_pressure_grad():
    elCharge = 1.60218e-19

    newModel = cm.collocatedPressureGrad("pGrad", "flux", "n", "T", elCharge)

    assert newModel.dict() == {
        "pGrad": {
            "type": "customModel",
            "termTags": ["bulkGrad", "leftBC", "rightBC"],
            "bulkGrad": {
                "evolvedVar": "flux",
                "implicitVar": "n",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["eVTemperature", "time", "length", "speed"],
                    "normPowers": [1.0, 1.0, -1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqColVars=["T"]).dict(),
                "stencilData": {
                    "stencilType": "centralDifferenceInterpolated",
                    "ignoreJacobian": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "leftBC": {
                "evolvedVar": "flux",
                "implicitVar": "n",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["eVTemperature", "time", "length", "speed"],
                    "normPowers": [1.0, 1.0, -1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqColVars=["T"]).dict(),
                "stencilData": {
                    "stencilType": "boundaryStencil",
                    "leftBoundary": True,
                    "ignoreJacobian": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "rightBC": {
                "evolvedVar": "flux",
                "implicitVar": "n",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["eVTemperature", "time", "length", "speed"],
                    "normPowers": [1.0, 1.0, -1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqColVars=["T"]).dict(),
                "stencilData": {
                    "stencilType": "boundaryStencil",
                    "leftBoundary": False,
                    "ignoreJacobian": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_staggered_advection():
    newModel = cm.staggeredAdvection(
        "adv",
        "var",
        "varflux",
        "u",
        lowerBoundVar="cs",
        leftOutflow=True,
        rightOutflow=True,
    )

    assert newModel.dict() == {
        "adv": {
            "type": "customModel",
            "termTags": ["divFlux", "leftBC", "rightBC"],
            "divFlux": {
                "evolvedVar": "var",
                "implicitVar": "varflux",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["speed", "time", "length"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {"stencilType": "staggeredDifferenceStencil"},
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "leftBC": {
                "evolvedVar": "var",
                "implicitVar": "var",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["speed", "time", "length"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "boundaryStencil",
                    "fluxJacVar": "u",
                    "leftBoundary": True,
                    "lowerBoundVar": "cs",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "rightBC": {
                "evolvedVar": "var",
                "implicitVar": "var",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["speed", "time", "length"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "boundaryStencil",
                    "fluxJacVar": "u",
                    "leftBoundary": False,
                    "lowerBoundVar": "cs",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_staggered_pressure_grad():
    elCharge = 1.60218e-19

    newModel = cm.staggeredPressureGrad("pGrad", "flux", "n", "T", elCharge)

    assert newModel.dict() == {
        "pGrad": {
            "type": "customModel",
            "termTags": ["bulkGrad"],
            "bulkGrad": {
                "evolvedVar": "flux",
                "implicitVar": "n",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["eVTemperature", "time", "length", "speed"],
                    "normPowers": [1.0, 1.0, -1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqColVars=["T"]).dict(),
                "stencilData": {
                    "stencilType": "staggeredDifferenceStencil",
                    "ignoreJacobian": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_lorentz_forces():
    elCharge = 1.60218e-19
    amu = 1.6605390666e-27

    species = [sc.Species("e", 0, charge=-1), sc.Species("ion", -1, charge=1)]

    newModel = cm.lorentzForces(
        "lorentz", "E", ["flux_e", "flux_ion"], ["ne", "nion"], species
    )

    assert newModel.dict() == {
        "lorentz": {
            "type": "customModel",
            "termTags": ["lorentzflux_e", "lorentzflux_ion"],
            "lorentzflux_e": {
                "evolvedVar": "flux_e",
                "implicitVar": "E",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elCharge / amu,
                    "normNames": ["EField", "time", "speed"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqRowVars=["ne"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "lorentzflux_ion": {
                "evolvedVar": "flux_ion",
                "implicitVar": "E",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": elCharge / amu,
                    "normNames": ["EField", "time", "speed"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqRowVars=["nion"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_lorentz_force_work():
    species = [sc.Species("e", 0, charge=-1), sc.Species("ion", -1, charge=1)]

    newModel = cm.lorentzForceWork(
        "lorentz", "E", ["flux_e", "flux_ion"], ["We", "Wion"], species
    )

    assert newModel.dict() == {
        "lorentz": {
            "type": "customModel",
            "termTags": ["lorentzWorkflux_e", "lorentzWorkflux_ion"],
            "lorentzWorkflux_e": {
                "evolvedVar": "We",
                "implicitVar": "E",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1,
                    "normNames": ["EField", "time", "speed", "eVTemperature"],
                    "normPowers": [1.0, 1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqColVars=["flux_e"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "lorentzWorkflux_ion": {
                "evolvedVar": "Wion",
                "implicitVar": "E",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 1,
                    "normNames": ["EField", "time", "speed", "eVTemperature"],
                    "normPowers": [1.0, 1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqColVars=["flux_ion"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_implicit_temperature():
    elCharge = 1.60218e-19
    amu = 1.6605390666e-27  # atomic mass unit

    species = [sc.Species("e", 0, charge=-1), sc.Species("ion", -1, charge=1)]

    newModel = cm.implicitTemperatures(
        "temp",
        ["flux_e", "flux_ion"],
        ["We", "Wion"],
        ["ne", "nion"],
        ["Te", "Tion"],
        species,
        ["ne_dual", "nion_dual"],
    )

    assert newModel.dict() == {
        "temp": {
            "type": "customModel",
            "termTags": [
                "identityTermTe",
                "wTermTe",
                "u2TermTe",
                "identityTermTion",
                "wTermTion",
                "u2TermTion",
            ],
            "identityTermTe": {
                "evolvedVar": "Te",
                "implicitVar": "Te",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "wTermTe": {
                "evolvedVar": "Te",
                "implicitVar": "We",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 2 / 3,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqRowVars=["ne"], reqRowPowers=[-1.0]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
            "u2TermTe": {
                "evolvedVar": "Te",
                "implicitVar": "flux_e",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -amu / (3 * elCharge),
                    "normNames": ["speed", "eVTemperature"],
                    "normPowers": [2.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(
                    reqColVars=["flux_e", "ne_dual"], reqColPowers=[1.0, -2.0]
                ).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "identityTermTion": {
                "evolvedVar": "Tion",
                "implicitVar": "Tion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "wTermTion": {
                "evolvedVar": "Tion",
                "implicitVar": "Wion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 2 / 3,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqRowVars=["nion"], reqRowPowers=[-1.0]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
            "u2TermTion": {
                "evolvedVar": "Tion",
                "implicitVar": "flux_ion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -amu / (3 * elCharge),
                    "normNames": ["speed", "eVTemperature"],
                    "normPowers": [2.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(
                    reqColVars=["flux_ion", "nion_dual"], reqColPowers=[1.0, -2.0]
                ).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_kinAdvX():
    grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        lMax=1,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    newModel = cm.kinAdvX("adv", "f", grid, evolvedHarmonics=[1, 2])

    assert newModel.dict() == {
        "adv": {
            "type": "customModel",
            "termTags": ["adv_plus1", "adv_minus2"],
            "adv_plus1": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": grid.vGrid.tolist(),
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0 / 3.0,
                    "normNames": ["time", "velGrid", "length"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "kineticSpatialDiffStencil",
                    "rowHarmonic": 1,
                    "colHarmonic": 2,
                },
                "skipPattern": False,
                "fixedMatrix": True,
            },
            "adv_minus2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": grid.vGrid.tolist(),
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": ["time", "velGrid", "length"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "kineticSpatialDiffStencil",
                    "rowHarmonic": 2,
                    "colHarmonic": 1,
                },
                "skipPattern": False,
                "fixedMatrix": True,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_ampere_maxwell():
    elCharge = 1.60218e-19
    epsilon0 = 8.854188e-12  # vacuum permittivity

    species = [sc.Species("e", 0, charge=-1), sc.Species("ion", -1, charge=1)]

    newModel = cm.ampereMaxwell("amp", "E", ["flux_e", "flux_ion"], species)

    assert newModel.dict() == {
        "amp": {
            "type": "customModel",
            "termTags": ["currentflux_e", "currentflux_ion"],
            "currentflux_e": {
                "evolvedVar": "E",
                "implicitVar": "flux_e",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": elCharge / epsilon0,
                    "normNames": ["density", "time", "speed", "EField"],
                    "normPowers": [1.0, 1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "currentflux_ion": {
                "evolvedVar": "E",
                "implicitVar": "flux_ion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elCharge / epsilon0,
                    "normNames": ["density", "time", "speed", "EField"],
                    "normPowers": [1.0, 1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {},
            "termGenerators": {"tags": []},
        }
    }


def test_E_adv():
    rk = RKWrapper()
    rk.grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        lMax=1,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    newModel = cm.advectionEx("eAdv", "f", "E", rk, "f_dual")

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    chargeMassRatio = elCharge / elMass

    assert newModel.dict() == {
        "eAdv": {
            "type": "customModel",
            "termTags": ["eAdv_H1", "eAdv_G2"],
            "eAdv_H1": {
                "evolvedVar": "f",
                "implicitVar": "E",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": chargeMassRatio / 3.0,
                    "normNames": ["EField", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["H_h=2"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [1],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "eAdv_G2": {
                "evolvedVar": "f",
                "implicitVar": "E",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": chargeMassRatio,
                    "normNames": ["EField", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -1.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["G_h=1"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {
                "modelboundDataType": "varlikeData",
                "dataNames": ["G_h=1", "H_h=1", "G_h=2", "H_h=2"],
                "G_h=1": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "G_h=1",
                    "requiredVarNames": ["f_dual"],
                },
                "H_h=1": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "H_h=1",
                    "requiredVarNames": ["f_dual"],
                },
                "G_h=2": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "G_h=2",
                    "requiredVarNames": ["f_dual"],
                },
                "H_h=2": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "H_h=2",
                    "requiredVarNames": ["f_dual"],
                },
            },
            "termGenerators": {"tags": []},
        }
    }


def test_eeCollIsotropic():
    rk = RKWrapper()
    rk.grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        lMax=1,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    newModel = cm.eeCollIsotropic("ee", "f", "T", "n", rk)

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)

    assert newModel.dict() == {
        "ee": {
            "type": "customModel",
            "termTags": ["dragTerm", "diffTerm"],
            "dragTerm": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1.0 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee"]).dict(),
                "stencilData": {
                    "stencilType": "ddvStencil",
                    "modelboundC": "dragCCL",
                    "modelboundInterp": "weightCCL",
                    "rowHarmonic": 1,
                    "colHarmonic": 1,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "diffTerm": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1.0 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee"]).dict(),
                "stencilData": {
                    "stencilType": "vDiffusionStencil",
                    "modelboundA": "diffCCL",
                    "rowHarmonic": 1,
                    "colHarmonic": 1,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {
                "modelboundDataType": "varlikeData",
                "dataNames": ["f0", "dragCCL", "diffCCL", "weightCCL", "logLee"],
                "f0": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "f0Extraction",
                    "requiredVarNames": ["f"],
                },
                "dragCCL": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "cclDragCoeff",
                    "requiredVarNames": ["f0"],
                },
                "diffCCL": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "cclDiffusionCoeff",
                    "requiredVarNames": ["f0", "weightCCL"],
                },
                "weightCCL": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "cclWeight",
                    "requiredVarNames": ["dragCCL", "diffCCL"],
                },
                "logLee": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "logLee",
                    "requiredVarNames": ["T", "n"],
                },
            },
            "termGenerators": {"tags": []},
        }
    }


def test_eiCollIsotropic():
    rk = RKWrapper()
    rk.grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        lMax=1,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    rk.addSpecies("ion", -1, 1, 1.0)

    newModel = cm.eiCollIsotropic(
        "ei", "f", "T", "n", "Tion", "nion", "ion", "Wion", rk
    )

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity
    amu = 1.6605390666e-27  # atomic mass unit

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)

    gamma0norm = gamma0norm * elMass / amu

    vGrid = rk.grid.vGrid
    dv = rk.grid.vWidths
    vBoundary = [vGrid[i] + dv[i] / 2 for i in range(len(dv))]
    vOuter = [1.0 / v**2 for v in rk.grid.vGrid]
    innerV = [1.0 / (2 * v) for v in vBoundary]

    assert newModel.dict() == {
        "ei": {
            "type": "customModel",
            "termTags": ["dragTerm", "diffTerm", "diffTermIon", "dragTermIon"],
            "dragTerm": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": vOuter,
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(
                    reqMBRowVars=["logLei"], reqRowVars=["nion"]
                ).dict(),
                "stencilData": {
                    "stencilType": "ddvStencil",
                    "modelboundC": "none",
                    "modelboundInterp": "none",
                    "rowHarmonic": 1,
                    "colHarmonic": 1,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "diffTerm": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": vOuter,
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(
                    reqMBRowVars=["logLei"], reqRowVars=["nion", "Tion"]
                ).dict(),
                "stencilData": {
                    "stencilType": "vDiffusionStencil",
                    "modelboundA": "none",
                    "rowHarmonic": 1,
                    "colHarmonic": 1,
                    "fixedA": innerV,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "diffTermIon": {
                "evolvedVar": "Wion",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 2,
                    "colHarmonic": 1,
                    "termName": "diffTerm",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "dragTermIon": {
                "evolvedVar": "Wion",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1.0,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 2,
                    "colHarmonic": 1,
                    "termName": "dragTerm",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {
                "modelboundDataType": "varlikeData",
                "dataNames": ["logLei"],
                "logLei": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "logLeiion",
                    "requiredVarNames": ["T", "n"],
                },
            },
            "termGenerators": {"tags": []},
        }
    }


def test_stationaryIonEIColl():
    rk = RKWrapper()
    rk.grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        lMax=2,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    rk.addSpecies("ion", -1, 1, 1.0)

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)

    newModel = cm.stationaryIonEIColl("ei", "f", "nion", "n", "T", "ion", [2, 3], rk)

    vProfile = [1.0 / v**3 for v in rk.grid.vGrid]

    assert newModel.dict() == {
        "ei": {
            "type": "customModel",
            "termTags": ["eiCollStationaryIons"],
            "eiCollStationaryIons": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [0.0, 1.0, 3.0],
                "velocityProfile": vProfile,
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -gamma0norm,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(
                    reqMBRowVars=["logLei"], reqRowVars=["nion"]
                ).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2, 3],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {
                "modelboundDataType": "varlikeData",
                "dataNames": ["logLei"],
                "logLei": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "logLeiion",
                    "requiredVarNames": ["T", "n"],
                },
            },
            "termGenerators": {"tags": []},
        }
    }


def test_flowingIonEIColl():
    rk = RKWrapper()
    rk.grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        lMax=1,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    rk.addSpecies("ion", -1, 1, 1.0)

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)
    amu = 1.6605390666e-27  # atomic mass unit

    elIonMassRatio = elMass / amu

    newModel = cm.flowingIonEIColl(
        "ei", "f", "nion", "uion", "n", "T", "ion", rk, [2], "f_dual", "G_ion"
    )

    assert newModel.dict() == {
        "ei": {
            "type": "customModel",
            "termTags": [
                "diffTermI2",
                "diffTermJ2",
                "dfdv2",
                "termLL2",
                "C1Il+2_h=2",
                "C1J-l-1_h=2",
                "C2Il_h=2",
                "C2J1-l_h=2",
                "C3Il+2_h=2",
                "C4J-l-1_h=2",
                "C5Il_h=2",
                "C6J1-l_h=2",
                "diffTermIIon",
                "diffTermJIon",
                "dfdvTermIon",
                "llTermIon",
                "ddf0TermIon1",
                "ddf0TermIon2",
                "ddf0TermIon3",
                "ddf0TermIon4",
                "df0TermIon1",
                "df0TermIon2",
                "df0TermIon3",
                "df0TermIon4",
            ],
            "diffTermI2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "CII2sh"]).dict(),
                "stencilData": {
                    "stencilType": "vDiffusionStencil",
                    "modelboundA": "none",
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "adfAtZero": [1.0 / rk.grid.vGrid[1], 0],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "diffTermJ2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "CIJ-1sh"]).dict(),
                "stencilData": {
                    "stencilType": "vDiffusionStencil",
                    "modelboundA": "none",
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "adfAtZero": [1.0 / rk.grid.vGrid[1], 0],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "dfdv2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "IJSum2"]).dict(),
                "stencilData": {
                    "stencilType": "ddvStencil",
                    "modelboundC": "none",
                    "modelboundInterp": "none",
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "termLL2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**3 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "IJSum"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C1Il+2_h=2": {
                "evolvedVar": "f",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 5,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "ddf0", "CII3"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C1J-l-1_h=2": {
                "evolvedVar": "f",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 5,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "ddf0", "CIJ-2"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C2Il_h=2": {
                "evolvedVar": "f",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 0.0,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "ddf0", "CII1"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C2J1-l_h=2": {
                "evolvedVar": "f",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 0.0,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "ddf0", "CIJ0"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C3Il+2_h=2": {
                "evolvedVar": "f",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -gamma0norm / 5,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "df0", "CII3"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C4J-l-1_h=2": {
                "evolvedVar": "f",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": (2.0 / 15.0 + 1.0 / 3.0) * gamma0norm,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "df0", "CIJ-2"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C5Il_h=2": {
                "evolvedVar": "f",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "df0", "CII1"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C6J1-l_h=2": {
                "evolvedVar": "f",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -0.0,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLei", "df0", "CIJ0"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "diffTermIIon": {
                "evolvedVar": "G_ion",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "diffTermI2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "diffTermJIon": {
                "evolvedVar": "G_ion",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "diffTermJ2",
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "dfdvTermIon": {
                "evolvedVar": "G_ion",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "dfdv2",
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "llTermIon": {
                "evolvedVar": "G_ion",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "termLL2",
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "ddf0TermIon1": {
                "evolvedVar": "G_ion",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C1Il+2_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "ddf0TermIon2": {
                "evolvedVar": "G_ion",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C1J-l-1_h=2",
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "ddf0TermIon3": {
                "evolvedVar": "G_ion",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C2Il_h=2",
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "ddf0TermIon4": {
                "evolvedVar": "G_ion",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C2J1-l_h=2",
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "df0TermIon1": {
                "evolvedVar": "G_ion",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C3Il+2_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "df0TermIon2": {
                "evolvedVar": "G_ion",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C4J-l-1_h=2",
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "df0TermIon3": {
                "evolvedVar": "G_ion",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C5Il_h=2",
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "df0TermIon4": {
                "evolvedVar": "G_ion",
                "implicitVar": "nion",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -elIonMassRatio / 3,
                    "normNames": ["velGrid", "speed"],
                    "normPowers": [
                        1.0,
                        -1.0,
                    ],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C6J1-l_h=2",
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "modelboundData": {
                "modelboundDataType": "varlikeData",
                "dataNames": [
                    "CII0",
                    "CII2",
                    "CIJ-1",
                    "CII0sh",
                    "CII2sh",
                    "CIJ-1sh",
                    "IJSum",
                    "IJSum2",
                    "logLei",
                    "df0",
                    "ddf0",
                    "CII1",
                    "CII3",
                    "CIJ-2",
                    "CIJ0",
                ],
                "CII0": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "CII0",
                    "requiredVarNames": ["uion"],
                },
                "CII2": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "CII2",
                    "requiredVarNames": ["uion"],
                },
                "CIJ-1": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "CIJ-1",
                    "requiredVarNames": ["uion"],
                },
                "CII0sh": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "f0Extraction",
                    "requiredVarNames": ["CII0"],
                },
                "CII2sh": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "f0Extraction",
                    "requiredVarNames": ["CII2"],
                },
                "CIJ-1sh": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "f0Extraction",
                    "requiredVarNames": ["CIJ-1"],
                },
                "IJSum": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "sumTerm",
                    "requiredVarNames": ["CII2sh", "CIJ-1sh", "CII0sh"],
                },
                "IJSum2": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "sumTerm2",
                    "requiredVarNames": ["CII2sh", "CIJ-1sh"],
                },
                "logLei": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "logLeiion",
                    "requiredVarNames": ["T", "n"],
                },
                "df0": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "df0/dv",
                    "requiredVarNames": ["f_dual"],
                },
                "ddf0": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "d2f0/dv2",
                    "requiredVarNames": ["f_dual"],
                },
                "CII1": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "CII1",
                    "requiredVarNames": ["uion"],
                },
                "CII3": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "CII3",
                    "requiredVarNames": ["uion"],
                },
                "CIJ-2": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "CIJ-2",
                    "requiredVarNames": ["uion"],
                },
                "CIJ0": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "CIJ0",
                    "requiredVarNames": ["uion"],
                },
            },
            "termGenerators": {"tags": []},
        }
    }


def test_eeCollHigherL():
    rk = RKWrapper()
    rk.grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        lMax=1,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)
    newModel = cm.eeCollHigherL("ee", "f", "T", "n", rk, [2], "f_dual")

    assert newModel.dict() == {
        "ee": {
            "type": "customModel",
            "termTags": [
                "8pi*f0*fl",
                "diffTermI2",
                "diffTermJ2",
                "dfdv2",
                "termLL",
                "C1Il+2_h=2",
                "C1J-l-1_h=2",
                "C2Il_h=2",
                "C2J1-l_h=2",
                "C3Il+2_h=2",
                "C4J-l-1_h=2",
                "C5Il_h=2",
                "C6J1-l_h=2",
            ],
            "8pi*f0*fl": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 8 * np.pi * gamma0norm,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "f0"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "diffTermI2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "I2"]).dict(),
                "stencilData": {
                    "stencilType": "vDiffusionStencil",
                    "modelboundA": "none",
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "adfAtZero": [1.0 / rk.grid.vGrid[1], 0],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "diffTermJ2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "J-1"]).dict(),
                "stencilData": {
                    "stencilType": "vDiffusionStencil",
                    "modelboundA": "none",
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "adfAtZero": [1.0 / rk.grid.vGrid[1], 0],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "dfdv2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "IJSum"]).dict(),
                "stencilData": {
                    "stencilType": "ddvStencil",
                    "modelboundC": "none",
                    "modelboundInterp": "none",
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "termLL": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [0.0, 1.0],
                "velocityProfile": [1 / v**3 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "IJSum"]).dict(),
                "stencilData": {
                    "stencilType": "diagonalStencil",
                    "evolvedXCells": [],
                    "evolvedHarmonics": [2],
                    "evolvedVCells": [],
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C1Il+2_h=2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 5,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "ddf0"]).dict(),
                "stencilData": {
                    "stencilType": "shkarofskyIJStencil",
                    "JIntegral": False,
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "integralIndex": 3,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C1J-l-1_h=2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 5,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "ddf0"]).dict(),
                "stencilData": {
                    "stencilType": "shkarofskyIJStencil",
                    "JIntegral": True,
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "integralIndex": -2,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C2Il_h=2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 0.0,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "ddf0"]).dict(),
                "stencilData": {
                    "stencilType": "shkarofskyIJStencil",
                    "JIntegral": False,
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "integralIndex": 1,
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C2J1-l_h=2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 0.0,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "ddf0"]).dict(),
                "stencilData": {
                    "stencilType": "shkarofskyIJStencil",
                    "JIntegral": True,
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "integralIndex": 0,
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C3Il+2_h=2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -gamma0norm / 5,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "df0"]).dict(),
                "stencilData": {
                    "stencilType": "shkarofskyIJStencil",
                    "JIntegral": False,
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "integralIndex": 3,
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C4J-l-1_h=2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 2 * gamma0norm / 15,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "df0"]).dict(),
                "stencilData": {
                    "stencilType": "shkarofskyIJStencil",
                    "JIntegral": True,
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "integralIndex": -2,
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C5Il_h=2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": gamma0norm / 3,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "df0"]).dict(),
                "stencilData": {
                    "stencilType": "shkarofskyIJStencil",
                    "JIntegral": False,
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "integralIndex": 1,
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "C6J1-l_h=2": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [1 / v**2 for v in rk.grid.vGrid],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -0.0,
                    "normNames": ["density", "time", "velGrid"],
                    "normPowers": [1.0, 1.0, -3.0],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData(reqMBRowVars=["logLee", "df0"]).dict(),
                "stencilData": {
                    "stencilType": "shkarofskyIJStencil",
                    "JIntegral": True,
                    "rowHarmonic": 2,
                    "colHarmonic": 2,
                    "integralIndex": 0,
                },
                "skipPattern": True,
                "fixedMatrix": False,
            },
            "modelboundData": {
                "modelboundDataType": "varlikeData",
                "dataNames": [
                    "f0",
                    "I0",
                    "I2",
                    "J-1",
                    "IJSum",
                    "logLee",
                    "df0",
                    "ddf0",
                ],
                "f0": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "f0Extraction",
                    "requiredVarNames": ["f_dual"],
                },
                "I0": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "I0",
                    "requiredVarNames": ["f0"],
                },
                "I2": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "I2",
                    "requiredVarNames": ["f0"],
                },
                "J-1": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "J-1",
                    "requiredVarNames": ["f0"],
                },
                "IJSum": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": True,
                    "derivationPriority": 0,
                    "ruleName": "sumTerm",
                    "requiredVarNames": ["I2", "J-1", "I0"],
                },
                "logLee": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": False,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "logLee",
                    "requiredVarNames": ["T", "n"],
                },
                "df0": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "df0/dv",
                    "requiredVarNames": ["f_dual"],
                },
                "ddf0": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isSingleHarmonic": True,
                    "isDerivedFromOtherData": False,
                    "derivationPriority": 0,
                    "ruleName": "d2f0/dv2",
                    "requiredVarNames": ["f_dual"],
                },
            },
            "termGenerators": {"tags": []},
        }
    }


def test_lbcModelRight():
    rk = RKWrapper()
    rk.grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        lMax=2,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    newModel = cm.lbcModel(
        "lbc", "f", rk, {"rule": True}, "j_b", evolvedHarmonics=[1, 3]
    )

    assert newModel.dict() == {
        "lbc": {
            "type": "customModel",
            "termTags": [
                "lbcPlus_odd1",
                "lbcPlus_even1",
                "lbcMinus_odd3",
                "lbcMinus_even3",
            ],
            "lbcPlus_odd1": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1 / 3,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "scalingLogicalBoundaryStencil",
                    "includedDecompHarmonics": [2],
                    "rowHarmonic": 1,
                    "colHarmonic": 2,
                    "leftBoundary": False,
                    "rule": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "lbcPlus_even1": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -1 / 3,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "scalingLogicalBoundaryStencil",
                    "includedDecompHarmonics": [1, 3],
                    "rowHarmonic": 1,
                    "colHarmonic": 2,
                    "leftBoundary": False,
                    "rule": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "lbcMinus_odd3": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -2 / 3,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "scalingLogicalBoundaryStencil",
                    "includedDecompHarmonics": [2],
                    "rowHarmonic": 3,
                    "colHarmonic": 2,
                    "leftBoundary": False,
                    "rule": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "lbcMinus_even3": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": -2 / 3,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "scalingLogicalBoundaryStencil",
                    "includedDecompHarmonics": [1, 3],
                    "rowHarmonic": 3,
                    "colHarmonic": 2,
                    "leftBoundary": False,
                    "rule": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {
                "modelboundDataType": "modelboundLBCData",
                "ionCurrentVarName": "j_b",
                "totalCurrentVarName": "none",
                "bisectionTolerance": 1e-12,
                "leftBoundary": False,
                "rule": True,
            },
            "termGenerators": {"tags": []},
        }
    }


def test_lbcModelLeft():
    rk = RKWrapper()
    rk.grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        lMax=2,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    newModel = cm.lbcModel(
        "lbc",
        "f",
        rk,
        {"rule": True},
        "j_b",
        evolvedHarmonics=[1, 3],
        leftBoundary=True,
    )

    assert newModel.dict() == {
        "lbc": {
            "type": "customModel",
            "termTags": ["lbcPlus1", "lbcMinus3"],
            "lbcPlus1": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 1 / 3,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "scalingLogicalBoundaryStencil",
                    "rowHarmonic": 1,
                    "colHarmonic": 2,
                    "leftBoundary": True,
                    "rule": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "lbcMinus3": {
                "evolvedVar": "f",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [1],
                "customNormConst": {
                    "multConst": 2 / 3,
                    "normNames": [],
                    "normPowers": [],
                },
                "timeSignalData": sc.TimeSignalData().dict(),
                "varData": sc.VarData().dict(),
                "stencilData": {
                    "stencilType": "scalingLogicalBoundaryStencil",
                    "rowHarmonic": 3,
                    "colHarmonic": 2,
                    "leftBoundary": True,
                    "rule": True,
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "modelboundData": {
                "modelboundDataType": "modelboundLBCData",
                "ionCurrentVarName": "j_b",
                "totalCurrentVarName": "none",
                "bisectionTolerance": 1e-12,
                "leftBoundary": True,
                "rule": True,
            },
            "termGenerators": {"tags": []},
        }
    }
