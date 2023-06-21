import RMK_support.crm_support as crm
import pytest
import numpy as np


def test_crm_mb_data():
    energies = np.geomspace(0.01, 1, 20)
    mbData = crm.ModelboundCRMData(energies)

    mbData.addTransitionEnergy(0.05)

    assert np.all(
        mbData.fixedTransitionEnergies == np.append(np.geomspace(0.01, 1, 20), [0.05])
    )
    assert mbData.energyResolution == 1.0e-16

    mbData.addTransition("t1", {"fixedEnergyIndex": 1})
    mbData.addTransition("t2", {"fixedEnergyIndex": 2})
    mbData.addTransition("d2", {"fixedEnergyIndex": 3})

    assert mbData.transitionTags == ["t1", "t2", "d2"]
    assert mbData.transitionProperties == {
        "t1": {"fixedEnergyIndex": 1},
        "t2": {"fixedEnergyIndex": 2},
        "d2": {"fixedEnergyIndex": 3},
    }

    assert mbData.dict() == {
        "modelboundDataType": "modelboundCRMData",
        "transitionTags": ["t1", "t2", "d2"],
        "inelasticGridData": {
            "active": True,
            "fixedTransitionEnergies": np.append(
                np.geomspace(0.01, 1, 20), [0.05]
            ).tolist(),
        },
        "transitions": {
            "t1": {"fixedEnergyIndex": 1},
            "t2": {"fixedEnergyIndex": 2},
            "d2": {"fixedEnergyIndex": 3},
        },
    }

    assert mbData.getTransitionIndicesAndEnergies("t") == ([1, 2], [1, 2])


def test_add_janev_transitions():
    mbData = crm.ModelboundCRMData()

    crm.addJanevTransitionsToCRMData(
        mbData, 2, 10, "f", "T", lowestCellEnergy=0.05**2
    )

    transitionEnergies = np.array(
        [
            13.6 * (1 - 1 / 2**2) / 10,
            -13.6 * (1 - 1 / 2**2) / 10,
            13.6 / 10 + 0.05**2,
            13.6 / 40 + 0.05**2,
            -13.6 / 10 - 0.05**2,
            -13.6 / 40 - 0.05**2,
        ]
    )

    assert np.all(mbData.fixedTransitionEnergies == transitionEnergies)

    dataRepr = mbData.dict()
    assert dataRepr["transitionTags"] == [
        "JanevEx1-2",
        "JanevDeex2-1",
        "JanevIon1",
        "JanevIon2",
        "JanevRecomb3b1",
        "JanevRecomb3b2",
        "JanevRecombRad1",
        "JanevRecombRad2",
    ]

    assert dataRepr["transitions"]["JanevEx1-2"] == {
        "type": "JanevCollExIon",
        "startHState": 1,
        "endHState": 2,
        "fixedEnergyIndex": 1,
        "distributionVarName": "f",
    }
    assert dataRepr["transitions"]["JanevRecomb3b1"] == {
        "type": "JanevCollDeexRecomb",
        "startHState": 0,
        "endHState": 1,
        "directTransitionFixedEnergyIndex": 3,
        "directTransitionIndex": 3,
        "fixedEnergyIndex": 5,
        "distributionVarName": "f",
        "electronTemperatureVar": "T",
        "crossSectionUpdatePriority": 0,
    }


def test_add_spont_trans():
    mbData = crm.ModelboundCRMData()

    transitions = {(2, 1): 10.0, (3, 1): 20.0}
    crm.addHSpontaneousEmissionToCRMData(mbData, transitions, 3, 1, 10.0, 10.0)

    dataRepr = mbData.dict()
    assert dataRepr["transitionTags"] == ["SpontEmissionH2-1", "SpontEmissionH3-1"]

    assert dataRepr["transitions"]["SpontEmissionH2-1"] == {
        "type": "simpleTransition",
        "ingoingState": 2,
        "outgoingState": 1,
        "fixedEnergy": 13.6 * (1 - 1 / 2**2) / 10,
        "rate": 100.0,
    }


def test_saha_boltzmann():
    dist = crm.hydrogenSahaBoltzmann(20, 10, 1e19)

    logdist = np.array([np.log(n / (i + 1) ** 2) for i, n in enumerate(dist[1:])])

    logdistDiff = logdist[1:] - logdist[:-1]

    expectedDiff = np.array(
        [13.6 / 10 * (1 / (i + 1) ** 2 - 1 / i**2) for i in range(1, 20)]
    )

    assert np.all(np.abs(logdistDiff - expectedDiff) < 1e-14)
