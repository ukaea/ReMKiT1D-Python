from RMK_support.grid import Grid
import RMK_support.crm_support as crm
import RMK_support.model_construction as mc
import RMK_support.variable_container as vc
from RMK_support.derivations import Derivation, Species
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
    )


def test_add_spontaneous_emission(grid: Grid):
    mbData = crm.CRMModelboundData(grid)

    transitions = {(2, 1): 10.0, (3, 1): 20.0}

    crm.addHSpontaneousEmissionToCRMData(mbData, transitions, 3, 1, 10.0, 10.0)

    assert mbData.dict()["transitionTags"] == ["SpontEmissionH2-1", "SpontEmissionH3-1"]

    assert mbData.dict()["transitions"]["SpontEmissionH2-1"] == {
        "type": "simpleTransition",
        "ingoingState": 2,
        "outgoingState": 1,
        "fixedEnergy": 13.6 * (1 - 1 / 2**2) / 10,
        "rate": 10.0 * 10.0,
    }

    assert mbData.dict()["transitions"]["SpontEmissionH3-1"] == {
        "type": "simpleTransition",
        "ingoingState": 3,
        "outgoingState": 1,
        "fixedEnergy": 13.6 * (1 - 1 / 3**2) / 10,
        "rate": 20.0 * 10.0,
    }


def test_saha_boltzmann():
    dist = crm.hydrogenSahaBoltzmann(20, 10, 1e19)

    logdist = np.array([np.log(n / (i + 1) ** 2) for i, n in enumerate(dist[1:])])

    logdistDiff = logdist[1:] - logdist[:-1]

    expectedDiff = np.array(
        [13.6 / 10 * (1 / (i + 1) ** 2 - 1 / i**2) for i in range(1, 20)]
    )

    assert np.all(np.abs(logdistDiff - expectedDiff) < 1e-14)


def test_crm_term_generator(grid: Grid):

    species = Species("a", 1, associatedVars=[vc.Variable("na", grid)])

    assert crm.CRMTermGenerator("crm", [species]).dict() == {
        "implicitGroups": [1],
        "generalGroups": [],
        "type": "CRMDensityEvolution",
        "evolvedSpeciesIDs": [species.speciesID],
        "includedTransitionIndices": [],
    }


def test_crm_el_energy_term_generator(grid: Grid):

    W = vc.Variable("W", grid)

    assert crm.CRMElEnergyTermGenerator("crmElEnergy", W).dict() == {
        "implicitGroups": [1],
        "generalGroups": [],
        "type": "CRMElectronEnergyEvolution",
        "electronEnergyDensity": W.name,
        "includedTransitionIndices": [],
    }
