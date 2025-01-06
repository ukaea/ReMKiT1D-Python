from RMK_support.grid import Grid
import RMK_support.crm_support as crm
import RMK_support.derivations as dv
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


def test_crm_mbdata(grid: Grid):
    mbData = crm.CRMModelboundData(grid)

    # Default settings of mbData (initialized with grid only)
    assert crm.CRMModelboundData(grid).dict() == {
        "modelboundDataType": "modelboundCRMData",
        "transitionTags": [],
        "inelasticGridData": {
            "active": False,
            "fixedTransitionEnergies": [],
        },
        "transitions": {},
        "electronStateID": 0,
    }
    assert mbData.energyResolution == 1e-16

    # Add fixed transition energies
    fixedTransitionEnergies = np.linspace(1.0, 3.0, 3)
    mbData = crm.CRMModelboundData(grid, fixedTransitionEnergies)

    np.testing.assert_array_equal(
        mbData.fixedTransitionEnergies, fixedTransitionEnergies
    )

    assert mbData.dict()["inelasticGridData"] == {
        "active": True,
        "fixedTransitionEnergies": fixedTransitionEnergies.tolist(),
    }

    # Registering transitions in a new mbData object
    na = vc.Variable("na", grid)
    a = Species("a", 1, associatedVars=[na])

    nb = vc.Variable("nb", grid)
    b = Species("b", 2, associatedVars=[nb])

    transitionEnergy = 13.7
    transitionRate = 99.0

    transition = crm.SimpleTransition("trans", a, b, transitionEnergy, transitionRate)

    mbData = crm.CRMModelboundData(grid)

    mbData.addTransition(transition)

    assert mbData.getTransitionIndices(transition.name) == [1]

    assert mbData.dict() == {
        "modelboundDataType": "modelboundCRMData",
        "transitionTags": [transition.name],
        "inelasticGridData": {
            "active": True,
            "fixedTransitionEnergies": [transitionEnergy],
        },
        "transitions": {transition.name: transition.dict()},
        "electronStateID": 0,
    }

    # Raise Error if fixed transition energies are closer apart than a threshold value (default 1e-16)

    with pytest.raises(AssertionError) as e_info:
        mbData = crm.CRMModelboundData(
            grid,
            fixedTransitionEnergies=np.array([1e-16, 2e-16]),
            energyResolution=1e-16,
        )
    assert (
        e_info.value.args[0]
        == "fixedTransitionEnergies in ModelboundCRMData contain elements closer than allowed energy resolution"
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


def test_derived_transition(grid: Grid):

    na = vc.Variable("na", grid)
    a = Species("a", 1, associatedVars=[na])

    nb = vc.Variable("nb", grid)
    b = Species("b", 2, associatedVars=[nb])

    rateDeriv = dv.SimpleDerivation("rateDeriv", 1.0, [1.0, 1.0])
    rateDerivClosure = dv.DerivationClosure(rateDeriv, na, nb)

    energyDeriv = dv.SimpleDerivation("energyDeriv", 1.0, [2.0, 2.0])
    energyDerivClosure = dv.DerivationClosure(energyDeriv, na, nb)

    transitionEnergy = 10.0

    # Derived transition with a derived rate and energy
    trans = crm.DerivedTransition(
        "derivTrans",
        inStates=[a],
        outStates=[b],
        rateDeriv=rateDerivClosure,
        energyRateDeriv=dv.DerivationClosure(energyDeriv, na, nb),
        transitionEnergy=transitionEnergy,
    )

    assert trans.dict() == {
        "type": "derivedTransition",
        "ingoingStates": [a.speciesID],
        "outgoingStates": [b.speciesID],
        "fixedEnergy": transitionEnergy,
        "ruleName": rateDeriv.name,
        "requiredVarNames": [arg.name for arg in rateDerivClosure.__args__],
        "momentumRateDerivationRule": "none",
        "momentumRateDerivationReqVarNames": [],
        "energyRateDerivationRule": energyDeriv.name,
        "energyRateDerivationReqVarNames": [na.name, nb.name],
    }

    # Register the transition's derivations
    # Method 1: in a textbook
    tb = dv.Textbook(grid)

    trans.registerDerivs(tb)

    tbWithDeriv = {
        "tags": [rateDeriv.name, energyDeriv.name],
        rateDeriv.name: rateDeriv.dict(),
        energyDeriv.name: energyDeriv.dict(),
    }

    assert tb.dict()["customDerivations"] == tbWithDeriv

    # Method 2: via CRMModelboundData.registerDerivs(textbook)
    tb = dv.Textbook(grid)

    mbData = crm.CRMModelboundData(grid)

    mbData.addTransition(trans)

    mbData.registerDerivs(tb)

    assert tb.dict()["customDerivations"] == tbWithDeriv

    # Bad cases

    # Raise error if trying to build a DerivedTransition without an energy equation or fixed energy
    with pytest.raises(AssertionError) as e_info:
        crm.DerivedTransition(
            "derivTrans",
            inStates=[a],
            outStates=[b],
            rateDeriv=rateDerivClosure,
        )
    assert (
        e_info.value.args[0]
        == "DerivedTransition must either have the energy rate derivation or a fixed energy"
    )

    # Raise error if trying to build a DerivedTransition with a partial closure
    # ...for density rate
    with pytest.raises(AssertionError) as e_info:
        crm.DerivedTransition(
            "derivTrans",
            inStates=[a],
            outStates=[b],
            rateDeriv=dv.DerivationClosure(rateDeriv, na),
            energyRateDeriv=energyDerivClosure,
        )
    assert (
        e_info.value.args[0] == "rateDeriv must be a full closure in DerivedTransition"
    )
    # ...for energy rate
    with pytest.raises(AssertionError) as e_info:
        crm.DerivedTransition(
            "derivTrans",
            inStates=[a],
            outStates=[b],
            rateDeriv=rateDerivClosure,
            energyRateDeriv=dv.DerivationClosure(energyDeriv, na),
        )
    assert (
        e_info.value.args[0]
        == "energyRateDeriv must be a full closure in DerivedTransition"
    )


def test_simple_transition(grid: Grid):

    na = vc.Variable("na", grid)
    a = Species("a", 1, associatedVars=[na])

    nb = vc.Variable("nb", grid)
    b = Species("b", 2, associatedVars=[nb])

    transitionEnergy = 13.7
    transitionRate = 99.0

    assert crm.SimpleTransition(
        "trans", a, b, transitionEnergy, transitionRate
    ).dict() == {
        "type": "simpleTransition",
        "ingoingState": a.speciesID,
        "outgoingState": b.speciesID,
        "fixedEnergy": transitionEnergy,
        "rate": transitionRate,
    }

    # Raise error if Transition rate is negative
    with pytest.raises(AssertionError) as e_info:
        crm.SimpleTransition("trans", a, b, transitionEnergy, transitionRate=-1.0)
    assert (
        e_info.value.args[0]
        == "Negative transition rate in SimpleTransition not allowed"
    )


def test_janev_transitions(grid: Grid):

    # TODO: Test mbData.varNames and mbData.getRate, which each return the "rate(...)index(...)" format name for a given transition
    # mbData.getRate needs to be changed to point to the correct dictionary key.

    temperature = vc.Variable("T", grid)
    electronDistribution = vc.Variable("f", grid, isDistribution=True)

    mbData = crm.CRMModelboundData(grid)

    lowestCellEnergy = 0

    energyNorm = 1.0

    maxState = 10

    crm.addJanevTransitionsToCRMData(
        mbData,
        maxState,
        energyNorm,
        electronDistribution,
        temperature,
        lowestCellEnergy,
    )

    for fixedEnergyIndex, tag in enumerate(mbData.transitionTags):
        startState, endState = -1, -1
        if "JanevEx" in tag:
            startState, endState = map(int, tag.replace("JanevEx", "").split("-"))

            trans = crm.CollExIonJanevTransition(
                tag,
                startState,
                endState,
                energyNorm,
                electronDistribution,
                lowestCellEnergy,
            )

        if "JanevDeex" in tag:
            startState, endState = map(int, tag.replace("JanevDeex", "").split("-"))

            directTransitionName = "JanevEx" + str(endState) + "-" + str(startState)

            trans = crm.CollDeexRecombJanevTransition(
                tag,
                startState,
                endState,
                energyNorm,
                electronDistribution,
                temperature,
                directTransitionName,
                crmData=mbData,
                lowestCellEnergy=lowestCellEnergy,
                csUpdatePriority=0,
            )

        if "JanevIon" in tag:
            startState = int(tag.replace("JanevIon", ""))
            endState = 0

            trans = crm.CollExIonJanevTransition(
                tag,
                startState,
                endState,
                energyNorm,
                electronDistribution,
                lowestCellEnergy,
            )

        if "JanevRecomb" in tag:
            if "Rad" in tag:
                # JanevRecombRad
                endState = int(tag.replace("JanevRecombRad", ""))

                directTransitionName = "JanevRecombRad" + str(endState)

                trans = crm.RadRecombJanevTransition(tag, endState, temperature)

            else:
                # JanevRecomb
                startState, endState = map(
                    int, tag.replace("JanevRecomb", "").split("b")
                )

                directTransitionName = "JanevIon" + str(endState)

                trans = crm.CollDeexRecombJanevTransition(
                    tag,
                    0,
                    endState,
                    energyNorm,
                    electronDistribution,
                    temperature,
                    directTransitionName,
                    crmData=mbData,
                    lowestCellEnergy=lowestCellEnergy,
                    csUpdatePriority=0,
                )

        # Transition fixedEnergyIndex must be set before dict() is called
        trans.setFixedEnergyIndex(fixedEnergyIndex)

        assert mbData.dict()["transitions"][tag] == trans.dict()

        # Kinetic electron term generators

        includedTransitionIndices = mbData.getTransitionIndices(tag)

        # Boltzmann term generator

        evolvedHarmonic = 1

        for absorptionTerms in [False, True]:

            # The Boltzmann term generator supports the following transition types:

            if isinstance(
                trans,
                (
                    crm.FixedECSTransition,
                    crm.DetailedBalanceTransition,
                    crm.CollExIonJanevTransition,
                    crm.CollDeexRecombJanevTransition,
                ),
            ):

                termGen = crm.CRMBoltzTermGenerator(
                    "exBoltz",
                    electronDistribution,
                    evolvedHarmonic,
                    includedTransitionIndices,
                    mbData,
                    absorptionTerms=absorptionTerms,
                )

                generatorDict = crm.CRMTermGenerator(
                    "exBoltz",
                    evolvedSpecies=[],
                    includedTransitionIndices=includedTransitionIndices,
                ).dict()

                assert termGen.dict() == {
                    "implicitGroups": [1],
                    "generalGroups": [],
                    "type": "CRMFixedBoltzmannCollInt",
                    "evolvedHarmonic": evolvedHarmonic,
                    "distributionVarName": electronDistribution.name,
                    "includedTransitionIndices": includedTransitionIndices,
                    "fixedEnergyIndices": [
                        mbData.transitions[ind - 1].fixedEnergyIndex + 1
                        for ind in includedTransitionIndices
                    ],
                    "absorptionTerm": absorptionTerms,
                    "detailedBalanceTerm": all(
                        isinstance(
                            mbData.transitions[ind - 1],
                            (
                                crm.DetailedBalanceTransition,
                                crm.CollDeexRecombJanevTransition,
                            ),
                        )
                        for ind in includedTransitionIndices
                    ),
                    "associatedVarIndex": 1,
                }

                assert termGen.evolvedVars == [termGen.__distribution__.name]

                assert (
                    termGen.onlyEvolving(electronDistribution).dict() == termGen.dict()
                )

            # Bad case - invalid transition class used for the Boltzmann term generator

            else:
                with pytest.raises(AssertionError) as e_info:
                    termGen = crm.CRMBoltzTermGenerator(
                        "exBoltz",
                        electronDistribution,
                        evolvedHarmonic,
                        includedTransitionIndices,
                        mbData,
                        absorptionTerms=absorptionTerms,
                    )
                assert (
                    e_info.value.args[0]
                    == "CRMBoltzTermGenerator can only be called with transition indices all corresponding to FixedECSTransitions or DetailedBalanceTransitions"
                )

        # Secondary electron term generator

        termGen = crm.CRMSecElTermGenerator(
            "secEl",
            electronDistribution,
            includedTransitionIndices,
        )

        assert termGen.dict() == {
            "implicitGroups": [1],
            "generalGroups": [],
            "type": "CRMSecondaryElectronTerms",
            "distributionVarName": electronDistribution.name,
            "includedTransitionIndices": includedTransitionIndices,
        }

        assert termGen.evolvedVars == [electronDistribution.name]
