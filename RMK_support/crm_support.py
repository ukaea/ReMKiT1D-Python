from typing import Union, List, Dict, cast
import numpy as np
import csv
import warnings
from typing import Tuple
from .simple_containers import TermGenerator


class ModelboundCRMData:
    """Property container of modelbound CRM data"""

    def __init__(
        self,
        fixedTransitionEnergies=np.array([]),
        energyResolution: float = 1e-16,
        elState: int = 0,
    ) -> None:
        """ModelboundCRMData constructor

        Args:
            fixedTransitionEnergies (np.ndarray, optional): Allowed fixed transition energies for construction of data for inelastic transitions on velocity grid. Defaults to [].
            energyResolution (float, optional): Minimum allowed absolute difference between elements of fixedTransitionEnergies. Defaults to 1e-12.
            elState (int, optional): State ID to treat as the electrons. Defaults to 0.
        """

        uniqueTransitionEnergies: List[float] = []
        for energy in fixedTransitionEnergies:
            assert all(
                abs(uniqueTransitionEnergies - energy) > energyResolution
            ), "fixedTransitionEnergies in ModelboundCRMData contain elements closer than allowed energy resolution"
            uniqueTransitionEnergies.append(energy)

        self.__fixedTransitionEnergies__ = fixedTransitionEnergies
        self.__inelGridData__ = {
            "active": len(fixedTransitionEnergies) > 0,
            "fixedTransitionEnergies": fixedTransitionEnergies.tolist(),
        }
        self.__transitionTags__: List[str] = []
        self.__transitionProperties__: Dict[str, object] = {}
        self.__energyResolution__ = energyResolution
        self.__elStateID__ = elState

    def addTransitionEnergy(self, transitionEnergy: float):
        """Add a transition energy to the list of fixed transition energies allowed in CRM modelbound data. If the energy is within
        energyResolution of another value, it is not added"

        Args:
            transitionEnergy (float): Value of the energy to be added.
        """

        if all(
            abs(self.__fixedTransitionEnergies__ - transitionEnergy)
            > self.__energyResolution__
        ):
            self.__fixedTransitionEnergies__ = np.append(
                self.__fixedTransitionEnergies__, [transitionEnergy]
            )
            self.__inelGridData__.update(
                {
                    "fixedTransitionEnergies": self.__fixedTransitionEnergies__.tolist(),
                    "active": True,
                }
            )

    def addTransition(self, transitionTag: str, transitionProperties: dict):
        """Add transition with given tag and properties.

        Args:
            transitionTag (str): Transition tag
            transitionProperties (dict): Dictionary with transition properties
        """

        self.__transitionTags__.append(transitionTag)
        self.__transitionProperties__[transitionTag] = transitionProperties

    @property
    def fixedTransitionEnergies(self):
        return self.__fixedTransitionEnergies__

    @property
    def transitionTags(self):
        return self.__transitionTags__

    @property
    def transitionProperties(self):
        return self.__transitionProperties__

    @property
    def energyResolution(self):
        return self.__energyResolution__

    def dict(self):
        """Returns dictionary form of ModelboundCRMData to be used in json output

        Returns:
            dict: Dictionary form of ModelboundCRMData to be used to update model properties
        """
        mbData = {
            "modelboundDataType": "modelboundCRMData",
            "transitionTags": self.__transitionTags__,
            "inelasticGridData": self.__inelGridData__,
            "transitions": {},
            "electronStateID": self.__elStateID__,
        }

        mbData["transitions"].update(self.__transitionProperties__)

        return mbData

    def getTransitionIndicesAndEnergies(
        self, prefix: str
    ) -> "Tuple[List[int],List[int]]":
        """Return transition indices and corresponding fixed energy indices of transitions whose tags start with given prefix

        Args:
            prefix (str): Prefix used to search transition tags

        Returns:
            Tuple[List[int],List[int]]: Tuple containing the list of transition indices and corresponding transition energies for all transitions starting with given prefix
        """

        transitionIndices = [
            i + 1 for i, x in enumerate(self.transitionTags) if x.startswith(prefix)
        ]

        transitionEnergyIndices = [
            (
                self.transitionProperties[tag]["fixedEnergyIndex"]
                if "fixedEnergyIndex" in self.transitionProperties[tag].keys()
                else 0
            )
            for tag in [self.transitionTags[ind - 1] for ind in transitionIndices]
        ]

        if 0 in transitionEnergyIndices:
            warnings.warn(
                "getTransitionIndicesAndEnergies was unable to find some transition energy indices"
            )

        return transitionIndices, transitionEnergyIndices


def simpleTransition(
    inState: int, outState: int, transitionEnergy: float, transitionRate: float
) -> dict:
    """Return transition properties for a simple transition - single ingoing/outgoing state, and fixed transition energy and rate

    Args:
        inState (int): Single species ID corresponding to ingoing state
        outState (int): Single species ID corresponding to outgoing state
        transitionEnergy (float): Fixed transition energy
        transitionRate (float): Fixed transition rate. Must be positive

    Returns:
        dict: Simple transition properties dictionary to be added to ModelboundCRMData object
    """

    assert (
        transitionRate > 0
    ), "Negative transition rate in simpleTransition not allowed"

    tProperties = {
        "type": "simpleTransition",
        "ingoingState": inState,
        "outgoingState": outState,
        "fixedEnergy": transitionEnergy,
        "rate": transitionRate,
    }

    return tProperties


def derivedTransition(
    inStates: List[int],
    outStates: List[int],
    transitionEnergy: float,
    ruleName: str,
    requiredVars: List[str],
    momentumRuleName="none",
    momentumRequiredVars: List[str] = [],
    energyRuleName="none",
    energyRequiredVars: List[str] = [],
) -> dict:
    """Return transition properties for a derived transition, where the rate is calculated using a derivation rule
        Optionally can calculate momentum and energy rates using rules as well.

    Args:
        inStates (List[int]): List of ingoing species IDs
        outStates (List[int]): List of outgoing species IDs
        transitionEnergy (float): Fixed transition energy
        ruleName (str): Derivation rule name used for transition rate calculation
        requiredVars (List[str]): Names of required variables for transition rate calculation
        momentumRuleName (str, optional): Derivation rule name used for transition momentum rate calculation. Defaults to "none"
        momentumRequiredVars (List[str], optional): Names of required variables for transition momentum rate calculation. Defaults to []
        energyRuleName (str, optional): Derivation rule name used for transition energy rate calculation. Defaults to "none"
        energyRequiredVars (List[str], optional): Names of required variables for transition energy rate calculation. Defaults to []

    Returns:
        dict: Derived transition properties dictionary to be added to ModelboundCRMData object
    """

    tProperties = {
        "type": "derivedTransition",
        "ingoingStates": inStates,
        "outgoingStates": outStates,
        "fixedEnergy": transitionEnergy,
        "ruleName": ruleName,
        "requiredVarNames": requiredVars,
        "momentumRateDerivationRule": momentumRuleName,
        "momentumRateDerivationReqVarNames": momentumRequiredVars,
        "energyRateDerivationRule": energyRuleName,
        "energyRateDerivationReqVarNames": energyRequiredVars,
    }

    return tProperties


def fixedECSTransition(
    inStates: List[int],
    outStates: List[int],
    transitionEnergy: float,
    fixedTransitionEnergies: np.ndarray,
    csData: List[Tuple[int, np.ndarray]],
    distributionVarName: str,
    takeMomentumMoment=False,
    energyResolution: float = 1e-16,
) -> dict:
    """Return fixed energy and cross-section transition properties

    Args:
        inStates (List[int]): List of ingoing species IDs
        outStates (List[int]): List of outgoing species IDs
        transitionEnergy (float): Fixed transition energy
        fixedTransitionEnergies (np.array): Array of energies used to look up index of allowed transition energy closest to transitionEnergy
        csData (List[Tuple[int,np.ndarray]]]): List of tuples of the form (l,crossSection), where l is the l harmonic number and crossSection is the corresponding cross section harmonic
        distributionVarName (str): Name of the distribution function variables used to get rates with corresponding csData
        takeMomentumMoment (bool, optional): If true, will calculate momentum rate based on l=1 harmonics of crossSection and distribution function. Defaults to False.
        energyResolution (float, optional): Tolerance for finding transitionEnergy in fixedTransitionEnergies. Defaults to 1e-12.

    Returns:
        dict: Fixed energy/cross-section transition properties dictionary to be added to ModelboundCRMData object
    """

    energyIndex, usedTransitionEnergy = min(
        enumerate(fixedTransitionEnergies), key=lambda x: abs(x[1] - transitionEnergy)
    )
    assert (
        abs(usedTransitionEnergy - transitionEnergy) < energyResolution
    ), "fixedECSTransition unable to find energy in fixedTransitionEnergies within energyResolution of passed transitionEnergy"

    presentCSHarmonics = [x[0] for x in csData]

    csDataDict = {
        "presentHarmonics": presentCSHarmonics,
    }

    for data in csData:
        csDataDict["l=" + str(data[0])] = data[1].tolist()

    tProperties = {
        "type": "fixedECSTransition",
        "ingoingStates": inStates,
        "outgoingStates": outStates,
        "fixedEnergyIndex": energyIndex + 1,
        "distributionVarName": distributionVarName,
        "takeMomentumMoment": takeMomentumMoment,
        "crossSectionData": csDataDict,
    }

    return tProperties


def variableECSTransition(
    inStates: List[int],
    outStates: List[int],
    csDerivs: List[Tuple[int, np.ndarray]],
    energyDeriv: dict,
    distributionVarName: str,
    takeMomentumMoment=False,
) -> dict:
    """Return variable energy and cross-section transition properties

    Args:
        inStates (List[int]): List of ingoing species IDs
        outStates (List[int]): List of outgoing species IDs
        csDerivs (List[Tuple[int,np.ndarray]]]): List of tuples of the form (l,derivationRule), where l is the l harmonic number and derivationRule is the corresponding cross section derivation rule
        energyDeriv (dict): derivationRule for the transition energy
        distributionVarName (str): Name of the distribution function variables used to get rates
        takeMomentumMoment (bool, optional): If true, will calculate momentum rate based on l=1 harmonics of crossSection and distribution function. Defaults to False.

    Returns:
        dict: Variable energy/cross-section transition properties dictionary to be added to ModelboundCRMData object
    """

    presentCSHarmonics = [x[0] for x in csDerivs]

    csDataDict: Dict[str, object] = {
        "crossSectionDerivationHarmonics": presentCSHarmonics,
    }

    for data in csDerivs:
        csDataDict["l=" + str(data[0])] = {}
        cast(Dict[str, object], csDataDict["l=" + str(data[0])]).update(data[1])

    tProperties = {
        "type": "variableECSTransition",
        "ingoingStates": inStates,
        "outgoingStates": outStates,
        "distributionVarName": distributionVarName,
        "takeMomentumMoment": takeMomentumMoment,
        "crossSectionDerivations": csDataDict,
        "energyDerivationName": energyDeriv["ruleName"],
        "energyDerivationReqVars": energyDeriv["requiredVarNames"],
    }

    return tProperties


def detailedBalanceTransition(
    inStates: List[int],
    outStates: List[int],
    directTransitionEnergy: float,
    fixedTransitionEnergies: np.ndarray,
    directTransitionIndex: int,
    maxResolvedCSHarmonic: int,
    distributionVarName: str,
    temperatureVarName: str,
    degeneracyRatio: float,
    degeneracyRule="none",
    degeneracyRuleReqVars: List[str] = [],
    takeMomentumMoment=False,
    energyResolution: float = 1e-16,
    csUpdatePriority=0,
) -> dict:
    """Return detailed balance transition property dictionary. Requires information on corresponding direct transition and is defined
       for electron impact processes.

    Args:
        inStates (List[int]): List of ingoing species IDs (into this transition, not the corresponding direct transition)
        outStates (List[int]): List of outgoing species IDs
        directTransitionEnergy (float): Energy of the direct transition (the generated transition will have -1 times this)
        fixedTransitionEnergies (np.array): Array of energies used to look up index of allowed transition energy closest to directTransitionEnergy
        directTransitionIndex (int): Index of the direct transition in the host modelbound CRM data object (Fortran 1-indexing!)
        maxResolvedCSHarmonic (int): Highest harmonic to be calculated (should not exceed highest present harmonic in direct transition)
        distributionVarName (str): Name of the distribution function variable
        temperatureVarName (str): Name of the temperature variable associated with the above distribution function (used to ensure numerical detailed balance)
        degeneracyRatio (float): Degeneracy ratio of final and initial states in this transition
        degeneracyRule (str, optional): Rule used to calculate any variable dependent part of the degeneracy ratio. Defaults to "none".
        degeneracyRuleReqVars (List[str], optional): Variables used in degeneracyRule. Defaults to [].
        takeMomentumMoment (bool, optional): If true, will calculate momentum rate based on l=1 harmonics of crossSection and distribution function. Defaults to False.
        energyResolution (float, optional): Tolerance for finding transitionEnergy in fixedTransitionEnergies. Defaults to 1e-12.
        csUpdatePriority (int, optional): Update priority of the detailed balance cross-section. Defaults to 0 (highest priority)

    Returns:
        dict: Detailed balance transition property dictionary to be added into ModelboundCRMData object
    """

    dirTransitionEnergyIndex, usedTransitionEnergy = min(
        enumerate(fixedTransitionEnergies),
        key=lambda x: abs(x[1] - directTransitionEnergy),
    )
    transitionEnergyIndex, dummyVar = min(
        enumerate(fixedTransitionEnergies),
        key=lambda x: abs(x[1] + directTransitionEnergy),
    )
    assert (
        abs(usedTransitionEnergy - directTransitionEnergy) < energyResolution
    ), "fixedECSTransition unable to find energy in fixedTransitionEnergies within energyResolution of passed directTransitionEnergy"

    tProperties = {
        "type": "detailedBalanceTransition",
        "ingoingStates": inStates,
        "outgoingStates": outStates,
        "directTransitionFixedEnergyIndex": dirTransitionEnergyIndex,
        "fixedEnergyIndex": transitionEnergyIndex + 1,
        "directTransitionIndex": directTransitionIndex + 1,
        "distributionVarName": distributionVarName,
        "electronTemperatureVar": temperatureVarName,
        "fixedDegeneracyRatio": degeneracyRatio,
        "degeneracyRuleName": degeneracyRule,
        "degeneracyRuleReqVars": degeneracyRuleReqVars,
        "takeMomentumMoment": takeMomentumMoment,
        "maxCrossSectionL": maxResolvedCSHarmonic,
        "crossSectionUpdatePriority": csUpdatePriority,
    }

    return tProperties


def radRecombJanevTransition(endState: int, temperatureVarName: str) -> dict:
    """Return transition property dictionary corresponding to radiative recombination transition into endState of hydrogen based on Janev fit.

    Args:
        endState (int): State into which hydrogenic ion recombines
        temperatureVarName (str): Name of electron temperature variable

    Returns:
        dict: Radiative recombination transition property dictionary based on Janev fit to be added into ModelboundCRMData object
    """

    tProperties = {
        "type": "JanevRadRecomb",
        "endHState": endState,
        "electronTemperatureVar": temperatureVarName,
    }

    return tProperties


def collExIonJanevTransition(
    startState: int,
    endState: int,
    energyNorm: float,
    distributionVarName: str,
    fixedTransitionEnergies: np.ndarray,
    energyResolution: float = 1e-16,
    lowestCellEnergy: float = 0,
) -> dict:
    """Return transition property dictionary corresponding to collisional excitation/ionization transition from startState into endState of hydrogen based on Janev fit.

    Args:
        startState (int): Starting state of hydrogen
        endState (int): Final state of hydrogen (0 assumes ion)
        energyNorm (float): Normalization constant in eV used for transition energy (in general same as temperature normalization)
        distributionVarName (str): Name of the distribution funcion variable
        fixedTransitionEnergies (np.array): Array of energies used to look up index of allowed transition energy closest to computed transition energy
        energyResolution (float, optional): Tolerance for finding transitionEnergy in fixedTransitionEnergies. Defaults to 1e-12.
        lowestCellEnergy (float, optional): Energy of lowest cell for secondary electron generation in normalized units. Defaults to 0 but should be supplied.

    Returns:
        dict: Collisional excitation/ionization transition property dictionary based on Janev fit to be added into ModelboundCRMData object
    """

    if endState > 0:
        transitionEnergy = 13.6 * (1 / startState**2 - 1 / endState**2) / energyNorm
    else:
        transitionEnergy = 13.6 * (1 / startState**2) / energyNorm + lowestCellEnergy

    transitionIndex, usedTransitionEnergy = min(
        enumerate(fixedTransitionEnergies), key=lambda x: abs(x[1] - transitionEnergy)
    )
    assert (
        abs(usedTransitionEnergy - transitionEnergy) < energyResolution
    ), "collExIonJanevTransition unable to find energy in fixedTransitionEnergies within energyResolution of calculated transitionEnergy"

    tProperties = {
        "type": "JanevCollExIon",
        "startHState": startState,
        "endHState": endState,
        "fixedEnergyIndex": transitionIndex + 1,
        "distributionVarName": distributionVarName,
    }

    return tProperties


def collDeexRecombJanevTransition(
    startState: int,
    endState: int,
    energyNorm: float,
    distributionVarName: str,
    temperatureVarName: str,
    directTransitionIndex: int,
    fixedTransitionEnergies: np.ndarray,
    energyResolution: float = 1e-16,
    csUpdatePriority=0,
    lowestCellEnergy: float = 0,
) -> dict:
    """Return transition property dictionary corresponding to collisional deexcitation/recombination transition from startState into endState of hydrogen based on Janev fit.

    Args:
        startState (int): Starting state of hydrogen
        endState (int): Final state of hydrogen (0 assumes ion)
        energyNorm (float): Normalization constant in eV used for transition energy (in general same as temperature normalization)
        distributionVarName (str): Name of the distribution funcion variable
        temperatureVarName (str): Name of the temperature variable associated with the above distribution function (used to ensure numerical detailed balance)
        directTransitionIndex (int): Index of the direct transition in the host modelbound CRM data object (Fortran 1-indexing!)
        fixedTransitionEnergies (np.array): Array of energies used to look up index of allowed transition energy closest to computed transition energy
        energyResolution (float, optional): Tolerance for finding transitionEnergy in fixedTransitionEnergies. Defaults to 1e-12.
        csUpdatePriority (int, optional): Update priority of the detailed balance cross-section. Defaults to 0 (highest priority)
        lowestCellEnergy (float, optional): Energy of lowest cell for secondary electron generation in normalized units. Defaults to 0 but should be supplied.

    Returns:
        dict: Collisional deexcitation/recombination transition property dictionary based on Janev fit to be added into ModelboundCRMData object
    """

    if startState > 0:
        directTransitionEnergy = (
            -13.6 * (1 / startState**2 - 1 / endState**2) / energyNorm
        )
    else:
        directTransitionEnergy = (
            13.6 * (1 / endState**2) / energyNorm + lowestCellEnergy
        )

    dirTransitionEnergyIndex, usedTransitionEnergy = min(
        enumerate(fixedTransitionEnergies),
        key=lambda x: abs(x[1] - directTransitionEnergy),
    )
    transitionEnergyIndex, dummyVar = min(
        enumerate(fixedTransitionEnergies),
        key=lambda x: abs(x[1] + directTransitionEnergy),
    )
    assert (
        abs(usedTransitionEnergy - directTransitionEnergy) < energyResolution
    ), "collDeexRecombJanevTransition unable to find energy in fixedTransitionEnergies within energyResolution of calculated directTransitionEnergy"

    tProperties = {
        "type": "JanevCollDeexRecomb",
        "startHState": startState,
        "endHState": endState,
        "directTransitionFixedEnergyIndex": dirTransitionEnergyIndex + 1,
        "directTransitionIndex": directTransitionIndex,
        "fixedEnergyIndex": transitionEnergyIndex + 1,
        "distributionVarName": distributionVarName,
        "electronTemperatureVar": temperatureVarName,
        "crossSectionUpdatePriority": csUpdatePriority,
    }

    return tProperties


def addJanevTransitionsToCRMData(
    mbData: ModelboundCRMData,
    maxState: int,
    energyNorm: float,
    distributionVarName: Union[str, None] = None,
    temperatureVarName: Union[str, None] = None,
    detailedBalanceCSPriority=0,
    processes=["ex", "deex", "ion", "recomb3b", "recombRad"],
    lowestCellEnergy: float = 0,
) -> None:
    allowedProcesses = ["ex", "deex", "ion", "recomb3b", "recombRad"]

    for process in processes:
        assert (
            process in allowedProcesses
        ), "process passed to addJanevTransitionsToCRMData not support"

        if process == "ex":
            assert (
                distributionVarName is not None
            ), "distributionVarName must be specified if excitation is added using addJanevTransitionsToCRMData"

            for startState in range(1, maxState + 1):
                for endState in range(startState + 1, maxState + 1):
                    transitionEnergy = (
                        13.6 * (1 / startState**2 - 1 / endState**2) / energyNorm
                    )
                    mbData.addTransitionEnergy(transitionEnergy)
                    transitionTag = "JanevEx" + str(startState) + "-" + str(endState)
                    transitionProperties = collExIonJanevTransition(
                        startState,
                        endState,
                        energyNorm,
                        distributionVarName,
                        mbData.fixedTransitionEnergies,
                        mbData.energyResolution,
                    )
                    mbData.addTransition(
                        transitionTag=transitionTag,
                        transitionProperties=transitionProperties,
                    )

        if process == "ion":
            assert (
                distributionVarName is not None
            ), "distributionVarName must be specified if ionization is added using addJanevTransitionsToCRMData"

            for startState in range(1, maxState + 1):
                transitionEnergy = (
                    13.6 * (1 / startState**2) / energyNorm + lowestCellEnergy
                )
                mbData.addTransitionEnergy(transitionEnergy)
                transitionTag = "JanevIon" + str(startState)
                transitionProperties = collExIonJanevTransition(
                    startState,
                    0,
                    energyNorm,
                    distributionVarName,
                    mbData.fixedTransitionEnergies,
                    mbData.energyResolution,
                    lowestCellEnergy=lowestCellEnergy,
                )
                mbData.addTransition(
                    transitionTag=transitionTag,
                    transitionProperties=transitionProperties,
                )
        if process == "deex":
            assert (
                "ex" in processes
            ), "If deexcitation added using addJanevTransitionsToCRMData so must be excitation"
            assert processes.index(process) > processes.index(
                "ex"
            ), "ex must be before deex in processes list"
            assert (
                distributionVarName is not None
            ), "distributionVarName must be specified if deexcitation is added using addJanevTransitionsToCRMData"
            assert (
                temperatureVarName is not None
            ), "temperatureVarName must be specified if deexcitation is added using addJanevTransitionsToCRMData"

            for endState in range(1, maxState + 1):
                for startState in range(endState + 1, maxState + 1):
                    transitionEnergy = (
                        13.6 * (1 / startState**2 - 1 / endState**2) / energyNorm
                    )
                    mbData.addTransitionEnergy(transitionEnergy)
                    transitionTag = "JanevDeex" + str(startState) + "-" + str(endState)
                    directTransitionTag = (
                        "JanevEx" + str(endState) + "-" + str(startState)
                    )
                    transitionProperties = collDeexRecombJanevTransition(
                        startState,
                        endState,
                        energyNorm,
                        distributionVarName,
                        temperatureVarName,
                        directTransitionIndex=mbData.transitionTags.index(
                            directTransitionTag
                        )
                        + 1,
                        fixedTransitionEnergies=mbData.fixedTransitionEnergies,
                        energyResolution=mbData.energyResolution,
                        csUpdatePriority=detailedBalanceCSPriority,
                    )
                    mbData.addTransition(
                        transitionTag=transitionTag,
                        transitionProperties=transitionProperties,
                    )

        if process == "recomb3b":
            assert (
                "ion" in processes
            ), "If 3-body recombination added using addJanevTransitionsToCRMData so must be ionization"
            assert processes.index(process) > processes.index(
                "ion"
            ), "ion must be before recomb3b in processes list"
            assert (
                distributionVarName is not None
            ), "distributionVarName must be specified if recomb3b is added using addJanevTransitionsToCRMData"
            assert (
                temperatureVarName is not None
            ), "temperatureVarName must be specified if recomb3b is added using addJanevTransitionsToCRMData"

            for endState in range(1, maxState + 1):
                transitionEnergy = (
                    -13.6 * (1 / endState**2) / energyNorm - lowestCellEnergy
                )
                mbData.addTransitionEnergy(transitionEnergy)
                transitionTag = "JanevRecomb3b" + str(endState)
                directTransitionTag = "JanevIon" + str(endState)
                transitionProperties = collDeexRecombJanevTransition(
                    0,
                    endState,
                    energyNorm,
                    distributionVarName,
                    temperatureVarName,
                    directTransitionIndex=mbData.transitionTags.index(
                        directTransitionTag
                    )
                    + 1,
                    fixedTransitionEnergies=mbData.fixedTransitionEnergies,
                    energyResolution=mbData.energyResolution,
                    csUpdatePriority=detailedBalanceCSPriority,
                    lowestCellEnergy=lowestCellEnergy,
                )
                mbData.addTransition(
                    transitionTag=transitionTag,
                    transitionProperties=transitionProperties,
                )

        if process == "recombRad":
            assert (
                temperatureVarName is not None
            ), "temperatureVarName must be specified if radiative recombination is added using addJanevTransitionsToCRMData"

            for endState in range(1, maxState + 1):
                transitionTag = "JanevRecombRad" + str(endState)
                transitionProperties = radRecombJanevTransition(
                    endState, temperatureVarName
                )
                mbData.addTransition(
                    transitionTag=transitionTag,
                    transitionProperties=transitionProperties,
                )


def addHSpontaneousEmissionToCRMData(
    mbData: ModelboundCRMData,
    transitionData: dict,
    maxStartState: int,
    maxEndState: int,
    timeNorm: float,
    energyNorm: float,
):
    """Adds hydrogen spontaneous emission transitions to CRM data based on passed dictionary.

    Args:
        transitionData (dict): Dictionary with keys of form (startState,endState), and with values in (s^-1)
        maxStartState (int): Highest starting state to add
        maxEndState (int): Highest final state to add
        timeNorm (float): Time normalization in s
        energyNorm (float): Energy normalization in eV
    """

    for endState in range(1, maxEndState + 1):
        for startState in range(endState + 1, maxStartState + 1):
            transitionEnergy = 13.6 * (1 / endState**2 - 1 / startState**2) / energyNorm
            assert (
                startState,
                endState,
            ) in transitionData.keys(), "(startState,endState) pair not found in transitionData keys for spontaneous emission"
            transitionTag = "SpontEmissionH" + str(startState) + "-" + str(endState)
            transitionProperties = simpleTransition(
                startState,
                endState,
                transitionEnergy,
                transitionData[(startState, endState)] * timeNorm,
            )
            mbData.addTransition(
                transitionTag=transitionTag, transitionProperties=transitionProperties
            )


def readNISTAkiCSV(filename: str) -> dict:
    res = {}
    with open(filename, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row["Aki(s^-1)"] != "Aki(s^-1)":
                rate = float(row["Aki(s^-1)"][2:-1])
                strStartState = row["conf_k"][2:-1]
                strEndState = row["conf_i"][2:-1]
                if strEndState == "1s":
                    strEndState = "1"
                if strStartState.isnumeric() and strEndState.isnumeric():
                    res[(int(strStartState), int(strEndState))] = rate

    return res


def hydrogenSahaBoltzmann(
    numStates: int,
    temperature: float,
    totDens: float,
    fixedIonizationDegree: Union[None, float] = None,
):
    """Return density distribution of electrons and atomic hydrogen states based on Saha-Boltzmann

    Args:
        numStates (int): Number of atomic hydrogen states
        temperature (float): Electron temperature in eV
        totDens (float): Total density (neutrals+plasma)
        fixedIonizationDegree (Union[None,float], optional): Optional ionization degree. Defaults to None, which calculate Saha equilibrium
    """
    hPlanck = 6.62607004e-34
    elMass = 9.10938e-31
    elCharge = 1.60218e-19
    hydrogenIonPot = 13.6

    vPlanck = (hPlanck**2 / (2 * np.pi * elMass * elCharge * temperature)) ** (-3 / 2)
    g = sum(
        [
            i**2 * np.exp(-hydrogenIonPot * (1 - 1 / i**2) / temperature)
            for i in range(1, numStates + 1)
        ]
    )

    A1 = vPlanck * np.exp(-hydrogenIonPot / temperature) / g
    A1 = A1 / totDens

    X = (A1 * np.sqrt(1 + 4 / A1) - A1) / 2
    if fixedIonizationDegree is not None:
        X = fixedIonizationDegree
    ne = totDens * X
    n1 = (1 - X) * totDens / g

    densDist = [ne]

    for i in range(0, numStates):
        densDist.append(
            n1
            * (i + 1) ** 2
            * np.exp(-hydrogenIonPot * (1 - 1 / (i + 1) ** 2) / temperature)
        )

    return densDist


def termGeneratorCRM(
    implicitTermGroups: List[int] = [1],
    evolvedSpeciesIDs: List[int] = [],
    includedTransitionIndices: List[int] = [],
) -> TermGenerator:
    """Return dictionary with implicit CRM density evolution term generator properties. The term generator must be added to a model
        which has a CRM modelbound data object

    Args:
        implicitTermGroups (List[int], optional): Implicit term group of host model to which the generator should add terms. Defaults to [1].
        evolvedSpeciesIDs (List[int], optional): List of species IDs whose density evolution terms should be added. Defaults to [], which is
                                                interpreted as all species.
        includedTransitionIndices (List[int], optional): List of transition indices in modelbound CRM which should be included in rate calculations
                                                        by this generator. Defaults to [], which is interpreted as all transitions.
    Returns:
        TermGenerator: CRM term generator ready to be added to custom model
    """

    crmTermGeneratorOptions: Dict[str, object] = {
        "type": "CRMDensityEvolution",
        "evolvedSpeciesIDs": evolvedSpeciesIDs,
        "includedTransitionIndices": includedTransitionIndices,
    }

    return TermGenerator(implicitTermGroups, [], crmTermGeneratorOptions)


def termGeneratorCRMElEnergy(
    electronEnergyDensVar: str,
    implicitTermGroups: List[int] = [1],
    includedTransitionIndices: List[int] = [],
) -> TermGenerator:
    """Return dictionary with implicit CRM electron energy evolution term generator properties. The term generator must be added to a model
        which has a CRM modelbound data object

    Args:
        electronEnergyDensVar (str): Name of the evolved electron energy variable
        implicitTermGroups (List[int], optional): Implicit term group of host model to which the generator should add terms. Defaults to [1].
        includedTransitionIndices (List[int], optional): List of transition indices in modelbound CRM which should be included in rate calculations
                                                        by this generator. Defaults to [], which is interpreted as all transitions.

    Returns:
         TermGenerator: CRM electron energy evolution term generator ready to be added to custom model
    """

    crmElEnergyTermGeneratorOptions: Dict[str, object] = {
        "type": "CRMElectronEnergyEvolution",
        "electronEnergyDensity": electronEnergyDensVar,
        "includedTransitionIndices": includedTransitionIndices,
    }

    return TermGenerator(implicitTermGroups, [], crmElEnergyTermGeneratorOptions)


def termGeneratorCRMBoltz(
    distributionVarName: str,
    evolvedHarmonic: int,
    includedTransitionIndices: List[int],
    fixedEnergyIndices: List[int],
    implicitTermGroups: List[int] = [1],
    associatedVarIndex=1,
    absorptionTerms=False,
    detailedBalanceTerms=False,
) -> TermGenerator:
    """Return dictionary with term generator for Boltzmann collision terms based on CRM modelbound data

    Args:
        distributionVarName (str): Name of the electorn distribution variable
        evolvedHarmonic (int): Harmonic evolved by these terms
        includedTransitionIndices (List[int]): Included transition indices (should be either fixed ECS or detailed balance transitions)
        fixedEnergyIndices (List[int]): Fixed energy indices corresponding to included transition indices
        implicitTermGroups (List[int], optional): Implicit term group of host model to which the generator should add terms. Defaults to [1].
        associatedVarIndex (int, optional): Index of required density variables in species associated variables. Defaults to 1.
        absorptionTerms (bool, optional): Set to true if the generated terms are absorption terms. Defaults to False.
        detailedBalanceTerms (bool, optional): Set to true if the generated terms are detailed balance terms. Defaults to False.

    Returns:
        TermGenerator: CRM Boltzmann term generator
    """

    crmBoltzGeneratorOptions: Dict[str, object] = {
        "type": "CRMFixedBoltzmannCollInt",
        "evolvedHarmonic": evolvedHarmonic,
        "distributionVarName": distributionVarName,
        "includedTransitionIndices": includedTransitionIndices,
        "fixedEnergyIndices": fixedEnergyIndices,
        "absorptionTerm": absorptionTerms,
        "detailedBalanceTerm": detailedBalanceTerms,
        "associatedVarIndex": associatedVarIndex,
    }

    return TermGenerator(implicitTermGroups, [], crmBoltzGeneratorOptions)


def termGeneratorCRMSecEl(
    distributionVarName: str,
    includedTransitionIndices: List[int] = [],
    implicitTermGroups: List[int] = [1],
) -> TermGenerator:
    """Return term generator property dictionary for secondary electron kinetic term generators, which put secondary electrons generated
    in given transitions into the lowest energy cell

    Args:
        distributionVarName (str): Name of the evolved electron distribution variable
        includedTransitionIndices (List[int], optional): Included transition indices. Defaults to [], resulting in all transitions.
        implicitTermGroups (List[int], optional): Implicit term group of host model to which the generator should add terms. Defaults to [1].

    Returns:
        TermGenerator: CRM term generator for kinetic secondary electron generation terms
    """

    secElGeneratorOptions: Dict[str, object] = {
        "type": "CRMSecondaryElectronTerms",
        "distributionVarName": distributionVarName,
        "includedTransitionIndices": includedTransitionIndices,
    }

    return TermGenerator(implicitTermGroups, [], secElGeneratorOptions)


def termGeneratorVarCRMBoltz(
    distributionVarName: str,
    evolvedHarmonic: int,
    includedTransitionIndices: List[int],
    implicitTermGroups: List[int] = [1],
    associatedVarIndex=1,
    absorptionTerms=False,
    superelasticTerms=False,
) -> TermGenerator:
    """Return dictionary with term generator for variable mapping Boltzmann collision terms based on CRM modelbound data

    Args:
        distributionVarName (str): Name of the electorn distribution variable
        evolvedHarmonic (int): Harmonic evolved by these terms
        includedTransitionIndices (List[int]): Included transition indices (should all be variable ECS transitions)
        implicitTermGroups (List[int], optional): Implicit term group of host model to which the generator should add terms. Defaults to [1].
        associatedVarIndex (int, optional): Index of required density variables in species associated variables. Defaults to 1.
        absorptionTerms (bool, optional): Set to true if the generated terms are absorption terms. Defaults to False.
        superelasticTerms (bool, optional): Set to true if the generated terms are superelastic terms. Defaults to False.

    Returns:
        TermGenerator: CRM Boltzmann term generator
    """

    crmBoltzGeneratorOptions: Dict[str, object] = {
        "type": "CRMFixedBoltzmannCollInt",
        "evolvedHarmonic": evolvedHarmonic,
        "distributionVarName": distributionVarName,
        "includedTransitionIndices": includedTransitionIndices,
        "absorptionTerm": absorptionTerms,
        "superelasticTerm": superelasticTerms,
        "associatedVarIndex": associatedVarIndex,
    }

    return TermGenerator(implicitTermGroups, [], crmBoltzGeneratorOptions)
