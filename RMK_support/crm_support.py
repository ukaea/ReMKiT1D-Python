from typing import Union, List, Dict, cast, Optional
from typing_extensions import Self
import numpy as np
import csv
from typing import Tuple
from .derivations import Species, Textbook, DerivationClosure
from .grid import Grid, Profile
from .model_construction import ModelboundData, TermGenerator
from .variable_container import Variable
from abc import ABC, abstractmethod
from .tex_parsing import numToScientificTex
import pylatex as tex  # type: ignore


class Transition(ABC):
    """Abstract transition class for use in CRM modelbound data"""

    def __init__(
        self,
        name: str,
        inStates: List[Species],
        outStates: List[Species],
        hasMomentumRate=False,
    ):
        """Abstract transition class

        Args:
            name (str): Name of the transition
            inStates (List[Species]): Species corresponding to ingoing states
            outStates (List[Species]): Species corresponding to outgoing states
            hasMomentumRate (bool, optional): True if the transition has a momentum transfer rate associated to it. Defaults to False.
        """
        self.__name__ = name
        self.__inStates__ = inStates
        self.__outStates__ = outStates
        self.__hasMomentumRate__ = hasMomentumRate

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name: str):
        self.__name__ = name

    @property
    def inStates(self):
        return self.__inStates__

    @property
    def outStates(self):
        return self.__outStates__

    @abstractmethod
    def dict(self) -> Dict:
        pass

    def latex(self, **kwargs) -> str:
        reactants = "+".join([species.latex() for species in self.inStates])
        products = "+".join([species.latex() for species in self.outStates])
        return "$" + reactants + "\\rightarrow" + products + "$"

    def registerDerivs(self, container: Textbook):
        pass

    @property
    def fixedEnergy(self) -> Optional[float]:
        return None

    @property
    def hasMomentumRate(self):
        return self.__hasMomentumRate__

    def setFixedEnergyIndex(self, ind: int):
        pass

    @property
    def fixedEnergyIndex(self):
        return None


class CRMModelboundData(ModelboundData):
    """Property container of modelbound CRM data"""

    def __init__(
        self,
        grid: Grid,
        fixedTransitionEnergies=np.array([]),
        energyResolution: float = 1e-16,
        elState: int = 0,
    ) -> None:
        """ModelboundCRMData constructor

        Args:
            grid (Grid): Grid object used in constructing modelbound variables
            fixedTransitionEnergies (np.ndarray, optional): Allowed fixed transition energies for construction of data for inelastic transitions on velocity grid. Defaults to []. This gets automatically filled when adding transitions.
            energyResolution (float, optional): Minimum allowed absolute difference between elements of fixedTransitionEnergies. Defaults to 1e-16.
            elState (int, optional): State ID to treat as the electrons. Defaults to 0.
        """

        self.__grid__ = grid
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
        self.__transitions__: List[Transition] = []
        self.__energyResolution__ = energyResolution
        self.__elStateID__ = elState

    def addTransitionEnergy(self, transitionEnergy: float):
        """Add a transition energy to the list of fixed transition energies allowed in CRM modelbound data. If the energy is within energyResolution of another value, it is not added"

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

    def addTransition(self, transition: Transition):
        """Add a transition to the CRM data, including its fixed transition energy if it has one

        Args:
            transition (Transition): Transition to be added
        """
        assert (
            transition.name not in self.transitionTags
        ), "Duplicate transition tag in CRMModelboundData"

        self.__transitions__.append(transition)
        if transition.fixedEnergy is not None:
            self.addTransitionEnergy(transition.fixedEnergy)
            energyIndex, _ = min(
                enumerate(self.__fixedTransitionEnergies__.tolist()),
                key=lambda x: abs(x[1] - transition.fixedEnergy),
            )
            transition.setFixedEnergyIndex(energyIndex)

    @property
    def fixedTransitionEnergies(self):
        return self.__fixedTransitionEnergies__

    @property
    def transitionTags(self):
        return [t.name for t in self.__transitions__]

    @property
    def transitions(self):
        return self.__transitions__

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
            "transitionTags": self.transitionTags,
            "inelasticGridData": self.__inelGridData__,
            "transitions": {},
            "electronStateID": self.__elStateID__,
        }

        mbData["transitions"].update({t.name: t.dict() for t in self.__transitions__})

        return mbData

    def getTransitionIndices(self, prefix: str) -> List[int]:
        """Return transition indices of transitions whose tags start with given prefix

        Args:
            prefix (str): Prefix used to search transition tags

        Returns:
            List[int]:List of transition indices for all transitions starting with given prefix
        """

        return [
            i + 1 for i, x in enumerate(self.transitionTags) if x.startswith(prefix)
        ]

    @property
    def varNames(self):
        varNames = []
        for i, t in enumerate(self.__transitions__):
            varNames += (
                [
                    "rate0index" + str(i + 1),
                    "rate1index" + str(i + 1),
                    "rate2index" + str(i + 1),
                ]
                if t.hasMomentumRate
                else ["rate0index" + str(i + 1), "rate2index" + str(i + 1)]
            )

        return varNames

    def __getitem__(self, key):
        if key not in self.varNames:
            raise KeyError()
        return Variable(key, self.__grid__, isDerived=True)

    def getRate(self, transition: Union[str, Transition], moment: int = 0) -> Variable:
        """Return rate associated with a given transition as a Variable

        Args:
            transition (Union[str, Transition]): Name of transition or transition object to get the associated rate of
            moment (int, optional): Moment associated with the rate - 0 for particle/reaction rate, 1 for momentum, 2 for energy. Defaults to 0.

        """
        transitionName = ""
        if isinstance(transition, Transition):
            transitionName = transition.name
        else:
            transitionName = transition

        assert transitionName in self.transitionTags, (
            "getRate called with unregistered transition " + transitionName
        )

        return self[
            "rate" + str(moment) + "index" + self.transitionTags.index(transitionName)
        ]

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        doc.append("CRM Modelbound data")
        with doc.create(tex.Subsubsection("Transitions")):
            with doc.create(tex.Itemize()) as itemize:
                for transition in self.transitions:
                    itemize.add_item(
                        tex.NoEscape(
                            transition.name
                            + ": "
                            + transition.latex(latexRemap=latexRemap)
                        )
                    )

    def registerDerivs(self, container):
        for transition in self.transitions:
            transition.registerDerivs(container)


class SimpleTransition(Transition):
    """A transition with a fixed rate and transition energy and a single ingoing and outgoing state"""

    def __init__(
        self,
        name: str,
        inState: Species,
        outState: Species,
        transitionEnergy: float,
        transitionRate: float,
    ):
        """A transition with a fixed rate and transition energy and a single ingoing and outgoing state. The energy rate is obtainer using the fixed energy and transition rates.

        Args:
            name (str): Name of the transition
            inState (Species): Species representing the ingoing state
            outState (Species): Species representing the outgoing state
            transitionEnergy (float): Fixed transition energy
            transitionRate (float): Fixed transition rate
        """
        self.__transitionEnergy__ = transitionEnergy
        assert (
            transitionRate > 0
        ), "Negative transition rate in SimpleTransition not allowed"
        self.__transitionRate__ = transitionRate
        super().__init__(name, [inState], [outState])

    @property
    def fixedEnergy(self):
        return self.__transitionEnergy__

    def dict(self):

        return {
            "type": "simpleTransition",
            "ingoingState": self.inStates[0].speciesID,
            "outgoingState": self.outStates[0].speciesID,
            "fixedEnergy": self.fixedEnergy,
            "rate": self.__transitionRate__,
        }

    def latex(self, **kwargs):
        equation = super().latex(**kwargs)
        equation += "\\newline "
        equation += (
            "Rate: $" + numToScientificTex(self.__transitionRate__) + "$\\newline "
        )
        equation += "Energy: $" + numToScientificTex(self.fixedEnergy) + "$"
        return equation


class DerivedTransition(Transition):
    """Transition where rates are calculated using derivation objects"""

    def __init__(
        self,
        name: str,
        inStates: List[Species],
        outStates: List[Species],
        rateDeriv: DerivationClosure,
        **kwargs
    ):
        """Transition where rates are calculated using transition objects

        Args:
            name (str): Name of the transition
            inStates (List[Species]): List of species representing ingoing states
            outStates (List[Species]): List of species representing outgoing states
            rateDeriv (DerivationClosure): Full derivation closure encapsulating the reaction rate derivation rule

        kwargs:

            energyRateDeriv (DerivationClosure): Full derivation closure encapsulating the reaction energy rate derivation rule. Must be present if transitionEnergy isn't

            transitionEnergy (float): Fixed transition energy. Must be present if energyRateDeriv isn't (if both are present the derivation takes precedence in ReMMKiT1D)

            momentumRateDeriv (DerivationClosure): Full derivation closure encapsulating the reaction momentum rate derivation rule.
        """

        self.__rateDeriv__ = rateDeriv
        assert (
            rateDeriv.numArgs == 0
        ), "rateDeriv must be a full closure in DerivedTransition"
        assert (
            "energyRateDeriv" in kwargs or "transitionEnergy" in kwargs
        ), "DerivedTransition must either have the energy rate derivation or a fixed energy"
        self.__energyRateDeriv__: Optional[DerivationClosure] = kwargs.get(
            "energyRateDeriv", None
        )
        if self.__energyRateDeriv__ is not None:
            assert (
                self.__energyRateDeriv__.numArgs == 0
            ), "energyRateDeriv must be a full closure in DerivedTransition"

        self.__transitionEnergy__: Optional[float] = kwargs.get(
            "transitionEnergy", None
        )

        self.__momentumRateDeriv__: Optional[DerivationClosure] = kwargs.get(
            "momentumRateDeriv", None
        )
        if self.__momentumRateDeriv__ is not None:
            assert (
                self.__momentumRateDeriv__.numArgs == 0
            ), "momentumRateDeriv must be a full closure in DerivedTransition"
        super().__init__(
            name, inStates, outStates, self.__momentumRateDeriv__ is not None
        )

    def registerDerivs(self, container):
        self.__rateDeriv__.registerComponents(container)
        if self.__energyRateDeriv__ is not None:
            self.__energyRateDeriv__.registerComponents(container)
        if self.__momentumRateDeriv__ is not None:
            self.__momentumRateDeriv__.registerComponents(container)

    @property
    def fixedEnergy(self):
        return self.__transitionEnergy__

    def dict(self):
        return {
            "type": "derivedTransition",
            "ingoingStates": [species.speciesID for species in self.inStates],
            "outgoingStates": [species.speciesID for species in self.outStates],
            "fixedEnergy": self.fixedEnergy if self.fixedEnergy is not None else 0.0,
            "ruleName": self.__rateDeriv__.name,
            "requiredVarNames": self.__rateDeriv__.fillArgs(),
            "momentumRateDerivationRule": (
                self.__momentumRateDeriv__.name
                if self.__momentumRateDeriv__ is not None
                else "none"
            ),
            "momentumRateDerivationReqVarNames": (
                self.__momentumRateDeriv__.fillArgs()
                if self.__momentumRateDeriv__ is not None
                else []
            ),
            "energyRateDerivationRule": (
                self.__energyRateDeriv__.name
                if self.__energyRateDeriv__ is not None
                else "none"
            ),
            "energyRateDerivationReqVarNames": (
                self.__energyRateDeriv__.fillArgs()
                if self.__energyRateDeriv__ is not None
                else []
            ),
        }

    def latex(self: Self, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        equation = super().latex(**kwargs)
        equation += "\\newline "
        remappedArgs = (
            (
                latexRemap[arg]
                if arg in latexRemap
                else "\\text{" + arg.replace("_", r"\_") + "}"
            )
            for arg in self.__rateDeriv__.fillArgs()
        )
        equation += "Rate: $" + self.__rateDeriv__.latex(*remappedArgs) + "$\\newline "
        if self.__momentumRateDeriv__ is not None:
            remappedArgs = (
                (
                    latexRemap[arg]
                    if arg in latexRemap
                    else "\\text{" + arg.replace("_", r"\_") + "}"
                )
                for arg in self.__momentumRateDeriv__.fillArgs()
            )
            equation += (
                "Momentum rate: $"
                + self.__momentumRateDeriv__(*remappedArgs)
                + "$\\newline "
            )
        if self.__energyRateDeriv__ is not None:
            remappedArgs = (
                (
                    latexRemap[arg]
                    if arg in latexRemap
                    else "\\text{" + arg.replace("_", r"\_") + "}"
                )
                for arg in self.__energyRateDeriv__.fillArgs()
            )
            equation += "Energy rate: $" + self.__energyRateDeriv__(*remappedArgs) + "$"
        else:
            equation += "Energy: $" + numToScientificTex(self.fixedEnergy) + "$"

        return equation


class FixedECSTransition(Transition):
    """Electron-impact transition with fixed energy and cross-section"""

    def __init__(
        self,
        name: str,
        inStates: List[Species],
        outStates: List[Species],
        transitionEnergy: float,
        csData: Dict[int, Profile],
        electronDistribution: Variable,
        takeMomentumMoment=False,
    ):
        """Electron-impact transition with fixed energy and cross-section

        Args:
            name (str): Name of transition
            inStates (List[Species]): List of species representing ingoing states
            outStates (List[Species]): List of species representing outgoing states
            transitionEnergy (float): Fixed transition energy
            csData (Dict[int, Profile]): Cross section data - a dictionary with (l-harmonic number, velocity space profile) key value pairs corresponding to the Legendre harmonic decomposition of the reaction cross-section
            electronDistribution (Variable): Variable corresponding to the electron distribution
            takeMomentumMoment (bool, optional): If true will take the momentum transfer harmonic and the transition will have a momentum transfer rate. Defaults to False.
        """
        self.__transitionEnergy__ = transitionEnergy
        self.__csData__ = csData
        assert all(
            cs.dim == "V" for _, cs in csData.items()
        ), "csData must be made up of velocity profiles"
        self.__electronDistribution__ = electronDistribution
        self.__energyIndex__: Optional[int] = None
        assert (
            electronDistribution.isDistribution
        ), "electronDistribution variable in FixedECSTranstion must be a distribution variable"
        super().__init__(name, inStates, outStates, takeMomentumMoment)

    @property
    def fixedEnergy(self):
        return self.__transitionEnergy__

    def setFixedEnergyIndex(self, ind):
        self.__energyIndex__ = ind

    @property
    def fixedEnergyIndex(self):
        return self.__energyIndex__

    def dict(self):
        presentCSHarmonics = list(self.__csData__.keys())

        csDataDict = {
            "presentHarmonics": presentCSHarmonics,
        }

        for l, cs in self.__csData__.items():
            csDataDict["l=" + str(l)] = cs.data.tolist()

        tProperties = {
            "type": "fixedECSTransition",
            "ingoingStates": [s.speciesID for s in self.inStates],
            "outgoingStates": [s.speciesID for s in self.outStates],
            "fixedEnergyIndex": self.__energyIndex__ + 1,
            "distributionVarName": self.__electronDistribution__.name,
            "takeMomentumMoment": self.__hasMomentumRate__,
            "crossSectionData": csDataDict,
        }

        return tProperties

    def latex(self, **kwargs):
        expression = super().latex(**kwargs)
        expression += "\\newline Fixed energy/cross-section transition"
        expression += "\\newline Energy: $" + numToScientificTex(self.fixedEnergy) + "$"
        return expression


class VariableECSTransition(Transition):
    """Electron-impact transition with variable energy and cross-section"""

    def __init__(
        self,
        name: str,
        inStates: List[Species],
        outStates: List[Species],
        csDerivs: Dict[int, DerivationClosure],
        energyDeriv: DerivationClosure,
        electronDistribution: Variable,
        takeMomentumMoment=False,
    ):
        """Electron-impact transition with variable energy and cross-section

        Args:
            name (str): Name of transition
            inStates (List[Species]): List of species representing ingoing states
            outStates (List[Species]): List of species representing outgoing states
            csDerivs (Dict[int, DerivationClosure]): Cross section data derivations - a dictionary with (l-harmonic number, single harmonic derivation closure) key value pairs corresponding to the Legendre harmonic decomposition of the reaction cross-section
            energyDeriv (DerivationClosure): Full derivation closure corresponding to the reaction energy
            electronDistribution (Variable): Variable corresponding to the electron distribution
            takeMomentumMoment (bool, optional): If true will take the momentum transfer harmonic and the transition will have a momentum transfer rate. Defaults to False.
        """
        assert (
            electronDistribution.isDistribution
        ), "electronDistribution variable in FixedECSTranstion must be a distribution variable"

        self.__electronDistribution__ = electronDistribution

        self.__csDerivs__ = csDerivs
        for l in csDerivs:
            assert (
                csDerivs[l].numArgs == 0
            ), "All csDerivs in VariableECSTransition must be complete closures"
        assert (
            energyDeriv.numArgs == 0
        ), "energyDeriv in VariableECSTransition must be complete closure"

        self.__energyDeriv__ = energyDeriv

        super().__init__(name, inStates, outStates, takeMomentumMoment)

    def dict(self: Self):

        presentCSHarmonics = list(self.__csDerivs__.keys())

        csDataDict: Dict[str, object] = {
            "crossSectionDerivationHarmonics": presentCSHarmonics,
        }

        for l, deriv in self.__csDerivs__.items():
            csDataDict["l=" + str(l)] = {}
            cast(Dict[str, object], csDataDict["l=" + str(l)]).update(
                {"ruleName": deriv.name, "requiredVarNames": deriv.fillArgs()}
            )

        tProperties = {
            "type": "variableECSTransition",
            "ingoingStates": [s.speciesID for s in self.inStates],
            "outgoingStates": [s.speciesID for s in self.outStates],
            "distributionVarName": self.__electronDistribution__.name,
            "takeMomentumMoment": self.hasMomentumRate,
            "crossSectionDerivations": csDataDict,
            "energyDerivationName": self.__energyDeriv__.name,
            "energyDerivationReqVars": self.__energyDeriv__.fillArgs(),
        }

        return tProperties

    def latex(self: Self, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        equation = super().latex(**kwargs)
        equation += "\\newline Variable energy/cross-section transition"
        remappedArgs = (
            (
                latexRemap[arg]
                if arg in latexRemap
                else "\\text{" + arg.replace("_", r"\_") + "}"
            )
            for arg in self.__energyDeriv__.fillArgs()
        )
        equation += "\\newline Energy: $" + self.__energyDeriv__(*remappedArgs) + "$"

    def registerDerivs(self, container):
        for _, deriv in self.__csDerivs__.items():
            deriv.registerComponents(container)
        self.__energyDeriv__.registerComponents(container)


class DetailedBalanceTransition(Transition):
    """Detailed balance transition corresponding to a direct fixed energy/cross-section electron-impact transition"""

    def __init__(
        self,
        name: str,
        directTransitionName: str,
        crmData: CRMModelboundData,
        temperature: Variable,
        electronDistribution: Variable,
        degeneracyRatio: float,
        maxResolvedCSHarmonic: int,
        **kwargs
    ):
        """Detailed balance transition corresponding to a direct fixed energy/cross-section electron-impact transition

        Args:
            name (str): Name of the transition
            directTransitionName (str): Name of the direct transition which this one balances
            crmData (CRMModelboundData): CRM modelbound data housing the direct transition
            temperature (Variable): Variable corresponding to electron temperature
            electronDistribution (Variable): Variable corresponding to the electron distribution function
            degeneracyRatio (float): Degeneracy ratio of final and initial states in this transition
            maxResolvedCSHarmonic (int): Highest harmonic to be calculated (should not exceed highest present harmonic in direct transition)

        kwargs:

            takeMomentumMoment (bool): If true, will calculate momentum rate based on l=1 harmonics of cross section and distribution function. Defaults to False.

            csUpdatePriority (int): Update priority of the detailed balance cross-section. Defaults to 0 (highest priority)

            degeneracyDeriv (DerivationClosure): Full derivation rule corresponding to the part of the degeneracy ratio that depends on variables. Defaults to None
        """

        self.__takeMomentumMoment__ = kwargs.get("takeMomentumMoment", False)
        self.__directTransition__ = crmData.transitions[
            crmData.transitionTags.index(directTransitionName)
        ]
        self.__temperature__ = temperature
        self.__electronDistribution__ = electronDistribution
        self.__degeneracyRatio__ = degeneracyRatio
        self.__maxResolvedCSHarmonics__ = maxResolvedCSHarmonic
        self.__csUpdatePriority__: int = kwargs.get("csUpdatePriority", 0)
        self.__degeneracyDeriv__: Optional[DerivationClosure] = kwargs.get(
            "degeneracyDeriv", None
        )
        self.__fixedEnergy__ = -self.__directTransition__.fixedEnergy
        self.__directTransitionIndex__ = crmData.transitionTags.index(
            directTransitionName
        )
        self.__directTransitionEnergyIndex__ = (
            self.__directTransition__.fixedEnergyIndex
        )
        self.__energyIndex__: Optional[int] = None
        super().__init__(
            name,
            self.__directTransition__.outStates,
            self.__directTransition__.inStates,
            self.__takeMomentumMoment__,
        )

    @property
    def fixedEnergy(self):
        return self.__fixedEnergy__

    def setFixedEnergyIndex(self, ind):
        self.__energyIndex__ = ind

    @property
    def fixedEnergyIndex(self):
        return self.__energyIndex__

    def dict(self):
        return {
            "type": "detailedBalanceTransition",
            "ingoingStates": [s.speciesID for s in self.inStates],
            "outgoingStates": [s.speciesID for s in self.outStates],
            "directTransitionFixedEnergyIndex": self.__directTransitionEnergyIndex__,
            "fixedEnergyIndex": self.fixedEnergyIndex + 1,
            "directTransitionIndex": self.__directTransitionIndex__ + 1,
            "distributionVarName": self.__electronDistribution__.name,
            "electronTemperatureVar": self.__temperature__.name,
            "fixedDegeneracyRatio": self.__degeneracyRatio__,
            "degeneracyRuleName": (
                self.__degeneracyDeriv__.name
                if self.__degeneracyDeriv__ is not None
                else ""
            ),
            "degeneracyRuleReqVars": (
                self.__degeneracyDeriv__.fillArgs()
                if self.__degeneracyDeriv__ is not None
                else []
            ),
            "takeMomentumMoment": self.hasMomentumRate,
            "maxCrossSectionL": self.__maxResolvedCSHarmonics__,
            "crossSectionUpdatePriority": self.__csUpdatePriority__,
        }

    def latex(self, **kwargs):
        equation = super().latex(**kwargs)
        equation += (
            "\\newline Detailed balance transition with direct transition "
            + self.__directTransition__.name
        )
        return equation


class RadRecombJanevTransition(Transition):
    """Radiative recombination transition for hydrogen based on Janev rate fit"""

    def __init__(self, name: str, endState: int, temperature: Variable):
        """Radiative recombination transition for hydrogen based on Janev rate fit

        Args:
            name (str): Name of this transition
            endState (int): Principal quantum number of final state
            temperature (Variable): Electron temperature variable
        """
        self.__endState__ = endState
        self.__temperature__ = temperature
        super().__init__(name, [], [])

    def dict(self):
        return {
            "type": "JanevRadRecomb",
            "endHState": self.__endState__,
            "electronTemperatureVar": self.__temperature__.name,
        }

    def latex(self, **kwargs):
        expression = (
            "$\\text{H}^{+} + e^{-} \\rightarrow \\text{H}(n="
            + str(self.__endState__)
            + ") + h\\nu$"
        )
        expression += "\\newline Janev radiative recombination transition"
        return expression


class CollExIonJanevTransition(Transition):
    """Electron-impact excitation/ionisation transition for hydrogen based on Janev cross-section fits"""

    def __init__(
        self,
        name: str,
        startState: int,
        endState: int,
        energyNorm: float,
        electronDistribution: Variable,
        lowestCellEnergy: float = 0,
    ):
        """Electron-impact excitation/ionisation transition based on Janev cross-section fits

        Args:
            name (str): Name of this transition
            startState (int): Initial state principle quantum number
            endState (int): Final state principle quantum number (if 0 assumes ionisation)
            energyNorm (float): Normalisation of energy in eV
            electronDistribution (Variable): Electron distribution variable
            lowestCellEnergy (float, optional): Energy associated to the lowest velocity space cell (in normalised units) used to account for finite secondary electron energies. Defaults to 0.
        """
        self.__startState__ = startState
        self.__endState__ = endState
        self.__energyNorm__ = energyNorm
        self.__electronDistribution__ = electronDistribution
        self.__lowestCellEnergy__ = lowestCellEnergy
        self.__energyIndex__: Optional[int] = None
        if endState > 0:
            self.__transitionEnergy__ = (
                13.6 * (1 / startState**2 - 1 / endState**2) / energyNorm
            )
        else:
            self.__transitionEnergy__ = (
                13.6 * (1 / startState**2) / energyNorm + lowestCellEnergy
            )
        super().__init__(name, [], [])

    @property
    def fixedEnergy(self):
        return self.__transitionEnergy__

    def setFixedEnergyIndex(self, ind):
        self.__energyIndex__ = ind

    @property
    def fixedEnergyIndex(self):
        return self.__energyIndex__

    def dict(self):
        return {
            "type": "JanevCollExIon",
            "startHState": self.__startState__,
            "endHState": self.__endState__,
            "fixedEnergyIndex": self.fixedEnergyIndex + 1,
            "distributionVarName": self.__electronDistribution__.name,
        }

    def latex(self, **kwargs):
        if self.__endState__ == 0:

            expression = (
                "$\\text{H}(n="
                + str(self.__startState__)
                + ") + e^{-} \\rightarrow \\text{H}^{+} + e^{-}  + e^{-} $"
            )
            expression += "\\newline Janev ionization transition"
            return expression

        expression = (
            "$\\text{H}(n="
            + str(self.__startState__)
            + ") + e^{-} \\rightarrow \\text{H}(n="
            + str(self.__endState__)
            + ") + e^{-}  $"
        )
        expression += "\\newline Janev excitation transition"
        return expression


class CollDeexRecombJanevTransition(Transition):
    """Electron-impact de-excitation/recombination transition for hydrogen based on Janev cross-section fits"""

    def __init__(
        self,
        name: str,
        startState: int,
        endState: int,
        energyNorm: float,
        electronDistribution: Variable,
        temperature: Variable,
        directTransitionName: str,
        crmData: CRMModelboundData,
        lowestCellEnergy: float = 0,
        csUpdatePriority: int = 0,
    ):
        """Electron-impact de-excitation/recombination transition for hydrogen based on Janev cross-section fits and detailed balance

        Args:
            name (str): Name of this transition
            startState (int): Initial state principle quantum number (if 0 assumes this is a recombination reaction)
            endState (int): Final state principle quantum number
            energyNorm (float): Energy normalisation in eV
            electronDistribution (Variable): Electron distribution function variable
            temperature (Variable): Electron temperature variable
            directTransitionName (str): Name of the direct transition to which this transition is the inverse
            crmData (CRMModelboundData): CRM modelbound data containing the direct transition
            lowestCellEnergy (float, optional): Energy associated to the lowest velocity space cell (in normalised units) used to account for finite secondary electron energies. Defaults to 0.
            csUpdatePriority (int, optional): Update priority of the detailed balance cross-section. Defaults to 0 (highest priority)
        """
        self.__startState__ = startState
        self.__endState__ = endState
        self.__energyNorm__ = energyNorm
        self.__electronDistribution__ = electronDistribution
        self.__lowestCellEnergy__ = lowestCellEnergy
        self.__temperature__ = temperature
        self.__directTransition__ = crmData.transitions[
            crmData.transitionTags.index(directTransitionName)
        ]
        self.__directTransitionIndex__ = crmData.transitionTags.index(
            directTransitionName
        )
        self.__directTransitionEnergyIndex__ = (
            self.__directTransition__.fixedEnergyIndex
        )
        self.__energyIndex__: Optional[int] = None
        self.__csUpdatePriority__: int = csUpdatePriority
        if startState > 0:
            self.__transitionEnergy__ = (
                13.6 * (1 / startState**2 - 1 / endState**2) / energyNorm
            )
        else:
            self.__transitionEnergy__ = (
                -13.6 * (1 / endState**2) / energyNorm - lowestCellEnergy
            )
        super().__init__(name, [], [])

    @property
    def fixedEnergy(self):
        return self.__transitionEnergy__

    def setFixedEnergyIndex(self, ind):
        self.__energyIndex__ = ind

    @property
    def fixedEnergyIndex(self):
        return self.__energyIndex__

    def dict(self):
        return {
            "type": "JanevCollDeexRecomb",
            "startHState": self.__startState__,
            "endHState": self.__endState__,
            "directTransitionFixedEnergyIndex": self.__directTransitionEnergyIndex__
            + 1,
            "directTransitionIndex": self.__directTransitionIndex__ + 1,
            "fixedEnergyIndex": self.fixedEnergyIndex + 1,
            "distributionVarName": self.__electronDistribution__.name,
            "electronTemperatureVar": self.__temperature__.name,
            "crossSectionUpdatePriority": self.__csUpdatePriority__,
        }

    def latex(self, **kwargs):
        if self.__startState__ == 0:

            expression = (
                "$\\text{H}^{+} + e^{-}  + e^{-}   \\rightarrow  \\text{H}(n="
                + str(self.__endState__)
                + ") + e^{-}$"
            )
            expression += "\\newline Janev three-body recombination transition"
            return expression

        expression = (
            "$\\text{H}(n="
            + str(self.__startState__)
            + ") + e^{-} \\rightarrow \\text{H}(n="
            + str(self.__endState__)
            + ") + e^{-}  $"
        )
        expression += "\\newline Janev de-excitation transition"
        return expression


def addJanevTransitionsToCRMData(
    mbData: CRMModelboundData,
    maxState: int,
    energyNorm: float,
    electronDistribution: Optional[Variable] = None,
    temperature: Optional[Variable] = None,
    detailedBalanceCSPriority=0,
    processes: List[str] = ["ex", "deex", "ion", "recomb3b", "recombRad"],
    lowestCellEnergy: float = 0,
) -> None:
    """Add Janev transitions to CRM modelbound data

    Args:
        mbData (CRMModelboundData): CRM modelbound data to add transitions to
        maxState (int): Highest principle quantum number to add transitions for
        energyNorm (float): Energy normalisation in eV
        electronDistribution (Optional[Variable], optional): Electron distribution function variable used for adding some of the transitions. Defaults to None.
        temperature (Optional[Variable], optional): Electron temperature variable used when adding some of the transitions. Defaults to None.
        detailedBalanceCSPriority (int, optional): Update priority of the detailed balance cross-section for detailed balance transitions if those are added. Defaults to 0.
        processes (List[str], optional): List of processes to add. Defaults to ["ex", "deex", "ion", "recomb3b", "recombRad"] - which are all of the allowed processes.
        lowestCellEnergy (float, optional): Energy associated to the lowest velocity space cell (in normalised units) used to account for finite secondary electron energies. Defaults to 0. Defaults to 0.
    """
    allowedProcesses: List[str] = ["ex", "deex", "ion", "recomb3b", "recombRad"]

    for process in processes:
        assert (
            process in allowedProcesses
        ), "process passed to addJanevTransitionsToCRMData not support"

        if process == "ex":
            assert (
                electronDistribution is not None
            ), "electronDistribution must be specified if excitation is added using addJanevTransitionsToCRMData"

            for startState in range(1, maxState + 1):
                for endState in range(startState + 1, maxState + 1):
                    transitionTag = "JanevEx" + str(startState) + "-" + str(endState)

                    mbData.addTransition(
                        CollExIonJanevTransition(
                            transitionTag,
                            startState,
                            endState,
                            energyNorm,
                            electronDistribution,
                            lowestCellEnergy,
                        )
                    )

        if process == "ion":
            assert (
                electronDistribution is not None
            ), "electronDistribution must be specified if ionization is added using addJanevTransitionsToCRMData"

            for startState in range(1, maxState + 1):

                transitionTag = "JanevIon" + str(startState)
                mbData.addTransition(
                    CollExIonJanevTransition(
                        transitionTag,
                        startState,
                        0,
                        energyNorm,
                        electronDistribution,
                        lowestCellEnergy,
                    )
                )

        if process == "deex":
            assert (
                "ex" in processes
            ), "If deexcitation added using addJanevTransitionsToCRMData so must be excitation"
            assert processes.index(process) > processes.index(
                "ex"
            ), "ex must be before deex in processes list"
            assert (
                electronDistribution is not None
            ), "electronDistribution must be specified if deexcitation is added using addJanevTransitionsToCRMData"
            assert (
                temperature is not None
            ), "temperature must be specified if deexcitation is added using addJanevTransitionsToCRMData"

            for endState in range(1, maxState + 1):
                for startState in range(endState + 1, maxState + 1):
                    transitionTag = "JanevDeex" + str(startState) + "-" + str(endState)
                    directTransitionTag = (
                        "JanevEx" + str(endState) + "-" + str(startState)
                    )

                    mbData.addTransition(
                        CollDeexRecombJanevTransition(
                            transitionTag,
                            startState,
                            endState,
                            energyNorm,
                            electronDistribution,
                            temperature,
                            directTransitionTag,
                            mbData,
                            lowestCellEnergy,
                            csUpdatePriority=detailedBalanceCSPriority,
                        )
                    )

        if process == "recomb3b":
            assert (
                "ion" in processes
            ), "If 3-body recombination added using addJanevTransitionsToCRMData so must be ionization"
            assert processes.index(process) > processes.index(
                "ion"
            ), "ion must be before recomb3b in processes list"
            assert (
                electronDistribution is not None
            ), "electronDistribution must be specified if recomb3b is added using addJanevTransitionsToCRMData"
            assert (
                temperature is not None
            ), "temperature must be specified if recomb3b is added using addJanevTransitionsToCRMData"

            for endState in range(1, maxState + 1):
                transitionTag = "JanevRecomb3b" + str(endState)
                directTransitionTag = "JanevIon" + str(endState)
                mbData.addTransition(
                    CollDeexRecombJanevTransition(
                        transitionTag,
                        0,
                        endState,
                        energyNorm,
                        electronDistribution,
                        temperature,
                        directTransitionTag,
                        mbData,
                        lowestCellEnergy,
                        csUpdatePriority=detailedBalanceCSPriority,
                    )
                )

        if process == "recombRad":
            assert (
                temperature is not None
            ), "temperature must be specified if radiative recombination is added using addJanevTransitionsToCRMData"

            for endState in range(1, maxState + 1):
                transitionTag = "JanevRecombRad" + str(endState)
                mbData.addTransition(
                    RadRecombJanevTransition(transitionTag, endState, temperature)
                )


def addHSpontaneousEmissionToCRMData(
    mbData: CRMModelboundData,
    transitionData: Dict[Tuple[int, int], float],
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
        timeNorm (float): Time normalisation in s
        energyNorm (float): Energy normalisation in eV
    """

    for endState in range(1, maxEndState + 1):
        for startState in range(endState + 1, maxStartState + 1):
            transitionEnergy = 13.6 * (1 / endState**2 - 1 / startState**2) / energyNorm
            assert (
                startState,
                endState,
            ) in transitionData, "(startState,endState) pair not found in transitionData keys for spontaneous emission"
            transitionTag = "SpontEmissionH" + str(startState) + "-" + str(endState)
            mbData.addTransition(
                SimpleTransition(
                    transitionTag,
                    Species(
                        "H" + str(startState),
                        startState,
                        latexName="\\text{H}(n=" + str(startState) + ")",
                    ),
                    Species(
                        "H" + str(endState),
                        endState,
                        latexName="\\text{H}(n=" + str(endState) + ")",
                    ),
                    transitionEnergy,
                    transitionData[(startState, endState)] * timeNorm,
                )
            )


def readNISTAkiCSV(filename: str) -> Dict[Tuple[int, int], float]:
    """Read NIST hydrogen Einstein coefficients from csv file

    Args:
        filename (str): Name of the file containing the coefficients
    """
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
    fixedIonizationDegree: Optional[float] = None,
):
    """Return density distribution of electrons and atomic hydrogen states based on Saha-Boltzmann

    Args:
        numStates (int): Number of atomic hydrogen states
        temperature (float): Electron temperature in eV
        totDens (float): Total density (neutrals+plasma)
        fixedIonizationDegree (float, optional): Optional ionization degree. Defaults to None, which calculate Saha equilibrium
    """
    hPlanck = 6.62607004e-34
    elMass = 9.10938e-31
    elCharge = 1.60218e-19
    hydrogenIonPot = 13.6

    vPlanck = (hPlanck**2 / (2 * np.pi * elMass * elCharge * temperature)) ** (-3 / 2)
    g = sum(
        i**2 * np.exp(-hydrogenIonPot * (1 - 1 / i**2) / temperature)
        for i in range(1, numStates + 1)
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


class CRMTermGenerator(TermGenerator):
    """Term generator for CRM contributions to density equations"""

    def __init__(
        self,
        name: str,
        evolvedSpecies: List[Species],
        implicitGroups: Optional[List[int]] = None,
        includedTransitionIndices: Optional[List[int]] = None,
    ) -> None:
        """Term generator for CRM contributions to density equations - matrix terms

        Args:
            name (str): Name of the generator
            evolvedSpecies (List[Species]): Evolved species - the first associated variable should be the density
            implicitGroups (Optional[List[int]], optional): Implicit groups to put the generated terms into. Defaults to None, using [1].
            includedTransitionIndices (Optional[List[int]], optional): Included transition indices in the CRM modelbound data. Defaults to None - including all transitions.
        """
        self.__evolvedSpecies__ = evolvedSpecies

        for species in evolvedSpecies:
            assert len(species.associatedVarNames) > 0, (
                "Species "
                + species.name
                + " does not have any associated variables - the first variable is interpreted as the density by CRMTermGenerator"
            )
        self.__includedTransitionIndices__ = includedTransitionIndices
        super().__init__(name, implicitGroups if implicitGroups is not None else [1])

    def dict(self) -> Dict:
        tg = super().dict()

        tg.update(
            {
                "type": "CRMDensityEvolution",
                "evolvedSpeciesIDs": [
                    species.speciesID for species in self.__evolvedSpecies__
                ],
                "includedTransitionIndices": (
                    self.__includedTransitionIndices__
                    if self.__includedTransitionIndices__ is not None
                    else []
                ),
            }
        )

        return tg

    @property
    def evolvedVars(self):
        return [sp.associatedVarNames[0] for sp in self.__evolvedSpecies__]

    def onlyEvolving(self, *args: Variable) -> Self:
        evolvedVarNames = [arg.name for arg in args]
        newSpecies: List[Species] = []
        for sp in self.__evolvedSpecies__:
            if sp.associatedVarNames[0] in evolvedVarNames:
                newSpecies.append(sp)
        return cast(
            Self,
            CRMTermGenerator(
                self.name,
                newSpecies,
                self.implicitGroups,
                self.__includedTransitionIndices__,
            ),
        )

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        varNames = ", ".join(
            (
                "$" + latexRemap[var] + "$"
                if var in latexRemap
                else "$\\text{" + var.replace("_", r"\_") + "}$"
            )
            for var in self.evolvedVars
        )
        doc.append(
            tex.NoEscape(
                self.name.replace("_", r"\_")
                + ": \\newline CRM density evolution term generator"
            )
        )
        doc.append(tex.NoEscape("\\newline Evolved densities: " + varNames))
        doc.append(
            tex.NoEscape(
                "\\newline $$\\frac{\\partial \\vec{n}}{\\partial t} = M \\cdot \\vec{n}$$"
            )
        )
        doc.append(
            tex.NoEscape(
                "\\newline Transitions included: "
                + (
                    ", ".join(str(i) for i in self.__includedTransitionIndices__)
                    + " \\newline"
                    if self.__includedTransitionIndices__ is not None
                    else "all \\newline"
                )
            )
        )


class CRMElEnergyTermGenerator(TermGenerator):
    """Term generator of electron energy sinks from CRM contributions"""

    def __init__(
        self,
        name: str,
        electronEnergyDens: Variable,
        implicitGroups: Optional[List[int]] = None,
        includedTransitionIndices: Optional[List[int]] = None,
    ) -> None:
        """Term generator of electron energy sinks from CRM contributions

        Args:
            name (str): Name of the generator
            electronEnergyDens (Variable): Electron energy density variable evolved by terms generated by this generator
            implicitGroups (Optional[List[int]], optional): Implicit groups to put generated terms into. Defaults to None using [1].
            includedTransitionIndices (Optional[List[int]], optional): Included transition indices in the CRM modelbound data. Defaults to None - including all transitions.
        """
        self.__electronEnergyDens__ = electronEnergyDens
        self.__includedTransitionIndices__ = includedTransitionIndices
        super().__init__(name, implicitGroups if implicitGroups is not None else [1])

    def dict(self) -> Dict:
        tg = super().dict()

        tg.update(
            {
                "type": "CRMElectronEnergyEvolution",
                "electronEnergyDensity": self.__electronEnergyDens__.name,
                "includedTransitionIndices": (
                    self.__includedTransitionIndices__
                    if self.__includedTransitionIndices__ is not None
                    else []
                ),
            }
        )

        return tg

    @property
    def evolvedVars(self):
        return [self.__electronEnergyDens__.name]

    def onlyEvolving(self, *args: Variable) -> Self:
        return self

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        doc.append(
            tex.NoEscape(
                self.name.replace("_", r"\_")
                + ": \\newline CRM Electron energy evolution term generator"
            )
        )
        doc.append(
            tex.NoEscape(
                "\\newline Evolved energy variable: $"
                + self.__electronEnergyDens__.latex(latexRemap)
                + "$"
            )
        )
        doc.append(
            tex.NoEscape(
                "\\newline Transitions included: "
                + (
                    ", ".join(str(i) for i in self.__includedTransitionIndices__)
                    + " \\newline"
                    if self.__includedTransitionIndices__ is not None
                    else "all \\newline"
                )
            )
        )


class CRMBoltzTermGenerator(TermGenerator):
    """Generator of Boltzmann collision terms with fixed cross-sections/energies using CRM modelbound data"""

    def __init__(
        self,
        name: str,
        distribution: Variable,
        evolvedHarmonic: int,
        includedTransitionIndices: List[int],
        mbData: CRMModelboundData,
        associatedVarIndex: int = 1,
        absorptionTerms=False,
        implicitGroups: Optional[List[int]] = None,
    ) -> None:
        """Generator of Boltzmann collision terms with fixed cross-sections/energies using CRM modelbound data

        Args:
            name (str): Name of the generator
            distribution (Variable): Electron distribution function variable
            evolvedHarmonic (int): Evolved harmonic index (Fortran 1-indexing)
            includedTransitionIndices (List[int]): Included transition indices (should be fixed energy/cross-section transitions)
            mbData (CRMModelboundData): Modelbound data containing the transitions
            associatedVarIndex (int, optional): Index of density in species associated variable list. Defaults to 1 (Fortran 1-indexing).
            absorptionTerms (bool, optional): If true will generate the Boltzmann absorption terms, otherwise generates the emission terms. Defaults to False.
            implicitGroups (Optional[List[int]], optional): Implicit groups to put the generated terms into. Defaults to None - using [1].
        """
        assert all(
            isinstance(
                mbData.transitions[ind - 1],
                (FixedECSTransition, CollExIonJanevTransition),
            )
            for ind in includedTransitionIndices
        ) or all(
            isinstance(
                mbData.transitions[ind - 1],
                (DetailedBalanceTransition, CollDeexRecombJanevTransition),
            )
            for ind in includedTransitionIndices
        ), "CRMBoltzTermGenerator can only be called with transition indices all corresponding to FixedECSTransitions or DetailedBalanceTransitions"

        self.__detailedBalanceTerms__ = all(
            isinstance(
                mbData.transitions[ind - 1],
                (DetailedBalanceTransition, CollDeexRecombJanevTransition),
            )
            for ind in includedTransitionIndices
        )
        self.__distribution__ = distribution
        self.__evolvedHarmonic__ = evolvedHarmonic
        self.__associatedVarIndex__ = associatedVarIndex
        self.__absorptionTerms__ = absorptionTerms
        self.__includedTransitionIndices__ = includedTransitionIndices
        self.__fixedEnergyIndices__ = [
            mbData.transitions[ind - 1].fixedEnergyIndex + 1
            for ind in includedTransitionIndices
        ]
        super().__init__(name, implicitGroups if implicitGroups is not None else [1])

    def dict(self) -> Dict:
        tg = super().dict()

        tg.update(
            {
                "type": "CRMFixedBoltzmannCollInt",
                "evolvedHarmonic": self.__evolvedHarmonic__,
                "distributionVarName": self.__distribution__.name,
                "includedTransitionIndices": self.__includedTransitionIndices__,
                "fixedEnergyIndices": self.__fixedEnergyIndices__,
                "absorptionTerm": self.__absorptionTerms__,
                "detailedBalanceTerm": self.__detailedBalanceTerms__,
                "associatedVarIndex": self.__associatedVarIndex__,
            }
        )

        return tg

    @property
    def evolvedVars(self):
        return [self.__distribution__.name]

    def onlyEvolving(self, *args: Variable) -> Self:
        return self

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        doc.append(
            tex.NoEscape(
                self.name.replace("_", r"\_")
                + ": \\newline CRM Boltzmann collision operator "
                + ("absorption" if self.__absorptionTerms__ else "emission")
                + " term generator"
            )
        )
        doc.append(
            tex.NoEscape(
                "\\newline Evolved distribution: "
                + self.__distribution__.latex(latexRemap)
            )
        )
        doc.append(
            tex.NoEscape(
                "\\newline Transitions included: "
                + (
                    ", ".join(str(i) for i in self.__includedTransitionIndices__)
                    + " \\newline"
                    if self.__includedTransitionIndices__ is not None
                    else "all \\newline"
                )
            )
        )


class CRMSecElTermGenerator(TermGenerator):
    """Generator for secondary electrons for the electron kinetic equation due to CRM contributions"""

    def __init__(
        self,
        name: str,
        distribution: Variable,
        includedTransitionIndices: Optional[List[int]] = None,
        implicitGroups: Optional[List[int]] = None,
    ) -> None:
        """Generator for secondary electrons for the electron kinetic equation due to CRM contributions. The electrons are all put into the lowest energy cell.

        Args:
            name (str): Name of the generator
            distribution (Variable): Electron distribution function variable
            includedTransitionIndices (Optional[List[int]], optional): List of included transition indices. Defaults to None - including all transitions.
            implicitGroups (Optional[List[int]], optional): Implicit groups to put generated terms into. Defaults to None - using [1].
        """

        self.__distribution__ = distribution
        self.__includedTransitionIndices__ = includedTransitionIndices
        super().__init__(name, implicitGroups if implicitGroups is not None else [1])

    def dict(self) -> Dict:
        tg = super().dict()

        tg.update(
            {
                "type": "CRMSecondaryElectronTerms",
                "distributionVarName": self.__distribution__.name,
                "includedTransitionIndices": (
                    self.__includedTransitionIndices__
                    if self.__includedTransitionIndices__ is not None
                    else []
                ),
            }
        )

        return tg

    @property
    def evolvedVars(self):
        return [self.__distribution__.name]

    def onlyEvolving(self, *args: Variable) -> Self:
        return self

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        doc.append(
            tex.NoEscape(
                self.name.replace("_", r"\_")
                + ": \\newline CRM secondary electron source term generator"
            )
        )
        doc.append(
            tex.NoEscape(
                "\\newline Evolved distribution: "
                + self.__distribution__.latex(latexRemap)
            )
        )
        doc.append(
            tex.NoEscape(
                "\\newline Transitions included: "
                + (
                    ", ".join(str(i) for i in self.__includedTransitionIndices__)
                    + " \\newline"
                    if self.__includedTransitionIndices__ is not None
                    else "all \\newline"
                )
            )
        )
