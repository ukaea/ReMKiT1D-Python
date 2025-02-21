from typing import Union, List, Dict, cast, Tuple, Type, Callable, Optional
from typing_extensions import Self
import numpy as np
from abc import ABC, abstractmethod
from . import calculation_tree_support as ct
from copy import copy, deepcopy
from .grid import Grid, Profile
import pylatex as tex  # type: ignore
from math import isclose
from .tex_parsing import numToScientificTex
from scipy import special  # type: ignore
from scipy.interpolate import RegularGridInterpolator  # type: ignore
import warnings


class DerivBase(ABC):
    """Abstract base for derivation-like objects"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def rename(self, name: str) -> Self:
        pass


class DerivationArgument(ABC):
    """Abstract base class for potential arguments of derivation - used primarily for Variables"""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def subtype(self) -> str:
        pass

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def apply(cls, deriv: DerivBase, *args: Self):
        assert len(args), "apply() args must be of non-zero length"
        pass

    @property
    @abstractmethod
    def isOnDualGrid(self):
        pass


class Species:
    """Contains species data"""

    def __init__(
        self,
        name: str,
        speciesID: int,
        atomicA: float = 1.0,
        charge: float = 0.0,
        associatedVars: Optional[List[DerivationArgument]] = None,
        latexName: Optional[str] = None,
    ) -> None:
        """A species object used by ReMKiT1D

        Args:
            name (str): Name of the species
            speciesID (int): Unique integer ID - note that 0 is reserved for an electron species
            atomicA (float, optional): Atomic mass in amu. Defaults to 1.0.
            charge (float, optional): Charge in elementary units. Defaults to 0.0.
            associatedVars (List[DerivationArgument], optional): List of associated variables. Defaults to None.
            latexName (Optional[str], optional): Optional non-default LaTeX name for the species, uses \\text{name} if None. Defaults to None.
        """
        assert atomicA > 0, "Species mass must be positive"

        self.__name__ = name
        self.__speciesID__ = speciesID
        self.__atomicA__ = atomicA
        self.__charge__ = charge
        self.__associatedVars__ = [] if associatedVars is None else associatedVars
        self.__latexName__ = latexName
        self.__varsByType__: Dict[str, DerivationArgument] = {}

    @property
    def name(self):
        return self.__name__

    def rename(self, name: str):
        newSp = deepcopy(self)
        newSp.__name__ = name
        return newSp

    @property
    def speciesID(self):
        return self.__speciesID__

    @property
    def atomicA(self):
        return self.__atomicA__

    @property
    def charge(self):
        return self.__charge__

    @property
    def associatedVarNames(self):
        return [var.name for var in self.__associatedVars__]

    def associateVar(self, *args: DerivationArgument, associateSubtype=False):
        """Associate variable with this species

        Args:
            associateSubtype (bool, optional): If true will try to associate variables by subtype as well, for example, if the variables subtype is "density", it can be accessed by indexing the species object using ["density"]. Only the first variable of a given subtype will be associated and warning will be raised for duplicates. Defaults to False.
        """
        for arg in args:
            if arg.name not in self.associatedVarNames:
                self.__associatedVars__.append(arg)
                if associateSubtype:
                    if arg.subtype not in self.__varsByType__:
                        self.__varsByType__[arg.subtype] = arg
                    else:
                        warnings.warn(
                            "Species "
                            + self.name
                            + " already has a variable with subtype "
                            + arg.subtype
                            + " associated. Variable "
                            + arg.name
                            + " will not be set as the "
                            + arg.subtype
                            + " variable for this species. You can still manually override this by overriding the ["
                            + arg.subtype
                            + "] element of this species."
                        )

    def __getitem__(self, key: str):
        return self.__varsByType__[key]

    def __setitem__(self, key: str, var: DerivationArgument):
        assert key == var.subtype, (
            "key "
            + key
            + " and variable subtype "
            + var.subtype
            + " must match for the variable to be assigned as the relevant subtype variable for the species"
        )
        assert (
            var.name in self.associatedVarNames
        ), "If setting subtype variable manually on species it must already be associated t"
        self.__varsByType__[key] = var

    def dict(self) -> dict:
        """Return Species objects as dictionary entry for JSON output

        Returns:
            dict: ReMKiT1D-ready dictionary form of species data
        """

        speciesData = {
            "ID": self.speciesID,
            "atomicMass": self.atomicA,
            "charge": self.charge,
            "associatedVars": self.associatedVarNames,
        }

        return speciesData

    def latex(self) -> str:
        """LaTeX represenation of the species

        Returns:
            str: LaTeX-compatible string representing the species
        """
        return (
            self.__latexName__
            if self.__latexName__ is not None
            else "\\text{" + self.name.replace("_", r"\_") + "}"
        )


class SpeciesContainer:
    """Container of multiple species"""

    def __init__(self, *args: Species):
        self.__species__ = list(args)

    @property
    def speciesNames(self):
        return [species.name for species in self.__species__]

    def dict(self) -> dict:

        speciesDict = {"names": self.speciesNames}

        for species in self.__species__:
            speciesDict.update({species.name: species.dict()})

        return speciesDict

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        if len(self.__species__) > 0:
            latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
            with doc.create(tex.Section("Species")):
                with doc.create(tex.Itemize()) as itemize:
                    for species in self.__species__:
                        associatedVarNames = [
                            (
                                "$" + latexRemap[varName] + "$"
                                if varName in latexRemap
                                else "$\\text{" + varName.replace("_", r"\_") + "}$"
                            )
                            for varName in species.associatedVarNames
                        ]
                        itemize.add_item(
                            tex.NoEscape(
                                "$"
                                + species.latex()
                                + "$"
                                + f": ID: {species.speciesID}; A: {species.atomicA:.4e}; Z: {species.charge:.2f}; Associated vars: "
                                + ",".join(associatedVarNames)
                            )
                        )

    def __getitem__(self, key: str):
        if key not in self.speciesNames:
            raise KeyError()
        return self.__species__[self.speciesNames.index(key)]

    def __setitem__(self, key: str, species: Species):
        if key not in self.speciesNames:
            self.__species__.append(species.rename(key))
        else:
            self.__species__[self.speciesNames.index(key)] = species.rename(key)

    def __delitem__(self, key: str):
        if key not in self.speciesNames:
            raise KeyError()
        del self.__species__[self.speciesNames.index(key)]

    def add(self, *args: Species):
        for arg in args:
            assert (
                arg.name not in self.speciesNames
            ), "Attempted to add duplicate species to SpeciesContainer - use key access if you wish to overwrite an existing species"

        self.__species__ += list(args)


class DerivationContainer(ABC):
    """Abstract base class for derivation containers"""

    @abstractmethod
    def register(self, deriv: DerivBase, ignoreDuplicates=False) -> None:
        pass


class Derivation(DerivBase):
    """Abstract Derivation class"""

    def __init__(
        self,
        name: str,
        numArgs: int,
        latexTemplate: Optional[str] = None,
        latexName: Optional[str] = None,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        super().__init__()
        self.__name__ = name
        assert numArgs >= 0, "Negative number of arguments in Derivation"
        self.__numArgs__ = numArgs
        self.__latexName__ = (
            latexName
            if latexName is not None
            else "\\text{" + name.replace("_", r"\_") + "}"
        )
        if latexTemplate is not None:
            for i in range(numArgs):
                assert f"${i}" in latexTemplate, f"${i} not in latexTemplate"
        self.__latexTemplate__ = latexTemplate

        if container is not None:
            container.register(self)

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name: str):
        self.__name__ = name

    def rename(self, name):
        self.__name__ = name
        return self

    @property
    def latexName(self):
        return self.__latexName__

    @property
    def numArgs(self):
        return self.__numArgs__

    @property
    def latexTemplate(self):
        return self.__latexTemplate__

    @abstractmethod
    def dict(self) -> dict:
        pass

    def latex(self, *args: str) -> str:
        if self.__latexTemplate__ is None:
            raise NotImplementedError(
                "latexTemplate not provided in Derivation and latex() function not overwritten"
            )
        expression = copy(self.__latexTemplate__)
        for i, name in enumerate(args):
            expression = expression.replace(f"${i}", name)

        return expression

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Derivation evaluation not implemented")

    def __call__(self, *args):
        if len(args):
            if isinstance(args[0], str):
                return self.latex(*args)
            if isinstance(args[0], np.ndarray):
                return self.evaluate(*args)
            if issubclass(type(args[0]), DerivationArgument):
                return type(args[0]).apply(self, *args)
            raise TypeError("Unsupported argument in Derivation __call__")

        else:
            return self.dict()

    def registerComponents(self, container: DerivationContainer):
        pass

    def fillArgs(self, *args: str) -> List[str]:
        return list(args)

    @property
    def resultProperties(self) -> Dict[str, object]:
        return {}

    @property
    def enclosedArgs(self) -> int:
        return 0

    @property
    def totNumArgs(self) -> int:
        return self.numArgs + self.enclosedArgs


class GenericDerivation(Derivation):
    """Generic derivation for wrapping ReMKiT1D dictionary representations of derivations"""

    def __init__(
        self,
        name: str,
        numArgs: int,
        properties: Dict[str, object],
        latexTemplate: Optional[str] = None,
        resultProperties: Dict[str, object] = {},
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Generic derivation wrapper taking in the ReMKiT1D dictionary representation

        Args:
            name (str): Name of the derivation
            numArgs (int): Expected number of arguments of the derivation
            properties (Dict[str,object]): Dictionary representation of the ReMKiT1D derivation
            latexTemplate (Optional[str], optional): Optional latex template. If present, needs to contain tokens of the form $0,$1, etc. in order and for each expected argument. These are then replaced by latex representations of the individual arguments. Defaults to None.
            resultProperties (Dict[str, object], optional): List of result properties in the form of kwargs for the Variable constructor. Defaults to {}.
            container (Optional[DerivationContainer], optional): Optional derivation container to register the derivation on construction. Defaults to None.
        """
        super().__init__(name, numArgs, latexTemplate, container=container)
        self.__properties__ = properties
        self.__resultProperties__ = resultProperties

    def dict(self) -> dict:
        return self.__properties__

    def latex(self, *args) -> str:
        assert (
            len(args) == self.numArgs
        ), "latex() called with args not conforming to the number of expected arguments"
        if self.latexTemplate is None:
            expression = self.latexName + "\\left("
            for name in args:
                expression += name + ","
            return expression[:-1] + "\\right)"
        else:
            return super().latex(*args)

    @property
    def resultProperties(self):
        return self.__resultProperties__


class NodeDerivation(Derivation):
    """Derivation based on calculation tree"""

    def __init__(
        self,
        name: str,
        node: ct.Node,
        latexTemplate: Optional[str] = None,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Derivation based on a calculation tree

        Args:
            name (str): Name of the derivation
            node (ct.Node): The node used for the derived variable. The number of arguments and the arguments themselves are inferred from the node.
            latexTemplate (Optional[str], optional): Optional latex template. If present, needs to contain tokens of the form $0,$1, etc. in order and for each expected argument. These are then replaced by latex representations of the individual arguments. Defaults to None.
            resultProperties (Dict[str, object], optional): List of result properties in the form of kwargs for the Variable constructor. Defaults to {}.
            container (Optional[DerivationContainer], optional): Optional derivation container to register the derivation on construction. Defaults to None.
        """
        super().__init__(name, 0, latexTemplate, container=container)
        self.__node__ = node

    def dict(self) -> dict:
        return ct.treeDerivation(self.__node__)

    def evaluate(self, *args) -> np.ndarray:
        assert (
            len(args) == self.totNumArgs
        ), "evaluate() called with args not conforming to the number of expected arguments"

        return self.__node__.evaluate(dict(zip(ct.getLeafVars(self.__node__), args)))

    def latex(self, *args: str) -> str:
        assert (
            len(args) == self.totNumArgs
        ), "latex() called with args not conforming to the number of expected arguments"
        if self.latexTemplate is None:
            remap = dict(zip(ct.getLeafVars(self.__node__), args))
            return self.__node__.latex(remap)
        else:
            return super().latex(*args)

    @property
    def node(self):
        return self.__node__

    @property
    def enclosedArgs(self):
        return len(ct.getLeafVars(self.__node__))

    def fillArgs(self, *args: str) -> List[str]:
        return ct.getLeafVars(self.__node__)


class SimpleDerivation(Derivation):
    """Simple derivation object which calculates its value as multConst * prod(vars**powers)"""

    def __init__(
        self,
        name: str,
        multConst: float,
        varPowers: List[float],
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Simple derivation object which calculates its value as multConst * prod(vars**powers)

        Args:
            name (str): Name of the derivation
            multConst (float): Multiplicative constant
            varPowers (List[float]): Powers to raise passed variables to.
            container (Optional[DerivationContainer], optional): Optional derivation container to register the derivation on construction. Defaults to None.
        """
        super().__init__(name, len(varPowers), container=container)
        self.__multConst__ = multConst
        self.__varPowers__ = varPowers

    def dict(self) -> dict:
        return {
            "type": "simpleDerivation",
            "multConst": self.__multConst__,
            "varPowers": self.__varPowers__,
        }

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        assert len(args) == len(
            self.__varPowers__
        ), "SimpleDerivation args must conform to the number of variable powers"
        val = self.__multConst__ * args[0] ** self.__varPowers__[0]
        for i in range(1, len(args)):
            val *= args[i] ** self.__varPowers__[i]
        return val

    def latex(self, *args: str) -> str:
        expression = numToScientificTex(self.__multConst__, removeUnity=True)
        numerator = ""
        denominator = ""

        for i, arg in enumerate(args):
            if self.__varPowers__[i] > 0:
                if isclose(self.__varPowers__[i], 1.0, rel_tol=1e-4):
                    numerator += " " + arg
                else:
                    power = (
                        f"{round(self.__varPowers__[i])}"
                        if isclose(
                            self.__varPowers__[i],
                            round(self.__varPowers__[i]),
                            rel_tol=1e-2,
                        )
                        else f"{self.__varPowers__[i]:.2f}"
                    )
                    numerator += " " + arg + "^{" + power + "}"
            else:
                if isclose(self.__varPowers__[i], -1.0, rel_tol=1e-4):
                    denominator += " " + arg
                else:
                    power = (
                        f"{round(-self.__varPowers__[i])}"
                        if isclose(
                            self.__varPowers__[i],
                            round(self.__varPowers__[i]),
                            rel_tol=1e-2,
                        )
                        else f"{-self.__varPowers__[i]:.2f}"
                    )
                    denominator += " " + arg + "^{" + power + "}"
        if len(denominator):
            return expression + "\\frac{" + numerator + "}{" + denominator + "}"
        return expression + numerator


class BuiltInDerivation(Derivation):
    """Abstract class for built-in derivations"""

    def __init__(
        self,
        name: str,
        numArgs: int,
        latexTemplate: Optional[str] = None,
        latexName: Optional[str] = None,
    ) -> None:
        super().__init__(name, numArgs, latexTemplate, latexName)

    def dict(self) -> dict:
        """No-op since these are built-in derivations"""
        return {}


class InterpolationDerivation(BuiltInDerivation):
    """Built-in interpolation derivation"""

    def __init__(self, grid: Grid, ontoDual=True, onDistribution=False):
        """Built-in interpolation derivation

        Args:
            grid (Grid): Grid used for the interpolation (variables should live on this grid)
            ontoDual (bool, optional): True if the interpolation is from grid to dual, otherwise false. Defaults to True.
            onDistribution (bool, optional): True if the interpolated variable is a distribution. Defaults to False.

        Raises:
            ValueError: If ontoDual is false for a distribution - distribution interpolation goes only from grid to dual due to the harmonic structure
        """
        if onDistribution and not ontoDual:
            raise ValueError(
                "InterpolationDerivation on distributions must be ontoDual"
            )
        name = "gridToDual" if ontoDual else "dualToGrid"
        if onDistribution:
            name = "distributionInterp"
        super().__init__(name, 1)
        self.__grid__ = grid

    def evaluate(self, *args: np.ndarray):
        assert (
            len(args) == 1
        ), "InterpolationDerivation evaluate() must have exactly one argument"
        if self.name == "distributionInterp":
            return self.__grid__.distFullInterp(args[0])
        if self.name == "gridToDual":
            return self.__grid__.gridToDual(args[0])
        return self.__grid__.dualToGrid(args[0])

    def latex(self, *args: str):
        assert (
            len(args) == 1
        ), "InterpolationDerivation latex() must have exactly one argument"

        if self.name == "gridToDual":
            return "\\mathcal{I}\\left(" + args[0] + "\\rightarrow \\text{dual}\\right)"
        if self.name == "dualToGrid":
            return "\\mathcal{I}\\left(" + args[0] + "\\rightarrow \\text{grid}\\right)"

        return (
            "\\mathcal{I}\\left(" + args[0] + "\\rightarrow \\text{grid/dual}\\right)"
        )


class Textbook(DerivationContainer):
    """Standard derivation container offering built-in derivations"""

    def __init__(
        self,
        grid: Grid,
        tempDerivSpeciesIDs: Optional[List[int]] = None,
        ePolyCoeff=1.0,
        ionPolyCoeff=1.0,
        electronSheathGammaIonSpeciesID=-1,
        removeLogLeiDiscontinuity=False,
    ) -> None:
        """The default container of ReMKiT1D derivations, including built-in derivations.

        Args:
            grid (Grid): Simulation grid
            tempDerivSpeciesIDs (Optional[List[int]], optional): Species ID's for which a built-in temperature derivation should be added. Defaults to None.
            ePolyCoeff (float, optional): Electron polytropic coefficient for built-in sonic speed calculation. Defaults to 1.0.
            ionPolyCoeff (float, optional): Ion polytropic coefficient for built-in sonic speed calculation. Defaults to 1.0.
            electronSheathGammaIonSpeciesID (int, optional): The species ID used to calculate the mass ratio that goes into the electron sheath heat flux transmission coefficient. Defaults to -1.
            removeLogLeiDiscontinuity (bool, optional): If true will remove the discontinuity in the electron-ion Coulomb logarithm present in the NRL Plasma Formulary. Defaults to False.

        All build-in derivations assume default normalisation (arguments counted using $0,$1, etc. where they are not obvious)
        Built-in derivations:

            "flowSpeedFromFlux" = $0/$1
            "sonicSpeed" for each ion species (negative IDs) = sqrt((poly_e * $0 + poly_i * $1)/m_i) where i is the index of the ion species (the derivation names will end in the name of the species, e.g. sonicSpeedD+)
            "tempFromEnergy" for each species in tempDerivSpeciesIDs = 2/3 * $0/$1 - mass * $2**2/(3*$1**2) (the derivation names will end in the name of the species, e.g. sonicSpeedD+)
            "leftElectronGamma" = sheath heat transmission coeffiecient for electrons at the left boundary using T_e and T_i
            "rightElectronGamma" = sheath heat transmission coefficient for electrons at the right boundary using T_e and T_i
            "densityMoment" = zeroth moment of f_0
            "energyMoment" = second order of f_0
            "cclDragCoeff" = Chang-Cooper-Langdon drag coefficient - assumes passed variable is the single harmonic f_0
            "cclDiffusionCoeff" = Chang-Cooper-Langdon diffusion coefficient - assumes passed variables are the single harmonic f_0 and cclWeight
            "cclWeight" = Chang-Cooper-Langdon interpolation weight - assumes passed variables are the single harmonic cclDragCoeff and cclDiffusionCoeff

            if l=1 is resolved:
            "fluxMoment" = first moment of f_1/3
            "heatFluxMoment" = third moment of f_1/3
            if l=2 is resolved:
            "viscosityTensorxxMoment" = second moment of 2*f_2/15

            Interpolation for staggered grids:
            "gridToDual" = interpolates one variable from regular to staggered(dual) grid
            "dualToGrid" = interpolates one variable from staggered(dual) to regular grid
            "distributionInterp" = interpolates one distribution function (even harmonics to dual, odd to regular grid)

            Other useful derivations:
            "gradDeriv" = calculates gradient of variable on regular grid, interpolating on cell faces and extrapolating at boundaries
            "logLei" = calculates electron-ion Coulomb log taking in electron temperature and density (derivations added for each ion species - e.g. logLeiD+)
            "logLee" = calculates electron self collision Coulomb log taking in electron temperature and density
            "logLii" = calculates ion-ion collision frequency for each ion collision combination taking in the two species temperatures and densities (used as"logLiis_S" where s and S are the first and second ion species name)
            "maxwellianDistribution" = calculates distribution function with Maxwellian l=0 harmonic and all other harmonics 0. Takes in temperature and density as first and second argument
        """
        super().__init__()

        self.__grid__ = grid
        self.__tempDerivSpeciesIDs__: List[int] = (
            [] if tempDerivSpeciesIDs is None else tempDerivSpeciesIDs
        )
        self.__ePolyCoeff__ = ePolyCoeff
        self.__ionPolyCoeff__ = ionPolyCoeff
        self.__electronSheathGammaIonSpeciesID__ = electronSheathGammaIonSpeciesID
        self.__removeLogLeiDiscontinuity__ = removeLogLeiDiscontinuity

        self.__derivations__: List[Derivation] = []

        # Built-in derivations
        self.__derivations__.append(
            GenericDerivation(
                "flowSpeedFromFlux", 2, {}, latexTemplate="\\frac{$0}{$1}"
            )
        )
        self.__derivations__.append(
            GenericDerivation(
                "leftElectronGamma",
                2,
                {},
                latexTemplate="\\gamma_{e,\\text{left}}\\left($0,$1\\right)",
                resultProperties={
                    "isScalar": True,
                    "isDistribution": False,
                    "isSingleHarmonic": False,
                },
            )
        )
        self.__derivations__.append(
            GenericDerivation(
                "rightElectronGamma",
                2,
                {},
                latexTemplate="\\gamma_{e,\\text{right}}\\left($0,$1\\right)",
                resultProperties={
                    "isScalar": True,
                    "isDistribution": False,
                    "isSingleHarmonic": False,
                },
            )
        )
        self.__derivations__.append(
            GenericDerivation(
                "densityMoment",
                1,
                {},
                latexTemplate="\\langle $0 \\rangle _{l=0,n=0}",
                resultProperties={
                    "isScalar": False,
                    "isDistribution": False,
                    "isSingleHarmonic": False,
                },
            )
        )
        self.__derivations__.append(
            GenericDerivation(
                "energyMoment",
                1,
                {},
                latexTemplate="\\langle $0 \\rangle _{l=0,n=2}",
                resultProperties={
                    "isScalar": False,
                    "isDistribution": False,
                    "isSingleHarmonic": False,
                },
            )
        )

        self.__derivations__.append(
            GenericDerivation(
                "cclDragCoeff",
                1,
                {},
                latexTemplate="C_{CCL}($0)",
                resultProperties={
                    "isScalar": False,
                    "isDistribution": False,
                    "isSingleHarmonic": True,
                },
            )
        )
        self.__derivations__.append(
            GenericDerivation(
                "cclDiffusionCoeff",
                2,
                {},
                latexTemplate="D_{CCL}($0,$1)",
                resultProperties={
                    "isScalar": False,
                    "isDistribution": False,
                    "isSingleHarmonic": True,
                },
            )
        )
        self.__derivations__.append(
            GenericDerivation(
                "cclWeight",
                2,
                {},
                latexTemplate="\\delta_{CCL}($0,$1)",
                resultProperties={
                    "isScalar": False,
                    "isDistribution": False,
                    "isSingleHarmonic": True,
                },
            )
        )

        if grid.lMax > 0:
            self.__derivations__.append(
                GenericDerivation(
                    "fluxMoment",
                    1,
                    {},
                    latexTemplate="\\frac{1}{3}\\langle $0 \\rangle _{l=1,n=1}",
                    resultProperties={
                        "isScalar": False,
                        "isDistribution": False,
                        "isSingleHarmonic": False,
                    },
                )
            )
            self.__derivations__.append(
                GenericDerivation(
                    "heatFluxMoment",
                    1,
                    {},
                    latexTemplate="\\frac{1}{3}\\langle $0 \\rangle _{l=1,n=3}",
                    resultProperties={
                        "isScalar": False,
                        "isDistribution": False,
                        "isSingleHarmonic": False,
                    },
                )
            )
        if grid.lMax > 1:
            self.__derivations__.append(
                GenericDerivation(
                    "viscosityTensorxxMoment",
                    1,
                    {},
                    latexTemplate="\\frac{2}{15}\\langle $0 \\rangle _{l=2,n=2}",
                    resultProperties={
                        "isScalar": False,
                        "isDistribution": False,
                        "isSingleHarmonic": False,
                    },
                )
            )

        self.__derivations__.append(InterpolationDerivation(grid))
        self.__derivations__.append(InterpolationDerivation(grid, False))
        self.__derivations__.append(InterpolationDerivation(grid, onDistribution=True))

        self.__derivations__.append(
            GenericDerivation(
                "gradDeriv", 1, {}, latexTemplate="\\nabla\\left($0\\right)"
            )
        )
        self.__derivations__.append(
            GenericDerivation(
                "logLee", 2, {}, "\\text{log}\\Lambda_{e,e}\\left($0,$1\\right)"
            )
        )

        self.__derivations__.append(
            GenericDerivation(
                "maxwellianDistribution",
                2,
                {},
                "f_M\\left($0,$1\\right)",
                resultProperties={
                    "isScalar": False,
                    "isDistribution": True,
                    "isSingleHarmonic": False,
                },
            )
        )

        self.__builtinDerivNum__ = len(self.__derivations__)

    @property
    def registeredDerivs(self):
        return [deriv.name for deriv in self.__derivations__]

    @property
    def tempDerivSpeciesIDs(self):
        return self.__tempDerivSpeciesIDs__

    @tempDerivSpeciesIDs.setter
    def tempDerivSpeciesIDs(self, ids: List[int]):
        self.__tempDerivSpeciesIDs__ = ids

    def __getitem__(self, name: str):
        if name not in self.registeredDerivs:
            if name.startswith("sonicSpeed"):
                return GenericDerivation(
                    name,
                    2,
                    {},
                    latexTemplate="c_{s," + name[10:] + "}\\left($0,$1\\right)",
                )
            if name.startswith("tempFromEnergy"):
                return GenericDerivation(
                    name,
                    3,
                    {},
                    latexTemplate="T_{" + name[14:] + "}\\left($0,$1,$2\\right)",
                )
            if name.startswith("logLei"):
                return GenericDerivation(
                    name,
                    2,
                    {},
                    latexTemplate="\\text{log}\\Lambda_{e,"
                    + name[6:]
                    + "}\\left($0,$1\\right)",
                )
            if name.startswith("logLii"):
                names = name.split("_")
                return GenericDerivation(
                    name,
                    4,
                    {},
                    latexTemplate="\\text{log}\\Lambda_{"
                    + names[0][6:]
                    + ","
                    + names[1]
                    + "}\\left($0,$1,$2,$3\\right)",
                )

            raise KeyError()
        return self.__derivations__[self.registeredDerivs.index(name)]

    def register(self, deriv: DerivBase, ignoreDuplicates=False) -> None:
        if not ignoreDuplicates:
            assert deriv.name not in self.registeredDerivs, (
                "Derivation " + deriv.name + " already registered in textbook"
            )
            for prefix in ["sonicSpeed", "tempFromEnergy", "logLei", "logLii"]:
                assert not deriv.name.startswith(prefix), (
                    "Derivation " + deriv.name + " starts with reserved prefix"
                )
        try:
            _ = self[deriv.name]
        except KeyError:
            cast(Derivation, deriv).registerComponents(self)
            try:
                _ = self[deriv.name]
            except KeyError:
                self.__derivations__.append(cast(Derivation, deriv))

    def dict(self):

        textbookDict = {
            "standardTextbook": {
                "temperatureDerivSpeciesIDs": self.__tempDerivSpeciesIDs__,
                "electronPolytropicCoeff": self.__ePolyCoeff__,
                "ionPolytropicCoeff": self.__ionPolyCoeff__,
                "electronSheathGammaIonSpeciesID": self.__electronSheathGammaIonSpeciesID__,
                "removeLogLeiDiscontinuity": self.__removeLogLeiDiscontinuity__,
            }
        }

        textbookDict["customDerivations"] = {
            "tags": self.registeredDerivs[self.__builtinDerivNum__ :]
        }
        for i in range(self.__builtinDerivNum__, len(self.__derivations__)):
            textbookDict["customDerivations"][self.__derivations__[i].name] = (
                self.__derivations__[i].dict()
            )

        return textbookDict

    def addSpeciesForTempDeriv(self, species: Species):
        ids = set(self.__tempDerivSpeciesIDs__)
        ids.add(species.speciesID)
        self.__tempDerivSpeciesIDs__ = list(ids)

    @property
    def ePolyCoeff(self):
        return self.__ePolyCoeff__

    @ePolyCoeff.setter
    def ePolyCoeff(self, val: float):
        self.__ePolyCoeff__ = val

    @property
    def ionPolyCoeff(self):
        return self.__ionPolyCoeff__

    @ionPolyCoeff.setter
    def ionPolyCoeff(self, val: float):
        self.__ionPolyCoeff__ = val

    def setSheathGammaSpecies(self, species: Species):
        self.__electronSheathGammaIonSpeciesID__ = species.speciesID

    @property
    def removeLogLeiDiscontinuity(self):
        return self.__removeLogLeiDiscontinuity__

    @removeLogLeiDiscontinuity.setter
    def removeLogLeiDiscontinuity(self, remove: bool):
        self.__removeLogLeiDiscontinuity__ = remove

    def addLatexToDoc(self, doc: tex.Document):
        if len(self.registeredDerivs) > self.__builtinDerivNum__:
            with doc.create(tex.Section("Custom derivations")):
                with doc.create(tex.Itemize()) as itemize:
                    for deriv in self.__derivations__[self.__builtinDerivNum__ :]:
                        argTuple = tuple(
                            "x_" + str(i)
                            for i in range(deriv.numArgs + deriv.enclosedArgs)
                        )
                        itemize.add_item(
                            tex.NoEscape(
                                deriv.name.replace("_", r"\_")
                                + f": ${deriv.latex(*argTuple)}$"
                            )
                        )


class MultiplicativeDerivation(Derivation):
    """Multiplicative composite derivation"""

    def __init__(
        self,
        name: str,
        innerDerivation: Derivation,
        outerDerivation: Optional[Derivation] = None,
        innerFunc: Optional[str] = None,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Multiplicative composite derivation. Takes up to two derivations and composes them as follows: fun(d1) * d2, where d2 is optional, and fun is an optional Fortran intrinsic function. Assumes that arguments are passed first for the inner then for the outer derivation.

        Args:
            name (str): Name of the derivation
            innerDerivation (Derivation): Inner derivation (d1 in  the text above)
            outerDerivation (Optional[Derivation], optional): Optional outer derivation (d2 in the text above). Defaults to None.
            innerFunc (Optional[str], optional): String representation of Fortran function to be applied to inner derivation. Defaults to None.
            container (Optional[DerivationContainer], optional): Optional container for registering this derivation at construction. Defaults to None.

        Allowed innerFunc names: exp,log,sin,cos,abs,tan,atan,asin,acos,sign,erf,erfc
        """
        numArgs = innerDerivation.numArgs
        if outerDerivation is not None:
            numArgs += outerDerivation.numArgs
        super().__init__(name, numArgs, container=container)

        self.__innerDeriv__ = innerDerivation
        self.__outerDeriv__: Optional[Derivation] = outerDerivation

        funMap = {
            "exp": np.exp,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
            "abs": np.abs,
            "tan": np.tan,
            "atan": np.arctan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "sign": np.sign,
            "erf": special.erf,
            "erfc": special.erfc,
        }

        self.__innerFunc__: Optional[str] = innerFunc
        self.__fun__: Union[Callable, None] = None
        if self.__innerFunc__ is not None:
            assert (
                self.__innerFunc__ in funMap
            ), "Unsupported function passed to MultiplicativeDerivation"
            self.__fun__ = funMap[self.__innerFunc__]

    @property
    def innerDeriv(self):
        return self.__innerDeriv__

    @property
    def outerDeriv(self):
        return self.__outerDeriv__

    @property
    def innerFunc(self):
        return self.__innerFunc__

    @property
    def enclosedArgs(self):
        return (
            self.innerDeriv.enclosedArgs
            if self.outerDeriv is None
            else self.innerDeriv.enclosedArgs + self.outerDeriv.enclosedArgs
        )

    def dict(self) -> Dict:

        deriv = {
            "type": "multiplicativeDerivation",
            "innerDerivation": self.innerDeriv.name,
            "innerDerivIndices": [i + 1 for i in range(self.innerDeriv.totNumArgs)],
            "innerDerivPower": 1.0,
            "outerDerivation": (
                "none" if self.outerDeriv is None else self.outerDeriv.name
            ),
            "outerDerivIndices": (
                []
                if self.outerDeriv is None
                else [
                    i + 1 + self.innerDeriv.totNumArgs
                    for i in range(
                        self.outerDeriv.numArgs + self.outerDeriv.enclosedArgs
                    )
                ]
            ),
            "outerDerivPower": 1.0,
            "innerDerivFuncName": (
                "none" if self.__innerFunc__ is None else self.__innerFunc__
            ),
        }

        return deriv

    def latex(self, *args: str) -> str:
        assert (
            len(args) == self.numArgs + self.enclosedArgs
        ), "latex() on MultiplicativeDerivation called with wrong number of arguments"
        result = (
            "\\left("
            + self.innerDeriv.latex(*args[: self.innerDeriv.totNumArgs])
            + "\\right)"
        )

        if self.innerFunc is not None:
            result = self.innerFunc + result

        if self.outerDeriv is not None:
            result += (
                "\\left("
                + self.outerDeriv.latex(*args[self.innerDeriv.totNumArgs :])
                + "\\right)"
            )

        return result

    def evaluate(self, *args):
        assert (
            len(args) == self.totNumArgs
        ), "evaluate() on MultiplicativeDerivation called with wrong number of arguments"
        result = self.innerDeriv.evaluate(*args[: self.innerDeriv.totNumArgs])

        if self.innerFunc is not None:
            result = self.__fun__(result)

        if self.outerDeriv is not None:
            result *= self.outerDeriv.evaluate(*args[self.innerDeriv.totNumArgs :])

        return result

    def fillArgs(self, *args):
        inner = self.innerDeriv.fillArgs(*args[: self.innerDeriv.numArgs])
        outer = []
        if self.outerDeriv is not None:
            outer = self.outerDeriv.fillArgs(*args[self.innerDeriv.numArgs :])
        return inner + outer

    def registerComponents(self, container):
        container.register(self.__innerDeriv__, ignoreDuplicates=True)
        if self.outerDeriv is not None:
            container.register(self.__outerDeriv__, ignoreDuplicates=True)


class AdditiveDerivation(Derivation):
    """Additive composite derivation"""

    def __init__(
        self,
        name: str,
        derivs: List[Derivation],
        resultPower: Union[int, float] = 1,
        linCoeffs: Optional[List[float]] = None,
        container: Optional[DerivationContainer] = None,
    ):
        """Additive composite derivation. Takes a linear combination of derivation output and optionally raises to a power. Derivation arguments are assumed to be passed in order.

        Args:
            name (str): Name of the derivation
            derivs (List[Derivation]): List of derivations in the linear combination
            resultPower (Union[int, float], optional): Power to raise the linear combination to. Defaults to 1.
            linCoeffs (Optional[List[float]], optional): Linear combination coefficients. Defaults to None.
            container (Optional[DerivationContainer], optional): Optional container to register the derivation in at construction. Defaults to None.
        """
        numArgs = sum(deriv.numArgs for deriv in derivs)
        super().__init__(name, numArgs, container=container)

        if linCoeffs is not None:
            assert len(derivs) == len(
                linCoeffs
            ), "derivs and linCoeffs in AdditiveDerivation must be of same size"

        self.__resultPower__ = resultPower
        self.__linCoeffs__ = (
            linCoeffs if linCoeffs is not None else [1.0 for _ in derivs]
        )
        self.__derivs__: List[Derivation] = []
        for deriv in derivs:
            if deriv.name in [d.name for d in self.__derivs__]:
                derivCopy = deepcopy(deriv)
                derivCopy.name += "_copy"
                while derivCopy.name in [d.name for d in self.__derivs__]:
                    derivCopy.name += "_copy"
                self.__derivs__.append(derivCopy)
            else:
                self.__derivs__.append(deriv)

    @property
    def derivs(self):
        return self.__derivs__

    @property
    def resultPower(self):
        return self.__resultPower__

    @property
    def linCoeffs(self):
        return self.__linCoeffs__

    @property
    def enclosedArgs(self):
        return sum(deriv.enclosedArgs for deriv in self.derivs)

    def fillArgs(self, *args: str):

        derivArgs: List[str] = []

        offset = 0
        for deriv in self.__derivs__:
            derivArgs += deriv.fillArgs(*args[offset : offset + deriv.numArgs])
            offset += deriv.numArgs

        return derivArgs

    def dict(self) -> Dict:

        derivIndices: List[List[int]] = []
        offset = 1
        for deriv in self.__derivs__:
            derivIndices.append([offset + k for k in range(deriv.totNumArgs)])
            offset += deriv.totNumArgs

        derivTags = [deriv.name for deriv in self.__derivs__]
        derivDict: Dict[str, object] = {
            "type": "additiveDerivation",
            "derivationTags": derivTags,
            "resultPower": self.__resultPower__,
            "linearCoefficients": self.__linCoeffs__,
        }

        for i, tag in enumerate(derivTags):
            derivDict[tag] = {"derivationIndices": derivIndices[i]}

        return derivDict

    def latex(self, *args: str) -> str:
        assert (
            len(args) == self.totNumArgs
        ), "latex() on AdditiveDerivation called with wrong number of arguments"
        result = ""
        offset = 0
        for i, deriv in enumerate(self.__derivs__):
            if self.__linCoeffs__[i] < 0:
                result += ""
            else:
                result += "+ " if i > 0 else " "
            coeffRepr = numToScientificTex(self.__linCoeffs__[i], removeUnity=True)
            result += (
                coeffRepr
                + "\\left("
                + deriv.latex(*args[offset : offset + deriv.totNumArgs])
                + "\\right)"
                if coeffRepr != ""
                else deriv.latex(*args[offset : offset + deriv.totNumArgs])
            )
            offset += deriv.numArgs
        powerRepr = numToScientificTex(self.__resultPower__)
        result = (
            "\\left(" + result + "\\right)^{" + powerRepr + "}"
            if self.__resultPower__ != 1
            else result
        )

        return result

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        assert (
            len(args) == self.totNumArgs
        ), "evaluate() on AdditiveDerivation called with wrong number of arguments"
        result = self.__linCoeffs__[0] * self.derivs[0].evaluate(
            *args[: self.derivs[0].totNumArgs]
        )
        offset = self.derivs[0].totNumArgs
        for i, deriv in enumerate(self.__derivs__[1:]):
            result += self.__linCoeffs__[i + 1] * deriv.evaluate(
                *args[offset : offset + deriv.totNumArgs]
            )
            offset += deriv.totNumArgs

        result = result**self.__resultPower__

        return result

    def registerComponents(self, container):
        for deriv in self.__derivs__:
            container.register(deriv, ignoreDuplicates=True)


class DerivationClosure(Derivation):
    """Derivation closure, allowing for storing a derivation and fixing one or more of its arguments. If all arguments are fixed, the closure can be used in additive and multiplicative arithmetic, producing new closures."""

    def __init__(
        self,
        deriv: Derivation,
        *args: DerivationArgument,
        container: Optional[DerivationContainer] = None,
        **kwargs,
    ):
        """Derivation closure, allowing for storing a derivation and fixing one or more of its arguments. If all arguments are fixed, the closure can be used in additive and multiplicative arithmetic, producing new closures.

        Args:
            deriv (Derivation): Enclosed derivation
            container (Optional[DerivationContainer], optional): Optional container to register the enclosed derivation at construction. Defaults to None.

        kwargs:

            argPositions (Tuple[int]): The positions of the enclosed arguments relative to the required arguments of the enclosed derivation. For example:
            if the enclosed derivation requires 3 arguments and argPositions = (0,2), the enclosed arguments are the first and the last argument, and the remaining arguments is free, i.e. the argument of the resulting closure.
        """
        self.__deriv__ = deriv
        self.__args__ = args
        self.__argPositions__ = kwargs.get("argPositions", tuple(range(len(args))))
        assert len(self.__argPositions__) == len(
            args
        ), "argPositions in DerivationClosure must conform to the number of args"
        # TODO: add bounds check for argPositions
        super().__init__(
            deriv.name,
            deriv.numArgs - len(args),
            latexTemplate=deriv.latexTemplate,
            latexName=deriv.latexName,
        )

        if container is not None:
            container.register(deriv, ignoreDuplicates=True)

    def rename(self, name):
        self.__name__ = name
        self.__deriv__.name = name
        return self

    def dict(self):
        return {}

    @property
    def enclosedArgs(self):
        return len(self.__args__) + self.__deriv__.enclosedArgs

    def fillArgs(self, *args: str):
        argNameList: List[str] = []

        argCounter = 0
        for i in range(self.__deriv__.numArgs):
            if i in self.__argPositions__:
                argNameList.append(self.__args__[self.__argPositions__.index(i)].name)
            else:
                argNameList.append(args[argCounter])
                argCounter += 1

        argNameList = self.__deriv__.fillArgs(*tuple(argNameList))

        return argNameList

    def latex(self, *args):

        return self.__deriv__.latex(*args)

    def evaluate(self, *args: np.ndarray):

        argValList: List[np.ndarray] = []

        argCounter = 0
        for i in range(self.__deriv__.totNumArgs):
            if i in self.__argPositions__:
                argValList.append(self.__args__[self.__argPositions__.index(i)].data)
            else:
                argValList.append(args[argCounter])
                argCounter += 1

        return self.__deriv__.evaluate(*tuple(argValList))

    def __generalAdd__(self, rhs: Self) -> Self:
        addDeriv = AdditiveDerivation(
            self.name + "_" + rhs.name, [self.__deriv__, rhs.__deriv__]
        )
        posMap = tuple(
            list(self.__argPositions__)
            + [self.__deriv__.numArgs + i for i in rhs.__argPositions__]
        )
        return cast(
            Self,
            DerivationClosure(
                addDeriv,
                *tuple(list(self.__args__) + list(rhs.__args__)),
                argPositions=posMap,
            ),
        )

    def __add__(self, rhs: Self) -> Self:
        assert (
            self.numArgs == 0 and rhs.numArgs == 0
        ), "Only complete closures can be added"
        if isinstance(self.__deriv__, AdditiveDerivation):
            if isinstance(rhs.__deriv__, AdditiveDerivation):
                if self.__deriv__.resultPower == 1 and rhs.__deriv__.resultPower == 1:
                    addDeriv = AdditiveDerivation(
                        self.name + "_" + rhs.name,
                        self.__deriv__.derivs + rhs.__deriv__.derivs,
                        resultPower=1,
                        linCoeffs=self.__deriv__.linCoeffs + rhs.__deriv__.linCoeffs,
                    )
                    posMap = tuple(
                        list(self.__argPositions__)
                        + [self.__deriv__.numArgs + i for i in rhs.__argPositions__]
                    )
                    return cast(
                        Self,
                        DerivationClosure(
                            addDeriv,
                            *tuple(list(self.__args__) + list(rhs.__args__)),
                            argPositions=posMap,
                        ),
                    )
                return self.__generalAdd__(rhs)
            if self.__deriv__.resultPower == 1:
                addDeriv = AdditiveDerivation(
                    self.name + "_" + rhs.name,
                    self.__deriv__.derivs + [rhs.__deriv__],
                    resultPower=1,
                    linCoeffs=self.__deriv__.linCoeffs + [1.0],
                )
                posMap = tuple(
                    list(self.__argPositions__)
                    + [self.__deriv__.numArgs + i for i in rhs.__argPositions__]
                )
                return cast(
                    Self,
                    DerivationClosure(
                        addDeriv,
                        *tuple(list(self.__args__) + list(rhs.__args__)),
                        argPositions=posMap,
                    ),
                )
            return self.__generalAdd__(rhs)

        if isinstance(rhs.__deriv__, AdditiveDerivation):
            if rhs.__deriv__.resultPower == 1:
                addDeriv = AdditiveDerivation(
                    self.name + "_" + rhs.name,
                    [self.__deriv__] + rhs.__deriv__.derivs,
                    resultPower=1,
                    linCoeffs=[1.0] + rhs.__deriv__.linCoeffs,
                )
                posMap = tuple(
                    list(self.__argPositions__)
                    + [self.__deriv__.numArgs + i for i in rhs.__argPositions__]
                )
                return cast(
                    Self,
                    DerivationClosure(
                        addDeriv,
                        *tuple(list(self.__args__) + list(rhs.__args__)),
                        argPositions=posMap,
                    ),
                )
        return self.__generalAdd__(rhs)

    def __generalMul__(self, rhs: Self) -> Self:
        mulDeriv = MultiplicativeDerivation(
            self.name + "X" + rhs.name,
            innerDerivation=self.__deriv__,
            outerDerivation=rhs.__deriv__,
        )
        posMap = tuple(
            list(self.__argPositions__)
            + [self.__deriv__.numArgs + i for i in rhs.__argPositions__]
        )
        return cast(
            Self,
            DerivationClosure(
                mulDeriv,
                *tuple(list(self.__args__) + list(rhs.__args__)),
                argPositions=posMap,
            ),
        )

    def __mul__(self, rhs: Self) -> Self:
        assert (
            self.numArgs == 0 and rhs.numArgs == 0
        ), "Only complete closures can be multiplied"
        if isinstance(self.__deriv__, MultiplicativeDerivation):
            if self.__deriv__.outerDeriv is None:
                mulDeriv = MultiplicativeDerivation(
                    self.name + "X" + rhs.name,
                    innerDerivation=self.__deriv__.innerDeriv,
                    innerFunc=self.__deriv__.innerFunc,
                    outerDerivation=rhs.__deriv__,
                )
                posMap = tuple(
                    list(self.__argPositions__)
                    + [self.__deriv__.numArgs + i for i in rhs.__argPositions__]
                )
                return cast(
                    Self,
                    DerivationClosure(
                        mulDeriv,
                        *tuple(list(self.__args__) + list(rhs.__args__)),
                        argPositions=posMap,
                    ),
                )
            return self.__generalMul__(rhs)
        if isinstance(rhs.__deriv__, MultiplicativeDerivation):
            return rhs.__mul__(self)
        return self.__generalMul__(rhs)

    def __rmul__(self, lhs: Union[float, int]) -> Self:
        assert isinstance(lhs, (int, float))
        assert self.numArgs == 0, "Only complete closures can be multiplied"
        if isinstance(self.__deriv__, AdditiveDerivation):
            addDeriv = AdditiveDerivation(
                self.name + "_rmul",
                self.__deriv__.derivs,
                resultPower=self.__deriv__.resultPower,
                linCoeffs=[
                    lhs ** (1 / self.__deriv__.resultPower) * coeff
                    for coeff in self.__deriv__.linCoeffs
                ],
            )
            return cast(
                Self,
                DerivationClosure(
                    addDeriv, *self.__args__, argPositions=self.__argPositions__
                ),
            )
        addDeriv = AdditiveDerivation(
            self.name + "_rmul", [self.__deriv__], linCoeffs=[lhs]
        )
        return cast(
            Self,
            DerivationClosure(
                addDeriv, *self.__args__, argPositions=self.__argPositions__
            ),
        )

    def __pow__(self, rhs: Union[float, int]) -> Self:
        assert isinstance(rhs, (int, float))
        assert self.numArgs == 0, "Only complete closures can be raised to powers"
        if isinstance(self.__deriv__, AdditiveDerivation):
            addDeriv = AdditiveDerivation(
                self.name + "_pow",
                self.__deriv__.derivs,
                resultPower=self.__deriv__.resultPower * rhs,
                linCoeffs=self.__deriv__.linCoeffs,
            )
            return cast(
                Self,
                DerivationClosure(
                    addDeriv, *self.__args__, argPositions=self.__argPositions__
                ),
            )
        addDeriv = AdditiveDerivation(
            self.name + "_pow", [self.__deriv__], linCoeffs=[1.0], resultPower=rhs
        )
        return cast(
            Self,
            DerivationClosure(
                addDeriv, *self.__args__, argPositions=self.__argPositions__
            ),
        )

    def registerComponents(self, container):
        container.register(self.__deriv__, ignoreDuplicates=True)


def funApply(funName: str, closure: DerivationClosure) -> DerivationClosure:
    """Apply a function to a complete closure (0 free arguments), resulting in a closure of a multiplicative derivation with that function.

    Args:
        funName (str): Fortran function name (see MultiplicativeDerivation for allowed names)
        closure (DerivationClosure): Closure to applu the function to

    Returns:
        DerivationClosure: Complete closure of a multiplicative derivation
    """
    assert (
        closure.numArgs == 0
    ), "Can only apply functions to complete derivation closures"
    if isinstance(closure.__deriv__, MultiplicativeDerivation):
        if closure.__deriv__.outerDeriv is None and closure.__deriv__.innerFunc is None:
            mulDeriv = MultiplicativeDerivation(
                funName + "_" + closure.name,
                innerDerivation=closure.__deriv__.innerDeriv,
                innerFunc=funName,
            )
            return DerivationClosure(
                mulDeriv, *closure.__args__, argPositions=closure.__argPositions__
            )
    mulDeriv = MultiplicativeDerivation(
        funName + "_" + closure.name,
        innerDerivation=closure.__deriv__,
        innerFunc=funName,
    )
    return DerivationClosure(
        mulDeriv, *closure.__args__, argPositions=closure.__argPositions__
    )


class PolynomialDerivation(GenericDerivation):
    """Polynomial derivation with arbitrary powers"""

    def __init__(
        self,
        name: str,
        constCoeff: Union[float, int],
        polyCoeffs: np.ndarray,
        polyPowers: np.ndarray,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Polynomial derivation with arbitrary powers

        Args:
            name (str): Name of the derivation
            constCoeff (Union[float, int]): Constant polynomial coefficient
            polyCoeffs (np.ndarray): Polynomial coefficients
            polyPowers (np.ndarray): Polynomial powers corresponding to each coefficient
            container (Optional[DerivationContainer], optional): Optional derivation container to register the derivation in at construction. Defaults to None.
        """
        assert len(polyCoeffs) == len(polyPowers)
        properties = {
            "type": "polynomialFunctionDerivation",
            "constantPolynomialCoefficient": constCoeff,
            "polynomialPowers": polyPowers.tolist(),
            "polynomialCoefficients": polyCoeffs.tolist(),
        }
        super().__init__(
            name, len(polyCoeffs), properties, latexTemplate=None, container=container
        )
        self.__constCoeff__ = constCoeff
        self.__polyCoeffs__ = polyCoeffs
        self.__polyPowers__ = polyPowers

    def latex(self, *args: str) -> str:
        assert (
            len(args) == self.numArgs
        ), "Unexpected number of arguments in PolynomialDerivation latex() call"
        result = (
            numToScientificTex(self.__constCoeff__) if self.__constCoeff__ != 0 else ""
        )
        for i, arg in enumerate(args):
            result += (
                "" if self.__polyCoeffs__[i] < 0 else (" + " if result != "" else "")
            )
            result += numToScientificTex(self.__polyCoeffs__[i], True)
            powerRepr = numToScientificTex(self.__polyCoeffs__[i])
            if powerRepr == "1.00":
                result += "\\left(" + arg + "\\right)^{" + powerRepr + "}"
            else:
                result += arg

        return result

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        assert (
            len(args) == self.totNumArgs
        ), "Unexpected number of arguments in PolynomialDerivation evaluate() call"
        result = self.__constCoeff__ * np.ones(args[0].shape)
        for i, arg in enumerate(args):
            result += self.__polyCoeffs__[i] * arg ** self.__polyPowers__[i]

        return result


class RangeFilterDerivation(Derivation):
    """A wrapper for a derivation, filtering it based on the values of arguments. All arguments, as well as the output of the derivation should conform in size."""

    def __init__(
        self,
        name: str,
        deriv: Derivation,
        filtering: List[Tuple[DerivationArgument, float, float]],
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """A wrapper for a derivation, filtering it based on the values of arguments. All arguments, as well as the output of the derivation should conform in size.

        Args:
            name (str): Name of the derivation
            deriv (Derivation): Filtered derivation
            filtering (List[Tuple[DerivationArgument, float, float]]): A list of filtering rules of the form (variable,lowerBound,upperBound), such that all variables must be within the bounds for the filter to return the derivation values.
            container (Optional[DerivationContainer], optional): Optional derivation container to register the derivation in at construction. Defaults to None.
        """
        super().__init__(name, deriv.numArgs, container=container)
        self.__deriv__ = deriv
        for _, range0, range1 in filtering:
            assert range0 < range1, "Ranges in RangeFilterDerivation must be monotonic"
        self.__filtering__ = filtering

    def dict(self) -> Dict:
        controlRangesDict = {}
        for i, fl in enumerate(self.__filtering__):
            _, range0, range1 = fl
            controlRangesDict["index" + str(i + 1)] = [range0, range1]

        deriv = {
            "type": "rangeFilterDerivation",
            "ruleName": self.__deriv__.name,
            "controlIndices": [i + 1 for i in range(len(self.__filtering__))],
            "controlRanges": controlRangesDict,
            "derivationIndices": [
                i + 1 + len(self.__filtering__) for i in range(self.numArgs)
            ],
        }

        return deriv

    def latex(self, *args: str) -> str:

        assert len(args) == self.numArgs + len(
            self.__filtering__
        ), "latex() on RangeFilterDerivation called with wrong number of arguments"
        result = (
            "\\left("
            + self.__deriv__.latex(*args[len(self.__filtering__) :])
            + "\\right)\\Pi\\left["
        )

        for i, fl in enumerate(self.__filtering__):
            _, range0, range1 = fl
            scale = range1 - range0
            offset = (range1 + range0) / 2
            sign = " - "
            if offset < 0:
                offset *= -1
                sign = " + "
            if i > 0:
                result += ", "
            result += (
                "\\left("
                + args[i]
                + sign
                + numToScientificTex(offset)
                + "\\right)/"
                + numToScientificTex(scale)
            )
        result += "\\right]"

        return result

    def evaluate(self, *args: np.ndarray) -> np.ndarray:

        filterVals: List[np.ndarray] = []
        for fl in self.__filtering__:
            filterVar, range0, range1 = fl
            filterVals.append(
                np.where(
                    (filterVar.data < range1) & (filterVar.data > range0),
                    np.ones(filterVar.data.shape),
                    np.zeros(filterVar.data.shape),
                )
            )

        result = self.__deriv__.evaluate(*args[len(self.__filtering__) :])
        for filter in filterVals:
            result *= filter

        return result

    def registerComponents(self, container: DerivationContainer):
        container.register(self.__deriv__)

    @property
    def enclosedArgs(self):
        return len(self.__filtering__) + self.__deriv__.enclosedArgs

    def fillArgs(self, *args: str) -> List[str]:
        argNames = [var.name for var, _, _ in self.__filtering__]
        argNames += self.__deriv__.fillArgs(*args)
        return argNames


class BoundedExtrapolationDerivation(Derivation):
    """Bounded extrapolation derivation, extrapolating a variable to one of the boundaries and optionally applying a lower and/or upper bound to the extrapolated value."""

    def __init__(
        self,
        name: str,
        extrapolationType: str = "lin",
        lowerBound: Union[float, DerivationArgument, None] = None,
        upperBound: Union[float, int, DerivationArgument, None] = None,
        container: Optional[DerivationContainer] = None,
        **kwargs,
    ):
        """Bounded extrapolation derivation, extrapolating a variable to one of the boundaries and optionally applying a lower and/or upper bound to the extrapolated value. Note that the upper and lower bounds must be properly ordered. If extrapolating to the left boundary, bounds are reflected around 0, i.e. the lower bound becomes a negative upper bound, and the upper bound becomes a negative lower bound.

        Args:
            name (str): Name of the derivation
            extrapolationType (str, optional): Extrapolation strategy - on of ["lin","log","loglin"]. "lin" in linear extrapolation, "log" is logarithmic extrapolation, and "linlog" is a mixture, performing logarithmic extrapolation using the value in the final cell and a linearly interpolated value on the penultimate cell. Defaults to "lin".
            lowerBound (Union[float, DerivationArgument, None], optional): Lower bound value, if using a variable will extrapolate it to the boundary. Defaults to None.
            upperBound (Union[float, int, DerivationArgument, None], optional): Upper bounda value, if using a variable will extrapolate it to the boundary. Defaults to None.
            container (Optional[DerivationContainer], optional): Optional container to register the derivation in at construction. Defaults to None.

        kwargs:
            leftBoundary (bool): Set to true if extrapolating to the left boundary
            staggeredVars (bool): Set to true if extrapolating staggered variables
            expectedHaloWidth (int): Expected halo size, set to 0 for modelbound variables
        """
        super().__init__(name, 1, container=container)
        assert extrapolationType in [
            "lin",
            "log",
            "linlog",
        ], "Unsupported extrapolation type detected"
        self.__extrapType__ = extrapolationType
        if isinstance(lowerBound, (float, int)):
            assert lowerBound >= 0, "Fixed lowerBound must be non-positive"
        if isinstance(upperBound, (float, int)):
            assert upperBound >= 0, "Fixed upperBound must be non-positive"

        if isinstance(lowerBound, (float, int)) and isinstance(
            upperBound, (float, int)
        ):
            assert (
                upperBound > lowerBound
            ), "Fixed upperBound must be greated than fixed lowerBound"
        self.__lowerBound__ = lowerBound
        self.__upperBound__ = upperBound

        self.__leftBoundary__: bool = kwargs.get("leftBoundary", False)
        self.__staggeredVars__: bool = kwargs.get("staggeredVars", False)
        self.__expectedHaloWidth__: Union[int, None] = kwargs.get(
            "expectedHaloWidth", None
        )

    def dict(self):

        deriv = {
            "type": "boundedExtrapolationDerivation",
            "expectUpperBoundVar": isinstance(self.__upperBound__, DerivationArgument),
            "expectLowerBoundVar": isinstance(self.__lowerBound__, DerivationArgument),
            "ignoreUpperBound": self.__upperBound__ is None,
            "ignoreLowerBound": self.__lowerBound__ is None,
            "fixedLowerBound": (
                self.__lowerBound__
                if isinstance(self.__lowerBound__, (float, int))
                else 0
            ),
            "extrapolationStrategy": {
                "type": self.__extrapType__ + "Extrapolation",
                "leftBoundary": self.__leftBoundary__,
                "staggeredVars": self.__staggeredVars__,
            },
        }

        if isinstance(self.__upperBound__, (float, int)):
            deriv["fixedUpperBound"] = self.__upperBound__

        if isinstance(self.__expectedHaloWidth__, int):
            deriv["extrapolationStrategy"].update(
                {"expectedHaloWidth": self.__expectedHaloWidth__}
            )

        return deriv

    @property
    def enclosedArgs(self):
        return int(isinstance(self.__lowerBound__, DerivationArgument)) + int(
            isinstance(self.__upperBound__, DerivationArgument)
        )

    def latex(self, *args):
        var = args[0]
        lowerBound = ""
        if isinstance(self.__lowerBound__, (float, int)):
            lowerBound = numToScientificTex(self.__lowerBound__)
        if isinstance(self.__lowerBound__, DerivationArgument):
            lowerBound = args[1]

        upperBound = ""
        if isinstance(self.__upperBound__, DerivationArgument):
            upperBound = (
                args[2]
                if isinstance(self.__lowerBound__, DerivationArgument)
                else args[1]
            )
        if isinstance(self.__upperBound__, (float, int)):
            upperBound = numToScientificTex(self.__upperBound__)

        side = "L" if self.__leftBoundary__ else "R"
        return (
            "\\text{Extrap}_{"
            + side
            + ","
            + self.__extrapType__
            + "}\\left("
            + var
            + "\\right)^{"
            + upperBound
            + "}_{"
            + lowerBound
            + "}"
        )

    def fillArgs(self, *args):
        newArgs = [args[0]]
        if isinstance(self.__lowerBound__, DerivationArgument):
            newArgs.append(self.__lowerBound__.name)
        if isinstance(self.__upperBound__, DerivationArgument):
            newArgs.append(self.__upperBound__.name)
        return newArgs

    @property
    def resultProperties(self):
        return {"isScalar": True, "isDistribution": False, "isSingleHarmonic": False}


def coldIonIDeriv(name: str, index: int) -> GenericDerivation:
    """Shkarofsky I integral assuming cold flowing ions. Does not include the ion density factor. The single argument is assumed to be the ion flow speed.

    Args:
        name (str): Name of the derivation
        index (int): Integral index

    Returns:
        GenericDerivation: Cold ion I integral derivation
    """
    deriv = {
        "type": "coldIonIJIntegralDerivation",
        "isJIntegral": False,
        "index": index,
    }
    return GenericDerivation(
        name, 1, deriv, "I_{c," + str(index) + "}\\left($0\\right)"
    )


def coldIonJDeriv(name: str, index: int):
    """Shkarofsky J integral assuming cold flowing ions. Does not include the ion density factor. The single argument is assumed to be the ion flow speed.

    Args:
        name (str): Name of the derivation
        index (int): Integral index

    Returns:
        GenericDerivation: Cold ion J integral derivation
    """
    deriv = {
        "type": "coldIonIJIntegralDerivation",
        "isJIntegral": True,
        "index": index,
    }
    return GenericDerivation(
        name, 1, deriv, "J_{c," + str(index) + "}\\left($0\\right)"
    )


def shkarofskyIIntegralDeriv(name: str, index: int):
    """Shkarofsky I integral. The single argument is assumed to be the electron distribution.

    Args:
        name (str): Name of the derivation
        index (int): Integral index

    Returns:
        GenericDerivation: I integral derivation
    """
    deriv = {
        "type": "IJIntegralDerivation",
        "isJIntegral": False,
        "index": index,
    }
    return GenericDerivation(name, 1, deriv, "I_{" + str(index) + "}\\left($0\\right)")


def shkarofskyJIntegralDeriv(name: str, index: int):
    """Shkarofsky I integral. The single argument is assumed to be the electron distribution.

    Args:
        name (str): Name of the derivation
        index (int): Integral index

    Returns:
        GenericDerivation: I integral derivation
    """
    deriv = {
        "type": "IJIntegralDerivation",
        "isJIntegral": True,
        "index": index,
    }
    return GenericDerivation(name, 1, deriv, "J_{" + str(index) + "}\\left($0\\right)")


class HarmonicExtractorDerivation(Derivation):
    """Harmonic extractor derivation, producing a single harmonic variable from a distribution"""

    def __init__(
        self,
        name: str,
        grid: Grid,
        index: int,
        container: Optional[DerivationContainer] = None,
    ):
        """Harmonic extractor derivation, producing a single harmonic variable from a distribution

        Args:
            name (str): Name of the derivation
            grid (Grid): Grid object used in evaluation
            index (int): Harmonic index (Fortran 1-indexing)
            container (Optional[DerivationContainer], optional): Optional container to register the derivation in at construction. Defaults to None.
        """
        super().__init__(name, 1, container=container)
        self.__grid__ = grid
        assert index > 0, "Harmonic index must be positive"
        assert index <= self.__grid__.numH, "Harmonic index out of bounds"
        self.__index__ = index

    def dict(self):
        return {"type": "harmonicExtractorDerivation", "index": self.__index__}

    def latex(self, *args):
        assert (
            len(args) == self.numArgs
        ), "Wrong number of args passed to HarmonicExtractorDerivation latex()"
        return "\\left(" + args[0] + "\\right)_{h=" + str(self.__index__) + "}"

    def evaluate(self, *args):
        assert (
            len(args) == self.numArgs
        ), "Wrong number of args passed to HarmonicExtractorDerivation evaluate()"
        assert args[0].shape == (
            self.__grid__.numX,
            self.__grid__.numH,
            self.__grid__.numV,
        ), "args[0] of HarmonicExtractorDerivation must be a full distribution"

        return args[0][:, self.__index__ - 1, :]

    @property
    def resultProperties(self):
        return {"isScalar": False, "isDistribution": False, "isSingleHarmonic": True}


class DDVDerivation(Derivation):
    """Velocity derivative derivation acting on a harmonic of a distribution variable with optional velocity function inside and/or outside of the derivative v_o * d(v_i * f)/dv"""

    def __init__(
        self,
        name: str,
        grid: Grid,
        harmonicIndex: int,
        innerV: Optional[Profile] = None,
        outerV: Optional[Profile] = None,
        vifAtZero: Optional[Tuple[float, float]] = None,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Velocity derivative derivation acting on a harmonic of a distribution variable with optional velocity function inside and/or outside of the derivative v_o * d(v_i * f)/dv

        Args:
            name (str): Name of the derivation
            grid (Grid): Grid object used ensure profile consistency
            harmonicIndex (int): Index of the differentiated harmonic
            innerV (Optional[Profile], optional): Optional velocity space profile inside the derivative (at right velocity cell edges). Defaults to None.
            outerV (Optional[Profile], optional): Optional velocity space profile outside the derivative. Defaults to None.
            vifAtZero (Optional[Tuple[float,float]], optional): Extrapolation coefficents (A1,A2) of v_i*f at zero in the form A1*f(v1)+A2*f(v2). Defaults to None, which results in (0,0).
            container (Optional[DerivationContainer], optional): Optiona container to register the distribution in at construction. Defaults to None.
        """
        super().__init__(name, 1, container=container)
        self.__grid__ = grid
        self.__harmonicIndex__ = harmonicIndex
        if innerV is not None:
            assert (
                innerV.dim == "V" and len(innerV.data) == grid.numV
            ), "innerV size does not conform to velocity grid"
        self.__innerV__ = innerV
        if outerV is not None:
            assert (
                outerV.dim == "V" and len(outerV.data) == grid.numV
            ), "outerV size does not conform to velocity grid"
        self.__outerV__ = outerV
        self.__vifAtZero__ = vifAtZero

    def dict(self) -> Dict:
        deriv: Dict[str, object] = {
            "type": "ddvDerivation",
            "targetH": self.__harmonicIndex__,
        }
        if self.__outerV__ is not None:
            deriv["outerV"] = self.__outerV__.data.tolist()
        if self.__innerV__ is not None:
            deriv["innerV"] = self.__innerV__.data.tolist()
        if self.__vifAtZero__ is not None:
            deriv["vifAtZero"] = list(self.__vifAtZero__)

        return deriv

    def latex(self, *args: str) -> str:
        assert self.numArgs == len(
            args
        ), "latex() on DDVDerivation called with wrong number of arguments"
        outerV = "" if self.__outerV__ is None else "V_o"
        if self.__innerV__ is None:
            return (
                outerV
                + "\\frac{\\partial {"
                + args[0]
                + "}_{h="
                + str(self.__harmonicIndex__)
                + "}}{\\partial v}"
            )
        return (
            outerV
            + "\\frac{\\partial }{\\partial v}\\left(V_i {"
            + args[0]
            + "}_{h="
            + str(self.__harmonicIndex__)
            + "}\\right)"
        )

    @property
    def resultProperties(self):
        return {"isScalar": False, "isDistribution": False, "isSingleHarmonic": True}


class D2DV2Derivation(Derivation):
    """Velocity space diffusion derivation acting on a single harmonic of a distribution variable. Can set outer and/or inner velocity profiles to calculate v_o * d(v_i * df/dv)/dv"""

    def __init__(
        self,
        name: str,
        grid: Grid,
        harmonicIndex: int,
        innerV: Optional[Profile] = None,
        outerV: Optional[Profile] = None,
        vidfdvAtZero: Optional[Tuple[float, float]] = None,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Velocity space diffusion derivation acting on a single harmonic of a distribution variable. Can set outer and/or inner velocity profiles to calculate v_o * d(v_i * df/dv)/dv

        Args:
            name (str): Name of the derivation
            grid (Grid): Grid used to check the consistency of the profiles
            harmonicIndex (int): Index of the harmonic the derivative is taken on
            innerV (Optional[Profile], optional): Diffusion coefficient (at right velocity cell edges). Defaults to None.
            outerV (Optional[Profile], optional): Velocity profile outside of the derivative. Defaults to None.
            vidfdvAtZero (Optional[Tuple[float,float]], optional): Extrapolation coefficents (A1,A2) of v_i*df/dv at zero in the form A1*f(v1)+A2*f(v2). Defaults to None, which results in (0,0).
            container (Optional[DerivationContainer], optional): Optional container to register the derivation in at construction. Defaults to None.
        """
        super().__init__(name, 1, container=container)
        self.__grid__ = grid
        self.__harmonicIndex__ = harmonicIndex
        if innerV is not None:
            assert (
                innerV.dim == "V" and len(innerV.data) == grid.numV
            ), "innerV size does not conform to velocity grid"
        self.__innerV__ = innerV
        if outerV is not None:
            assert (
                outerV.dim == "V" and len(outerV.data) == grid.numV
            ), "outerV size does not conform to velocity grid"
        self.__outerV__ = outerV
        self.__vidfdvAtZero__ = vidfdvAtZero

    def dict(self) -> Dict:
        deriv: Dict[str, object] = {
            "type": "d2dv2Derivation",
            "targetH": self.__harmonicIndex__,
        }
        if self.__outerV__ is not None:
            deriv["outerV"] = self.__outerV__.data.tolist()
        if self.__innerV__ is not None:
            deriv["innerV"] = self.__innerV__.data.tolist()
        if self.__vidfdvAtZero__ is not None:
            deriv["vidfdvAtZero"] = list(self.__vidfdvAtZero__)

        return deriv

    def latex(self, *args: str) -> str:
        assert self.numArgs == len(
            args
        ), "latex() on D2DV2Derivation called with wrong number of arguments"

        outerV = "" if self.__outerV__ is None else "V_o"
        if self.__innerV__ is None:
            return (
                outerV
                + "\\frac{\\partial^2 {"
                + args[0]
                + "}_{h="
                + str(self.__harmonicIndex__)
                + "}}{\\partial v^2}"
            )
        inner = (
            "\\frac{\\partial {"
            + args[0]
            + "}_{h="
            + str(self.__harmonicIndex__)
            + "}}{\\partial v}"
        )
        return (
            outerV + "\\frac{\\partial }{\\partial v}\\left(V_i " + inner + "\\right)"
        )

    @property
    def resultProperties(self):
        return {"isScalar": False, "isDistribution": False, "isSingleHarmonic": True}


class MomentDerivation(Derivation):
    """Derivation taking the moment of a single harmonic of a distribution variable (optionally multiplied by a velocity space vector), optionally multiplying it with a variables raised to powers and a scalar."""

    def __init__(
        self,
        name: str,
        grid: Grid,
        momentHarmonic: int,
        momentOrder: int,
        multConst: float = 1.0,
        varPowers: Optional[List[int]] = None,
        gVec: Optional[Profile] = None,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Derivation taking the moment of a single harmonic of a distribution variable (optionally multiplied by a velocity space vector), optionally multiplying it with a variables raised to powers and a scalar. If variables are present, it is assumed that the first argument of the derivation is the distribution, and then the variables in order.

        Args:
            name (str): Name of the derivation
            grid (Grid): Grid object used to check consistency of velocity space vector
            momentHarmonic (int): Harmonic to take moment of
            momentOrder (int): Moment order
            multConst (float, optional): Constant to multiply the moment with. Defaults to 1.0.
            varPowers (Optional[List[int]], optional): Optional fluid variable argument powers. Defaults to None.
            gVec (Optional[Profile], optional): Optional velocity space vector to multiply the distribution with before taking the moment. Defaults to None.
            container (Optional[DerivationContainer], optional): Optional container to register the derivation in at construction. Defaults to None.
        """
        self.__varPowers__: List[int] = varPowers if varPowers is not None else []
        super().__init__(name, 1 + len(self.__varPowers__), container=container)
        self.__momentHarmonic__ = momentHarmonic
        self.__momentOrder__ = momentOrder
        self.__multConst__ = multConst
        self.__grid__ = grid
        if gVec is not None:
            assert (
                gVec.dim == "V" and len(gVec.data) == grid.numV
            ), "gVec size does not conform to velocity grid"
        self.__gVec__ = gVec

    def dict(self) -> Dict:
        deriv = {
            "type": "momentDerivation",
            "momentHarmonic": self.__momentHarmonic__,
            "momentOrder": self.__momentOrder__,
            "multConst": self.__multConst__,
            "varPowers": self.__varPowers__,
        }

        if self.__gVec__ is not None:
            deriv["gVector"] = list(self.__gVec__.data)

        return deriv

    def latex(self, *args: str) -> str:
        assert (
            len(args) == self.numArgs
        ), "latex() on MomentDerivation called with wrong number of arguments"
        expression = numToScientificTex(self.__multConst__, removeUnity=True)
        numerator = ""
        denominator = ""
        if len(args) > 1:

            for i, arg in enumerate(args[1:]):
                if self.__varPowers__[i] > 0:
                    if isclose(self.__varPowers__[i], 1.0, rel_tol=1e-4):
                        numerator += " " + arg
                    else:
                        power = (
                            f"{round(self.__varPowers__[i])}"
                            if isclose(
                                self.__varPowers__[i],
                                round(self.__varPowers__[i]),
                                rel_tol=1e-2,
                            )
                            else f"{self.__varPowers__[i]:.2f}"
                        )
                        numerator += " " + arg + "^{" + power + "}"
                else:
                    if isclose(self.__varPowers__[i], -1.0, rel_tol=1e-4):
                        denominator += " " + arg
                    else:
                        power = (
                            f"{round(-self.__varPowers__[i])}"
                            if isclose(
                                self.__varPowers__[i],
                                round(self.__varPowers__[i]),
                                rel_tol=1e-2,
                            )
                            else f"{-self.__varPowers__[i]:.2f}"
                        )
                        denominator += " " + arg + "^{" + power + "}"
            if len(denominator):
                expression += "\\frac{" + numerator + "}{" + denominator + "}"
            else:
                expression += numerator
        g = ""
        if self.__gVec__ is not None:
            g = "g(v)"
        return (
            expression
            + "\\langle "
            + g
            + args[0]
            + " \\rangle _{h="
            + str(self.__momentHarmonic__)
            + ",n="
            + str(self.__momentOrder__)
            + "}"
        )

    @property
    def resultProperties(self):
        return {"isScalar": False, "isDistribution": False, "isSingleHarmonic": False}


class GenIntPolynomialDerivation(Derivation):
    """Generalised polynomial derivation with integer powers. Calculates a polynomial in multiple variables and optionally applies a function to the result with a multiplicative constant - multConst * fun(sum c_i prod vars**powers), where prod vars**powers is shorthand for the product of passed variables raised to powers corresponding to polynomial coefficient c_i"""

    def __init__(
        self,
        name: str,
        polyPowers: np.ndarray,
        polyCoeffs: np.ndarray,
        multConst: float = 1.0,
        funcName: Union[None, str] = None,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Generalised polynomial derivation with integer powers. Calculates a polynomial in multiple variables and optionally applies a function to the result with a multiplicative constant - multConst * fun(sum c_i prod vars**powers), where prod vars**powers is shorthand for the product of passed variables raised to powers corresponding to polynomial coefficient c_i

        Args:
            name (str): Name of the derivation
            polyPowers (np.ndarray): An n x m matrix where n corresponds to the coefficient, and m corresponds to the variable. For example, for a function of two variables, this is an n x 2 matrix, so that the contribution to the polynomial of the i-th element is c_i * var1**polyPowers[i,0] * var2**polyPowers[i,1].
            polyCoeffs (np.ndarray): Polynomial coefficients corresponding to each product of variables raised to powers
            multConst (float, optional): Optional multiplicative constant to multiply the result of the derivation with. Defaults to 1.0.
            funcName (Union[None, str], optional): Optional Fortran function name to apply to the polynomial result before multiplication by multConst. For allowed functions see MultiplicativeDerivation. Defaults to None.
            container (Optional[DerivationContainer], optional): Optional container to register the derivation in at construction. Defaults to None.
        """
        super().__init__(name, polyPowers.shape[1], container=container)
        self.__polyPowers__ = polyPowers
        self.__polyCoeffs__ = polyCoeffs
        self.__multConst__ = multConst

        funMap = {
            "exp": np.exp,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
            "abs": np.abs,
            "tan": np.tan,
            "atan": np.arctan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "sign": np.sign,
            "erf": special.erf,
            "erfc": special.erfc,
        }

        self.__funcName__: Optional[str] = funcName
        self.__fun__: Union[Callable, None] = None
        if self.__funcName__ is not None:
            assert (
                self.__funcName__ in funMap
            ), "Unsupported function passed to GenIntPolynomialDerivation"
            self.__fun__ = funMap[self.__funcName__]

    def dict(self) -> Dict:
        polyPowersDict = {}
        for i in range(self.__polyPowers__.shape[0]):
            polyPowersDict["index" + str(i + 1)] = self.__polyPowers__[i, :].tolist()

        deriv = {
            "type": "generalizedIntPowerPolyDerivation",
            "multConst": self.__multConst__,
            "polynomialPowers": polyPowersDict,
            "polynomialCoefficients": self.__polyCoeffs__.tolist(),
        }

        if self.__funcName__ is not None:
            deriv["functionName"] = self.__funcName__

        return deriv

    def latex(self, *args: str) -> str:
        assert (
            len(args) == self.numArgs
        ), "Unexpected number of arguments in GenIntPolynomialDerivation latex() call"
        expression = numToScientificTex(self.__multConst__, removeUnity=True)
        innerExpr = "\\sum_i c_i " + " ".join(
            [arg + "^{p_{i," + str(j) + "}}" for j, arg in enumerate(args)]
        )
        if self.__funcName__ is not None:
            return expression + self.__funcName__ + "\\left( " + innerExpr + "\\right)"

        return expression + innerExpr

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        assert (
            len(args) == self.numArgs
        ), "Unexpected number of arguments in GenIntPolynomialDerivation evaluate() call"
        result = np.zeros(args[0].shape)
        for ind, coeff in np.ndenumerate(self.__polyCoeffs__):
            temp = coeff
            for j, arg in enumerate(args):
                temp *= arg ** self.__polyPowers__[ind[0], j]
            result += temp
        if self.__funcName__ is not None:
            result = cast(Callable, self.__fun__)(result)
        return result * self.__multConst__


class LocValExtractorDerivation(Derivation):
    """Derivation extracting a scalar value at a given spatial location from a fluid variable"""

    def __init__(
        self,
        name: str,
        grid: Grid,
        targetX: int,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """Derivation extracting a scalar value at a given spatial location from a fluid variable

        Args:
            name (str): Name of the derivation
            grid (Grid): Grid object used for bounds checking
            targetX (int): Target spatial index (Fortran 1-indexing)
            container (Optional[DerivationContainer], optional): Optional container to register the derivation in at construction. Defaults to None.
        """
        super().__init__(name, 1, container=container)
        self.__grid__ = grid
        self.__targetX__ = targetX
        assert (
            targetX > 0 and targetX <= grid.numX
        ), "LocValExtractorDerivation targetX out of bounds"

    def dict(self) -> Dict:
        return {"type": "locValExtractorDerivation", "targetX": self.__targetX__}

    def latex(self, *args: str) -> str:
        assert (
            len(args) == self.numArgs
        ), "LocValExtractorDerivation latex() call unexpected number of arguments"
        return "\\left(" + args[0] + "\\right)_{X=" + str(self.__targetX__) + "}"

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        assert (
            len(args) == self.numArgs
        ), "LocValExtractorDerivation evaluate() call unexpected number of arguments"
        assert args[0].shape == (
            self.__grid__.numX,
        ), "LocValExtractorDerivation evaluate() argument does not conform to the X dimension of the grid"
        return np.array([args[0][self.__targetX__ - 1]])

    @property
    def resultProperties(self):
        return {"isScalar": True, "isDistribution": False, "isSingleHarmonic": False}


class NDInterpolationDerivation(Derivation):
    """N-dimensional linear interpolation assuming data on Cartesian N-D grid. Values outside of the hypercube are set to 0."""

    def __init__(
        self,
        name: str,
        grids: List[np.ndarray],
        data: np.ndarray,
        container: Optional[DerivationContainer] = None,
    ) -> None:
        """N-dimensional linear interpolation assuming data on Cartesian N-D grid. Values outside of the hypercube are set to 0.

        Args:
            name (str): Name of the derivation
            grids (List[np.ndarray]): Coordinates of individual dimensions
            data (np.ndarray): N-dimensional data corresponding to the points determined by the grids
            container (Optional[DerivationContainer], optional): Optional container to register the derivation in at construction. Defaults to None.
        """
        super().__init__(name, len(grids), container=container)
        self.__grids__ = grids
        self.__data__ = data
        dataShape = np.shape(data)
        assert len(grids) == len(
            dataShape
        ), "The number of orthogonal grids in NDInterpolationDerivation must be the equal to the number of data dimensions"

        for i, grid in enumerate(grids):
            assert len(grid) == dataShape[i], (
                "Grid "
                + str(i)
                + " does not conform to the corresponding dimension of interpolation data"
            )

    def dict(self) -> Dict:
        deriv = {
            "type": "nDLinInterpDerivation",
            "data": {
                "dims": list(self.__data__.shape),
                "values": self.__data__.flatten(order="F").tolist(),
            },
            "grids": {"names": ["grid" + str(i) for i in range(len(self.__grids__))]},
        }
        for i, name in enumerate(["grid" + str(i) for i in range(len(self.__grids__))]):
            cast(Dict[str, object], deriv["grids"])[name] = self.__grids__[i].tolist()

        return deriv

    def latex(self, *args: str) -> str:
        return "\\mathcal{I}_{ND}\\left(" + ",".join(args) + "\\right)"

    def evaluate(self, *args: np.ndarray) -> np.ndarray:

        return RegularGridInterpolator(
            tuple(self.__grids__), self.__data__, fill_value=0
        )(np.array(list(zip(*args))))
