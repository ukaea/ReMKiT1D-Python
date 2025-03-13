from typing import Union, List, Dict, cast, Tuple, Optional
from .variable_container import VariableContainer, Variable, MultiplicativeArgument
from .derivations import Textbook, GenericDerivation, DerivationClosure
from typing_extensions import Self
from .grid import Profile, Grid
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
import pylatex as tex  # type: ignore
from .tex_parsing import numToScientificTex


class ModelboundData(ABC):
    """Abstract base class for modelbound data"""

    @abstractmethod
    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        pass

    @abstractmethod
    def dict(self) -> dict:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> Variable:
        pass

    @property
    @abstractmethod
    def varNames(self) -> List[str]:
        pass

    def registerDerivs(self, container: Textbook):
        pass


class Term(ABC):
    """Abstract base term class"""

    def __init__(
        self,
        name: str,
        evolvedVar: Optional[Variable] = None,
        implicitGroups: Optional[List[int]] = None,
        generalGroups: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.__name__ = name
        self.__evolvedVar__ = evolvedVar
        self.__implicitGroups__ = implicitGroups if implicitGroups is not None else []
        self.__generalGroups__ = generalGroups if generalGroups is not None else []

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name: str):
        self.__name__ = name

    def rename(self, name: str):
        """Return a renamed copy of this term

        Args:
            name (str): New term name
        """
        newTerm = deepcopy(self)
        newTerm.__name__ = name
        return newTerm

    @property
    def evolvedVar(self):
        return self.__evolvedVar__

    @evolvedVar.setter
    def evolvedVar(self, name: Variable):
        self.__evolvedVar__ = name

    def withEvolvedVar(self, name: str) -> Self:
        """Change the evolved variable of this term and return the term

        Args:
            name (str): New evolved variable name
        """
        self.evolvedVar = name
        return self

    @property
    def implicitGroups(self):
        return self.__implicitGroups__

    @property
    def generalGroups(self):
        return self.__generalGroups__

    def regroup(
        self,
        implicitGroups: Optional[List[int]] = None,
        generalGroups: Optional[List[int]] = None,
    ):
        """Change the implicit/general groups of the term and return the temr

        Args:
            implicitGroups (Optional[List[int]], optional): If present, will change the implicit groups of the term. Defaults to None.
            generalGroups (Optional[List[int]], optional): If present, will change the general groups of the term. Defaults to None.
        """
        if implicitGroups is not None:
            self.__implicitGroups__ = implicitGroups
        if generalGroups is not None:
            self.__generalGroups__ = generalGroups
        return self

    @abstractmethod
    def dict(self) -> dict:
        pass

    def checkTerm(
        self, varCont: VariableContainer, mbData: Optional[ModelboundData] = None
    ):

        assert len(self.implicitGroups) or len(self.generalGroups), (
            "Both implicit and general groups empty in term " + self.name
        )

        assert self.__evolvedVar__ is not None, "Term evolvedVar not set"

        assert self.__evolvedVar__.name in varCont.varNames, (
            "Evolved variable "
            + self.__evolvedVar__.name
            + " not registered in used variable container"
        )

    @abstractmethod
    def latex(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def __add__(self, rhs: Self):
        pass

    @abstractmethod
    def registerDerivs(self, container: Textbook):
        pass

    def __rmul__(self, rhs: Union[int, float]):
        raise NotImplementedError(
            "Attempted __rmul__ for Term type which doesn't have it implemented"
        )

    def __neg__(self):
        raise NotImplementedError(
            "Attempted __neg__ for Term type which doesn't have it implemented"
        )


class TermCollection:
    """A collection of terms evolving a single variable. Provides access to the underlying terms via their names, as well as handling some term arithmetic operations."""

    def __init__(
        self,
        evolvedVar: Variable,
        modelName: str,
        latexName: str,
        derivativeTex: str = "\\partial",
    ):
        """A collection of terms evolving a single variable. Provides access to the underlying terms via their names, as well as handling some term arithmetic operations.

        Args:
            evolvedVar (Variable): The evolved variable (common to all terms in the collection)
            modelName (str): Name of the model this collection belongs to
            latexName (str): LaTeX-compatible string name for this collection
            derivativeTex (str, optional): The derivative symbol used in the time derivative representation of this collection. Defaults to "\\partial".
        """
        self.__evolvedVar__ = evolvedVar
        self.__modelName__ = modelName
        self.__latexName__ = latexName
        self.__derivativeTex__ = derivativeTex
        self.__terms__: List[Term] = []

    @property
    def terms(self):
        return self.__terms__

    @property
    def termNames(self):
        return [term.name for term in self.__terms__]

    @property
    def evolvedVar(self):
        return self.__evolvedVar__

    @property
    def modelName(self):
        return self.__modelName__

    @property
    def activeImplicitGroups(self):
        """Return list of active implicit groups in the collection (with duplicates)"""
        return sum([term.implicitGroups for term in self.__terms__], [])

    @property
    def activeGeneralGroups(self):
        """Return list of active general groups in the collection (with duplicates)"""
        return sum([term.generalGroups for term in self.__terms__], [])

    def __getitem__(self, key: str):
        if key not in self.termNames:
            raise KeyError()
        return self.__terms__[self.termNames.index(key)]

    def __setitem__(self, key: str, term: Term):
        if key not in self.termNames:
            self.__terms__.append(term)
        else:
            self.__terms__[self.termNames.index(key)] = term

    def __delitem__(self, key: str):
        if key not in self.termNames:
            raise KeyError()
        del self.__terms__[self.termNames.index(key)]

    def dict(self):
        return {term.name: term.dict() for term in self.__terms__}

    def addLatexToDoc(self, doc: tex.Alignat, latexRemap: Dict[str, str] = {}):
        evolvedVarTex = (
            latexRemap[self.__evolvedVar__.name]
            if self.__evolvedVar__.name in latexRemap
            else "\\text{" + self.__evolvedVar__.name.replace("_", r"\_") + "}"
        )
        doc.append(
            "\\left(\\frac{"
            + self.__derivativeTex__
            + " "
            + evolvedVarTex
            + "}{"
            + self.__derivativeTex__
            + " t}\\right)_{"
            + self.__latexName__
            + "} &= "
        )

        for i, term in enumerate(self.__terms__):
            buffer = ("+ " if i > 0 else " ") + term.latex(latexRemap=latexRemap)
            if i == 0:
                doc.append(
                    (buffer[1:] if buffer.startswith("+  -") else buffer) + "\\\\"
                )
            else:
                doc.extend(
                    [
                        "&"
                        + (buffer[1:] if buffer.startswith("+  -") else buffer)
                        + "\\\\"
                    ]
                )

    def __add__(self, rhs: Union[Term, Self]) -> Self:
        assert isinstance(
            rhs, (Term, TermCollection)
        ), "cannot add non-Terms to term collection"
        if isinstance(rhs, Term):
            assert (
                rhs.name not in self.termNames
            ), "duplicate term name in TermCollection - use Term.rename()"
            assert (
                rhs.evolvedVar is None
                or rhs.evolvedVar.name == self.__evolvedVar__.name
            ), "Cannot add Term to TermCollection if it evolves a variable different from the collection"
            newCollection = deepcopy(self)
            newCollection.__terms__.append(deepcopy(rhs))
            newCollection.__terms__[-1].evolvedVar = self.__evolvedVar__

        if isinstance(rhs, TermCollection):
            newCollection = deepcopy(self)
            for term in rhs.terms:
                newCollection += term.withEvolvedVar(self.evolvedVar)

        return newCollection

    def __sub__(self, rhs: Union[Term, Self]):

        assert isinstance(
            rhs, (Term, TermCollection)
        ), "cannot add (negative) non-Terms to term collection"
        if isinstance(rhs, Term):
            assert (
                rhs.name not in self.termNames
            ), "duplicate term name in TermCollection - use Term.rename()"
            assert (
                rhs.evolvedVar is None
                or rhs.evolvedVar.name == self.__evolvedVar__.name
            ), "Cannot add (negative) Term to TermCollection if it evolves a variable different from the collection"
            newCollection = deepcopy(self)
            newCollection.__terms__.append(deepcopy(-1 * rhs))
            newCollection.__terms__[-1].evolvedVar = self.__evolvedVar__

        if isinstance(rhs, TermCollection):
            newCollection = deepcopy(self)
            for term in rhs.terms:
                newCollection -= term.withEvolvedVar(self.evolvedVar)

        return newCollection

    def __neg__(self):
        newCollection = TermCollection(
            self.evolvedVar, self.modelName, self.__latexName__, self.__derivativeTex__
        )
        for term in self.terms:
            newCollection -= term
        return newCollection

    def withSuffix(self, suffix: str):
        """Append a suffix to all terms in this collection and return as copy

        Args:
            suffix (str): Suffix to append
        """
        newCollection = deepcopy(self)
        for term in newCollection.terms:
            term.name += suffix
        return newCollection

    def filterByGroup(
        self, groups: Optional[List[int]] = None, general: bool = False
    ) -> Self:
        """Produce a new term collection containing only terms in a given group

        Args:
            group (Optional[List[int]]): Group indices (Fortran 1-indexing). Defaults to [1]
            general (bool, optional): If true, will filter with respect to general groups, otherwise filters with respect to implicit groups. Defaults to False.
        """
        newCollection = TermCollection(
            self.evolvedVar,
            modelName=self.modelName,
            latexName=self.__latexName__,
            derivativeTex=self.__derivativeTex__,
        )

        usedGroups = groups if groups is not None else [1]
        for term in self.__terms__:
            if general:
                if any(group in term.generalGroups for group in usedGroups):
                    newCollection += term
            else:
                if any(group in term.implicitGroups for group in usedGroups):
                    newCollection += term

        return cast(Self, newCollection)

    def checkTerms(
        self, varCont: VariableContainer, mbData: Optional[ModelboundData] = None
    ):

        for term in self.__terms__:
            print("   Checking term " + term.name)
            term.checkTerm(varCont, mbData)

    def registerDerivs(self, container: Textbook):
        for term in self.__terms__:
            term.registerDerivs(container)


class VarData:
    """A container for required variable data in matrix terms"""

    def __init__(
        self,
        reqRowVars: Optional[MultiplicativeArgument] = None,
        reqColVars: Optional[MultiplicativeArgument] = None,
        reqMBRowVars: Optional[MultiplicativeArgument] = None,
        reqMBColVars: Optional[MultiplicativeArgument] = None,
    ) -> None:
        """A container for required variable data in matrix terms

        Args:
            reqRowVars (MultiplicativeArgument, optional): Required row variables. Defaults to MultiplicativeArgument().
            reqColVars (MultiplicativeArgument, optional): Required column variables. Defaults to MultiplicativeArgument().
            reqMBRowVars (MultiplicativeArgument, optional): Required row modelbound variables. Defaults to MultiplicativeArgument().
            reqMBColVars (MultiplicativeArgument, optional): Required column modelbound variables. Defaults to MultiplicativeArgument().
        """
        rrV = reqRowVars if reqRowVars is not None else MultiplicativeArgument()
        rcV = reqColVars if reqColVars is not None else MultiplicativeArgument()
        rmbrV = reqMBRowVars if reqMBRowVars is not None else MultiplicativeArgument()

        rmbcV = reqMBColVars if reqMBColVars is not None else MultiplicativeArgument()

        self.__reqRowVars___ = list(rrV.argMultiplicity.keys())
        self.__reqRowPowers___ = list(rrV.argMultiplicity.values())
        self.__reqColVars___ = list(rcV.argMultiplicity.keys())
        self.__reqColPowers___ = list(rcV.argMultiplicity.values())
        self.__reqMBRowVars___ = list(rmbrV.argMultiplicity.keys())
        self.__reqMBRowPowers___ = list(rmbrV.argMultiplicity.values())
        self.__reqMBColVars___ = list(rmbcV.argMultiplicity.keys())
        self.__reqMBColPowers___ = list(rmbcV.argMultiplicity.values())

    def checkRowColVars(
        self,
        varCont: VariableContainer,
        rowVarOnDual=False,
        colVarOnDual=False,
        mbData: Optional[ModelboundData] = None,
    ):

        for var in self.__reqRowVars___:
            assert var in varCont.varNames, (
                "Required row variable " + var + " not found in used variable container"
            )

            if not varCont[var].isScalar:
                if varCont[var].isOnDualGrid is not rowVarOnDual:
                    warnings.warn(
                        "Variable "
                        + var
                        + " appears in required row variables for evolved variable on "
                        + ("dual" if rowVarOnDual else "regular")
                        + " grid but doesn't live on that grid - ignore if evolved variable is a distribution"
                    )

        for var in self.__reqColVars___:
            assert var in varCont.varNames, (
                "Required column variable "
                + var
                + " not found in used variable container"
            )

            assert not varCont[var].isScalar, (
                "Error: Required column variable " + var + " is a scalar"
            )

            if varCont[var].isOnDualGrid is not colVarOnDual:
                warnings.warn(
                    "Variable "
                    + var
                    + " appears in required column variables for implicit variable on "
                    + ("dual" if colVarOnDual else "regular")
                    + " grid but doesn't live on that grid - ignore if implicit variable is a distribution"
                )

        if len(self.__reqMBColVars___) or len(self.__reqMBRowVars___):
            assert (
                mbData is not None
            ), "No modelbound data available when modelbound variables required by a term"

            for var in self.__reqMBRowVars___:
                assert var in mbData.varNames, (
                    "Required row variable "
                    + var
                    + " not found in used modelbound data"
                )

                if not mbData[var].isScalar:
                    if mbData[var].isOnDualGrid is not rowVarOnDual:
                        warnings.warn(
                            "Variable "
                            + var
                            + " appears in modelbound required row variables for evolved variable on "
                            + ("dual" if rowVarOnDual else "regular")
                            + " grid but doesn't live on that grid - ignore if evolved variable is a distribution"
                        )

            for var in self.__reqMBColVars___:
                assert var in mbData.varNames, (
                    "Required column variable " + var + " not found in used modelbound"
                )

                assert not mbData[var].isScalar, (
                    "Error: Required modelbound column variable " + var + " is a scalar"
                )

                if mbData[var].isOnDualGrid is not colVarOnDual:
                    warnings.warn(
                        "Variable "
                        + var
                        + " appears in required modelbound column variables for implicit variable on "
                        + ("dual" if colVarOnDual else "regular")
                        + " grid but doesn't live on that grid - ignore if implicit variable is a distribution"
                    )

    def dict(self):

        varData = {
            "requiredRowVarNames": self.__reqRowVars___,
            "requiredRowVarPowers": self.__reqRowPowers___,
            "requiredColVarNames": self.__reqColVars___,
            "requiredColVarPowers": self.__reqColPowers___,
            "requiredMBRowVarNames": self.__reqMBRowVars___,
            "requiredMBRowVarPowers": self.__reqMBRowPowers___,
            "requiredMBColVarNames": self.__reqMBColVars___,
            "requiredMBColVarPowers": self.__reqMBColPowers___,
        }

        return varData


# TODO: rework this class
class TimeSignalData:
    """Container for matrix term time dependence options"""

    def __init__(
        self,
        signalType="none",
        period=0.0,
        params: Optional[List[float]] = None,
        realPeriod=False,
    ) -> None:
        """TimeSignalData constructor

        Args:
            signalType (str, optional): Type of time signal. Defaults to "none".
            period (float, optional): Period of time signal. Defaults to 0.0 and should be specified if type not "none".
            params (List[float], optional): Signal parameters. Defaults to [] and depend on time signal type.
            realPeriod (bool, optional): If true ReMKiT1D will interpret the period units as seconds. Defaults to False.

        Usage:
            Available non-trivial signal types are "hat" and "cutSine", the first representing a box-shaped signal and the second half a sine period with an amplitude of 1. params(1) and params(2) should then represent the points in the period when the signal starts and ends, respectively, as a fraction of the period (e.g. [0.1,0.3] means that the signal starts after one tenth of the period and lasts for two tenths).
        """

        self.__signalType__ = signalType
        self.__period__ = period
        self.__params__: List[float] = params if params is not None else []
        self.__realPeriod__ = realPeriod

    def dict(self):

        tData = {
            "timeSignalType": self.__signalType__,
            "timeSignalPeriod": self.__period__,
            "timeSignalParams": self.__params__,
            "realTimePeriod": self.__realPeriod__,
        }

        return tData

    def latex(self) -> str:
        return "T"


class AbstractStencil(ABC):
    """Abstract stencil base class"""

    @abstractmethod
    def latex(self, arg: MultiplicativeArgument, **kwargs) -> str:
        pass

    @abstractmethod
    def dict(self) -> dict:
        pass

    @abstractmethod
    def __call__(self, *args: MultiplicativeArgument, **kwds) -> Term:
        pass

    def registerDerivs(self, container: Textbook):
        pass


class MatrixTerm(Term):
    """Matrix terms are of the form :math:`LHS=M_{ij}u_j` where the indices correspond to the evolved (row) and implicit (column) variables, and u is the implicit variable. The matrix M has the following form: :math:`M_{ij} = c*X_i*H_i*V_i*T_i*R_i*S_{ij}*C_j`"""

    def __init__(
        self,
        name: str,
        stencil: AbstractStencil,
        evolvedVar: Optional[Variable] = None,
        implicitVar: Optional[Variable] = None,
        profiles: Optional[Dict[str, Profile]] = None,
        R: Optional[MultiplicativeArgument] = None,
        modelboundR: Optional[MultiplicativeArgument] = None,
        C: Optional[MultiplicativeArgument] = None,
        modelboundC: Optional[MultiplicativeArgument] = None,
        T: Optional[TimeSignalData] = None,
        implicitGroups: Optional[List[int]] = None,
        generalGroups: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Matrix terms are of the form :math:`LHS=M_{ij}u_j` where the indices correspond to the evolved (row) and implicit (column) variables, and u is the implicit variable. The matrix M has the following form: :math:`M_{ij} = c*X_i*H_i*V_i*T_i*R_i*S_{ij}*C_j`, where this constructor sets the individual components.

        This constructor intentionally produces incomplete MatrixTerms, and shouldn't be invoked explicitly except in low level code.

        Args:
            name (str): Name of the term
            stencil (AbstractStencil): Stencil - S in the above formula
            evolvedVar (Optional[Variable], optional): Evolved variable of this term. Defaults to None.
            implicitVar (Optional[Variable], optional): Implicit variable - u in the above formula. Defaults to None.
            profiles (Optional[Dict[str, Profile]], optional): Profile dictionary with possible keys in ["X","V","H"], corresponding to the X,V, and H in the above formula. Defaults to None.
            R (Optional[MultiplicativeArgument], optional): Row variables, together with the multiplicative constant c in the above formula. Defaults to None.
            modelboundR (Optional[MultiplicativeArgument], optional): Modelbound row variables in R in the above formula. Defaults to None.
            C (Optional[MultiplicativeArgument], optional): Column variables in the above formula. Defaults to None.
            modelboundC (Optional[MultiplicativeArgument], optional): Modelbound column variable in the above formula. Defaults to None.
            T (Optional[TimeSignalData], optional): Time signal component of the matrix term. Defaults to None.
            implicitGroups (Optional[List[int]], optional): Implicit groups this term belongs to. Defaults to None, resulting in [1].
            generalGroups (Optional[List[int]], optional): General groups this term belongs to. Defaults to None.

        kwargs:

            skipPattern (bool): If true, assumes that the same stencil acting on the same implicit variable and with the same evolved variable has been used elsewhere. This is an optimisation flag to speed up the setup of large kinetic stencils. Defaults to False.

            fixedMatrix (bool): If true will assume that the matrix is constant and that it doesn't need to be updated during the run. This is an optimisation flag useful with large constant matrices. Defaults to False.

            copyTermName (Optional[str]): Name of term whose matrix is to be copied and multiplied element-wise with this term's stencil. They must have the shame sparsity pattern, i.e. the same stencil shape. Defaults to None.

            evaluatedTermGroup (int): Term group in parent model to be optionally evaluated as additional row variable (multiplying R in the above formula). Defaults to 0, not evaluating any group.

            constLatex (Optional[str]): LaTeX representation of the multiplicative scalar component of the term. Defaults to None, using the numerical value.
        """
        super().__init__(
            name,
            evolvedVar,
            implicitGroups if implicitGroups is not None else [1],
            generalGroups,
        )

        self.__stencil__ = stencil

        self.__implicitVar__ = implicitVar

        self.__profiles__: Dict[str, Profile] = profiles if profiles is not None else {}

        assert all(
            k in ["X", "H", "V"] for k in self.__profiles__.keys()
        ), "Profiles in MatrixTerm constructor must have keys X,H, or V"

        self.__R__ = R if R is not None else MultiplicativeArgument()
        self.__C__ = C if C is not None else MultiplicativeArgument()
        self.__modelboundR__ = (
            modelboundR if modelboundR is not None else MultiplicativeArgument()
        )
        self.__modelboundC__ = (
            modelboundC if modelboundC is not None else MultiplicativeArgument()
        )
        self.__T__: Union[TimeSignalData, None] = T
        self.__skipPattern__ = kwargs.get("skipPattern", False)
        self.__fixedMatrix__ = kwargs.get("fixedMatrix", False)
        self.__copyTermName__: Optional[str] = kwargs.get("copyTermName", None)
        self.__evaluatedTermGroup__: int = kwargs.get("evaluatedTermGroup", 0)
        self.__constLatex__: Optional[str] = kwargs.get("constLatex", None)

    @property
    def implicitVar(self):
        return (
            self.__implicitVar__
            if self.__implicitVar__ is not None
            else self.evolvedVar
        )

    @property
    def stencil(self):
        return self.__stencil__

    @property
    def constLatex(self):
        """LaTeX representation of the multiplicative constant. If None, used the numerical value."""
        return self.__constLatex__

    @constLatex.setter
    def constLatex(self, expression: str):
        self.__constLatex__ = expression

    @property
    def fixedMatrix(self):
        """Return true if this term has a fixed matrix - i.e. it does not evolve"""
        return self.__fixedMatrix__

    def withFixedMatrix(self, fixed=True):
        """Return a copy of this term with fixed/variable matrix

        Args:
            fixed (bool, optional): Set to true if the new term should have its matrix fixed. Defaults to True.
        """
        newTerm = deepcopy(self)
        newTerm.__fixedMatrix__ = fixed
        return newTerm

    @property
    def skipPattern(self):
        """Set to true if a term with the same evolved and implicit variables as well as the stencil has already been added and this term should be skipped when calculating the sparsity pattern. This is an optimisation option for large kinetic simulations."""
        return self.__skipPattern__

    def withSkippingPattern(self, skip=True):
        """Return a copy of this term with the sparsity pattern addition skipped/included

        Args:
            skip (bool, optional): True if the sparsity pattern addition should be skipped. Defaults to True.
        """
        newTerm = deepcopy(self)
        newTerm.__skipPattern__ = skip
        return newTerm

    @property
    def copyTermName(self):
        """Name of term whose matrix is to be copied and multiplied element-wise with this term's stencil. They must have the shame sparsity pattern, i.e. the same stencil shape."""
        return self.__copyTermName__

    @copyTermName.setter
    def copyTermName(self, name: str):
        self.__copyTermName__ = name

    @property
    def evaluatedTermGroup(self):
        """evaluatedTermGroup (int): Term group in parent model to be optionally evaluated as additional row variable (multiplying R in the above formula). Defaults to 0, not evaluating any group."""
        return self.__evaluatedTermGroup__

    @evaluatedTermGroup.setter
    def evaluatedTermGroup(self, group: int):
        self.__evaluatedTermGroup__ = group

    @property
    def multConst(self):
        """Term multiplicative scalar constant"""
        return self.__R__.scalar * self.__modelboundR__.scalar

    def checkTerm(
        self, varCont: VariableContainer, mbData: Optional[ModelboundData] = None
    ):
        """Perform consistency check on term

        Args:
            varCont (VariableContainer): Variable container to be used with this term
            mbData (Optional[ModelboundData]): Modelbound data to check modelbound row and column vars. Defaults to None.
        """

        super().checkTerm(varCont)

        assert self.__implicitVar__ is not None, "MatrixTerm implicitVar not set"

        rowVarOnDual = self.evolvedVar.isOnDualGrid

        assert self.__implicitVar__.name in [v.name for v in varCont.implicitVars], (
            "Implicit variable "
            + self.__implicitVar__.name
            + " not registered in used variable container"
        )

        colVarOnDual = self.__implicitVar__.isOnDualGrid

        vData = VarData(
            self.__R__, self.__C__, self.__modelboundR__, self.__modelboundC__
        )

        vData.checkRowColVars(varCont, rowVarOnDual, colVarOnDual, mbData)

    def dict(self):

        assert (
            self.evolvedVar is not None
        ), "Called dict() on MatrixTerm without setting evolved variable"
        assert (
            self.__implicitVar__ is not None
        ), "Called dict() on MatrixTerm without setting implicit variable"

        gTerm = {
            "termType": "matrixTerm",
            "evolvedVar": self.evolvedVar.name,
            "implicitVar": self.__implicitVar__.name,
            "spatialProfile": (
                self.__profiles__["X"].data.tolist() if "X" in self.__profiles__ else []
            ),
            "harmonicProfile": (
                self.__profiles__["H"].data.tolist() if "H" in self.__profiles__ else []
            ),
            "velocityProfile": (
                self.__profiles__["V"].data.tolist() if "V" in self.__profiles__ else []
            ),
            "evaluatedTermGroup": self.__evaluatedTermGroup__,
            "implicitGroups": self.implicitGroups,
            "generalGroups": self.generalGroups,
            "customNormConst": {"multConst": self.multConst},
            "timeSignalData": (
                self.__T__.dict() if self.__T__ is not None else TimeSignalData().dict()
            ),
            "varData": VarData(
                self.__R__, self.__C__, self.__modelboundR__, self.__modelboundC__
            ).dict(),
            "stencilData": self.__stencil__.dict(),
            "skipPattern": self.__skipPattern__,
            "fixedMatrix": self.__fixedMatrix__,
        }

        if self.__copyTermName__ is not None:
            gTerm["multCopyTermName"] = self.__copyTermName__

        return gTerm

    def latex(self, *args: str, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        result = " " + self.__stencil__.latex(
            self.implicitVar * self.__C__ * self.__modelboundC__, latexRemap=latexRemap
        )
        result = " " + self.__modelboundR__.latex(latexRemap) + result
        result = " " + self.__R__.latex(latexRemap) + result
        for key in self.__profiles__:
            result = " " + self.__profiles__[key].latex() + result
        if self.__T__ is not None:
            result = " " + self.__T__.latex() + result
        constRepr = numToScientificTex(self.multConst, removeUnity=True)
        result = (
            " " + (constRepr if self.constLatex is None else self.constLatex) + result
        )
        return result

    def __neg__(self):
        newMat = deepcopy(self)
        newMat.__R__ *= -1
        return newMat

    def __rmul__(
        self,
        lhs: Union[
            float, int, Profile, TimeSignalData, Variable, MultiplicativeArgument
        ],
    ) -> Self:
        if isinstance(lhs, (int, float)):
            newMat = deepcopy(self)
            newMat.__R__ *= lhs
        if isinstance(lhs, Profile):
            newMat = deepcopy(self)
            if lhs.dim not in newMat.__profiles__:
                newMat.__profiles__[lhs.dim] = lhs
            else:
                newMat.__profiles__[lhs.dim] = Profile(
                    lhs.data * newMat.__profiles__[lhs.dim].data,
                    lhs.dim,
                    latexName=lhs.latex() + newMat.__profiles__[lhs.dim].latex(),
                )
        if isinstance(lhs, TimeSignalData):
            assert (
                self.__T__ is None
            ), "Cannot multiply MatrixTerm that already has explicit time dependece by a TimeSignal"
            newMat = deepcopy(self)
            newMat.__T__ = lhs
        if isinstance(lhs, (Variable, MultiplicativeArgument)):
            newMat = deepcopy(self)
            newMat.__R__ = lhs * newMat.__R__

        return newMat

    def __matmul__(self, rhs: Union[Variable, MultiplicativeArgument]) -> Self:
        if isinstance(rhs, (Variable, MultiplicativeArgument)):
            newMat = deepcopy(self)
            newMat.__modelboundR__ = rhs * newMat.__modelboundR__

        return newMat

    def __add__(self, rhs: Term) -> TermCollection:
        newCollection = TermCollection(self.evolvedVar, "", "")
        newCollection += self
        newCollection += rhs

        return newCollection

    def __sub__(self, rhs: Term) -> TermCollection:
        newCollection = TermCollection(self.evolvedVar, "", "")
        newCollection += self
        newCollection += -rhs

        return newCollection

    def registerDerivs(self, container: Textbook):
        self.__stencil__.registerDerivs(container)


class Stencil(AbstractStencil):
    """Generic stencil class"""

    def __init__(
        self,
        latexTemplate: Optional[str] = None,
        properties: Optional[Dict[str, object]] = None,
    ):
        """Generic stencil class

        Args:
            latexTemplate (Optional[str], optional): Stencil LaTeX template - must contain $0 - this is where the argument will go. Defaults to None.
            properties (Optional[Dict[str, object]], optional): ReMKiT1D JSON representation of the stencil. Defaults to None.
        """
        super().__init__()
        if latexTemplate is not None:
            assert (
                "$0" in latexTemplate
            ), "Stencil latexTemplate argument must be specified"
        self.__latexTemplate__ = latexTemplate
        self.__properties__ = properties

    def dict(self) -> dict:
        assert (
            self.__properties__ is not None
        ), "Stencil default dict properties not provided"
        return self.__properties__

    def __call__(
        self, *args: Union[MultiplicativeArgument, Variable], **kwargs
    ) -> MatrixTerm:

        assert (
            len(args) == 1 or len(args) == 2
        ), "Stencil __call__ must have 1 or 2 args"
        C = (
            args[0]
            if isinstance(args[0], MultiplicativeArgument)
            else MultiplicativeArgument((args[0], 1.0))
        )
        modelboundC = (
            MultiplicativeArgument()
            if len(args) == 1
            else (
                args[1]
                if isinstance(args[1], MultiplicativeArgument)
                else MultiplicativeArgument((args[1], 1.0))
            )
        )
        assert (
            len(C.args) > 0
        ), "First arg in Stencil call must have at least one variable"
        for arg in args:
            if isinstance(arg, MultiplicativeArgument):
                assert (
                    arg.scalar == 1
                ), "Stencils must act on MultiplicativeArguments without a non-trivial scalar component"
        return MatrixTerm(
            "unnamed_term",
            self,
            implicitVar=C.firstArg,
            C=C / C.firstArg,
            modelboundC=modelboundC,
        )

    def latex(self, arg: MultiplicativeArgument, **kwargs):
        assert (
            self.__latexTemplate__ is not None
        ), "latex() on Stencil called without a default latexTemplate set"
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        return self.__latexTemplate__.replace("$0", arg.latex(latexRemap))


class DiagonalStencil(Stencil):
    """Diagonal stencil allowing for evolving only specific grid points. If row and column variables are on different grids this stencil will perform linear interpolation/extrapolation."""

    def __init__(
        self,
        evolvedXCells: Optional[List[int]] = None,
        evolvedHarmonics: Optional[List[int]] = None,
        evolvedVCells: Optional[List[int]] = None,
    ):
        """Diagonal stencil allowing for evolving only specific grid points. If row and column variables are on different grids this stencil will perform linear interpolation/extrapolation.

        Args:
            evolvedXCells (Optional[List[int]], optional): List of evolved spatial cells (Fortran 1-indexing). Defaults to None, evolving all cells.
            evolvedHarmonics (Optional[List[int]], optional): List of evolved harmonics (Fortran 1-indexing). Defaults to None, evolving all harmonics.
            evolvedVCells (Optional[List[int]], optional): List of evolved velocity cells. Defaults to None, evolving all cells.
        """
        properties: Dict[str, object] = {
            "stencilType": "diagonalStencil",
            "evolvedXCells": evolvedXCells if evolvedXCells is not None else [],
            "evolvedHarmonics": (
                evolvedHarmonics if evolvedHarmonics is not None else []
            ),
            "evolvedVCells": evolvedVCells if evolvedVCells is not None else [],
        }
        super().__init__("$0", properties)


class DerivationTerm(Term):
    """Derivation-based explicit term. The result of evaluating a derivation term is the result of the derivation optionally multiplied by a modelbound variable. Does not support evolving distributions."""

    def __init__(
        self,
        name: str,
        closure: DerivationClosure,
        evolvedVar: Optional[Variable] = None,
        mbVar: Optional[Variable] = None,
        generalGroups: Optional[List[int]] = None,
    ) -> None:
        """Derivation term

        Args:
            evolvedVar (str): Name of the evolved variable. Distributions not supported.
            derivationRule (dict): Derivation rule containing name and required variables.
            mbVar (Union[str,None], optional): Optional modelbound variable. Defaults to None.
            generalGroups (list, optional): General groups this term belongs to within its model. Defaults to [1].
        """

        assert closure.numArgs == 0, "DerivationTerm requires complete closure"
        super().__init__(
            name, evolvedVar, [], generalGroups if generalGroups is not None else [1]
        )

        self.__derivation__ = closure
        self.__mbVar__ = mbVar

    def checkTerm(
        self, varCont: VariableContainer, mbData: Optional[ModelboundData] = None
    ):
        """Perform consistency check on term, including the required variables

        Args:
            varCont (VariableContainer): Variable container to be used with this term
        """

        super().checkTerm(varCont)

        derivArgs = self.__derivation__.fillArgs()
        for name in derivArgs:
            assert name in varCont.varNames, (
                "Required derivation variable "
                + name
                + " not registered in used variable container"
            )

        if self.__mbVar__ is not None:
            assert (
                mbData is not None
            ), "Modelbound variable present in derivation term when no mbData passed"
            assert self.__mbVar__.name in mbData.varNames, (
                "Variable " + self.__mbVar__.name + " not in modelbound data"
            )

            if self.__mbVar__.isOnDualGrid is not self.evolvedVar.isOnDualGrid:
                warnings.warn(
                    "Variable "
                    + self.__mbVar__.name
                    + " appears in required row variables for evolved variable on "
                    + ("dual" if self.evolvedVar.isOnDualGrid else "regular")
                    + " grid but doesn't live on that grid"
                )

    def dict(self):

        gTerm = {
            "termType": "derivationTerm",
            "evolvedVar": self.evolvedVar.name,
            "generalGroups": self.generalGroups,
        }

        if self.__mbVar__ is not None:
            gTerm["requiredMBVarName"] = self.__mbVar__.name

        gTerm.update(
            {
                "ruleName": self.__derivation__.name,
                "requiredVarNames": self.__derivation__.fillArgs(),
            }
        )

        return gTerm

    def latex(self, *args: str, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        remappedArgs = (
            (
                latexRemap[arg]
                if arg in latexRemap
                else "\\text{" + arg.replace("_", r"\_") + "}"
            )
            for arg in self.__derivation__.fillArgs()
        )
        result = " " + self.__derivation__.latex(*remappedArgs)
        if self.__mbVar__ is not None:
            result = (
                latexRemap[self.__mbVar__.name]
                if self.__mbVar__.name in latexRemap
                else "\\text{" + self.__mbVar__.name.replace("_", r"\_") + "} " + result
            )
        return result

    def __add__(self, rhs: Term) -> TermCollection:
        newCollection = TermCollection(self.evolvedVar, "", "")
        newCollection += self
        newCollection += rhs

        return newCollection

    def registerDerivs(self, container: Textbook):
        container.register(self.__derivation__, ignoreDuplicates=True)


class TermGenerator(ABC):
    """Abstract term generator class"""

    def __init__(
        self,
        name: str,
        implicitGroups: Optional[List[int]] = None,
        generalGroups: Optional[List[int]] = None,
    ) -> None:
        super().__init__
        self.__name__ = name
        self.__implicitGroups__ = implicitGroups if implicitGroups is not None else [1]
        self.__generalGroups__ = generalGroups if generalGroups is not None else []

    @property
    def name(self):
        return self.__name__

    @property
    def implicitGroups(self) -> List[int]:
        return self.__implicitGroups__

    @property
    def generalGroups(self) -> List[int]:
        return self.__generalGroups__

    @abstractmethod
    def dict(self) -> dict:

        tgDict: Dict[str, object] = {
            "implicitGroups": self.__implicitGroups__,
            "generalGroups": self.__generalGroups__,
        }

        return tgDict

    @abstractmethod
    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        pass

    @property
    @abstractmethod
    def evolvedVars(self) -> List[str]:
        pass

    @abstractmethod
    def onlyEvolving(self, *args: Variable) -> Self:
        pass


class DDT:
    """Wrapper for term collections in models, providing indexing based on variables"""

    def __init__(self, modelName: str, modelLatexName: str):
        self.__termCollections__: List[TermCollection] = []
        self.__modelName__ = modelName
        self.__modelLatexName__ = modelLatexName

    def __getitem__(self, var: Variable):
        if var.name not in self.evolvedVars:
            self.__termCollections__.append(
                TermCollection(var, self.__modelName__, self.__modelLatexName__)
            )
            return self.__getitem__(var)
        ind = self.evolvedVars.index(var.name)
        return self.__termCollections__[ind]

    def __setitem__(self, var: Variable, tc: TermCollection):
        if var.name not in self.evolvedVars:
            self.__termCollections__.append(
                TermCollection(var, self.__modelName__, self.__modelLatexName__)
            )
            self.__setitem__(var, tc)
            return
        ind = self.evolvedVars.index(var.name)
        self.__termCollections__[ind] = tc

    def __delitem__(self, var: Variable):
        if var.name not in self.evolvedVars:
            raise KeyError()
        del self.__termCollections__[self.evolvedVars.index(var.name)]

    @property
    def evolvedVars(self):
        return [tc.evolvedVar.name for tc in self.__termCollections__]

    def registerDerivs(self, container: Textbook):
        for tc in self.__termCollections__:
            tc.registerDerivs(container)

    def filterByGroup(
        self, groups: Optional[List[int]] = None, general: bool = False
    ) -> Self:
        """Filter all terms by group and return new DDT object

        Args:
            group (Optional[List[int]]): Group indices (Fortran 1-indexing). Defaults to [1]
            general (bool, optional): If true, will filter with respect to general groups, otherwise filters with respect to implicit groups. Defaults to False.
        """

        newDDT = DDT(self.__modelName__, self.__modelLatexName__)

        for tc in self.__termCollections__:
            if len(tc.filterByGroup(groups, general).terms):
                newDDT.__termCollections__.append(tc.filterByGroup(groups, general))

        return cast(Self, newDDT)


class Model:
    """A container for terms, term generators, and modelbound data"""

    def __init__(
        self, name: str, latexName: Optional[str] = None, isIntegrable=True
    ) -> None:
        """A container for terms, term generators, and modelbound data

        Args:
            name (str): Name of the model
            latexName (Optional[str], optional): Optional LaTeX-compatible string for the model name. Defaults to None.
            isIntegrable (bool, optional): Optional flag for integrable models. Defaults to True.
        """
        self.__name__ = name
        self.__latexName__ = (
            latexName
            if latexName is not None
            else "\\text{" + name.replace("_", r"\_") + "}"
        )
        self.__modelboundData__: Optional[ModelboundData] = None
        self.__termGenerators__: List[TermGenerator] = []
        self.__isIntegrable__ = isIntegrable

        self.ddt = DDT(name, self.__latexName__)

    @property
    def name(self):
        return self.__name__

    def rename(self, name: str, latexName: Optional[str] = None):
        newModel = deepcopy(self)
        newModel.__name__ = name
        if latexName is not None:
            newModel.__latexName__ = latexName
        return newModel

    @property
    def latexName(self):
        return self.__latexName__

    @property
    def evolvedVars(self):
        evolvedVars = self.ddt.evolvedVars
        for tg in self.__termGenerators__:
            evolvedVars += tg.evolvedVars

        return list(set(evolvedVars))

    @property
    def mbData(self):
        return self.__modelboundData__

    @property
    def activeImplicitGroups(self):
        activeGroups = sum(
            [tc.activeImplicitGroups for tc in self.ddt.__termCollections__], []
        )
        activeGroups += sum(
            [tg.implicitGroups for tg in self.__termGenerators__],
            [],
        )

        return list(set(activeGroups))

    @property
    def activeGeneralGroups(self):
        activeGroups = sum(
            [tc.activeGeneralGroups for tc in self.ddt.__termCollections__], []
        )
        activeGroups += sum(
            [tg.generalGroups for tg in self.__termGenerators__],
            [],
        )

        return list(set(activeGroups))

    @property
    def isIntegrable(self):
        return self.__isIntegrable__

    @isIntegrable.setter
    def isIntegrable(self, integrable: bool):
        self.__isIntegrable__ = integrable

    def onlyEvolving(self, *args: Variable) -> Self:
        """Produce a new model only evolving a given list of variables"""
        newModel = Model(self.name, self.__latexName__, self.isIntegrable)

        for arg in args:
            if arg.name in self.evolvedVars:
                newModel.ddt[arg] = self.ddt[arg]

        argNames = [arg.name for arg in args]
        for tg in self.__termGenerators__:
            if any(name in tg.evolvedVars for name in argNames):
                newModel.addTermGenerator(tg.onlyEvolving(*args))

        newModel.setModelboundData(self.mbData)

        return cast(Self, newModel)

    def filterByGroup(
        self, groups: Optional[List[int]] = None, general: bool = False
    ) -> Self:
        """Filter all terms and term generators by group and return new Model

        Args:
            group (Optional[List[int]]): Group indices (Fortran 1-indexing). Defaults to [1]
            general (bool, optional): If true, will filter with respect to general groups, otherwise filters with respect to implicit groups. Defaults to False.
        """

        newModel = Model(self.name, self.__latexName__, self.isIntegrable)

        newModel.ddt = self.ddt.filterByGroup(groups, general)
        usedGroups = groups if groups is not None else [1]

        for tg in self.__termGenerators__:
            if general:
                if any(group in tg.generalGroups for group in usedGroups):
                    newModel.addTermGenerator(tg)
            else:
                if any(group in tg.implicitGroups for group in usedGroups):
                    newModel.addTermGenerator(tg)

        return cast(Self, newModel)

    def addTerm(self, termTag: str, term: Term):
        assert (
            term.evolvedVar is not None
        ), "Cannot add Term without evolved variable directly"

        self.ddt[term.evolvedVar] += term.rename(termTag)

    def addTermGenerator(self, generator: TermGenerator):
        assert generator.name not in [tg.name for tg in self.__termGenerators__], (
            "Term generator with name "
            + generator.name
            + " already in model "
            + self.name
        )

        self.__termGenerators__.append(generator)

    def setModelboundData(self, mbData: ModelboundData):
        self.__modelboundData__ = mbData

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        with doc.create(tex.Subsection(tex.NoEscape("$" + self.__latexName__ + "$"))):
            with doc.create(tex.Subsubsection("Equation contributions")):
                for tc in self.ddt.__termCollections__:
                    if len(tc.termNames):
                        with doc.create(
                            tex.Alignat(numbering=False, escape=False)
                        ) as agn:
                            tc.addLatexToDoc(agn, latexRemap)

            if self.__modelboundData__ is not None:
                with doc.create(tex.Subsubsection("Modelbound data")):
                    self.__modelboundData__.addLatexToDoc(doc, **kwargs)

            if len(self.__termGenerators__):
                with doc.create(tex.Subsubsection("Term generators")):
                    for tg in self.__termGenerators__:
                        tg.addLatexToDoc(doc, **kwargs)

    def checkTerms(self, varCont: VariableContainer):
        """Check terms in this model for consistency

        Args:
            varCont (VariableContainer): Variable container to be used in this check
        """
        print("Checking terms in model " + self.name + ":")
        for tc in self.ddt.__termCollections__:
            tc.checkTerms(varCont, self.mbData)

    def dict(self):

        cModel = {
            "type": "customModel",
            "termTags": sum([tc.termNames for tc in self.ddt.__termCollections__], []),
            "termGenerators": {"tags": [tg.name for tg in self.__termGenerators__]},
        }
        if self.__modelboundData__ is not None:
            cModel["modelboundData"] = self.__modelboundData__.dict()
        cModel["termGenerators"].update(
            {tg.name: tg.dict() for tg in self.__termGenerators__}
        )
        for tc in self.ddt.__termCollections__:
            cModel.update(tc.dict())

        return cModel

    def registerDerivs(self, container: Textbook):
        self.ddt.registerDerivs(container)
        if self.mbData is not None:
            cast(ModelboundData, self.__modelboundData__).registerDerivs(container)


class ModelCollection:
    """Container object for models"""

    def __init__(self, *args: Model):
        self.__models__: List[Model] = list(args)

    @property
    def modelNames(self):
        return [model.name for model in self.__models__]

    @property
    def models(self):
        return self.__models__

    def __getitem__(self, key: str):
        if key not in self.modelNames:
            raise KeyError()
        return self.models[self.modelNames.index(key)]

    def __setitem__(self, key: str, model: Model):
        if key not in self.modelNames:
            self.models.append(model)
        else:
            self.models[self.modelNames.index(key)] = model

    def __delitem__(self, key: str):
        if key not in self.modelNames:
            raise KeyError()
        del self.models[self.modelNames.index(key)]

    def add(self, *args: Model):
        for arg in args:
            assert (
                arg.name not in self.modelNames
            ), "Attempted to add duplicate model to ModelCollection - use key access if you wish to overwrite an existing model"

        self.__models__ += list(args)

    def dict(self) -> dict:

        modelDict = {"tags": [m.name for m in self.__models__]}

        for model in self.__models__:
            modelDict.update({model.name: model.dict()})

        return modelDict

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        with doc.create(tex.Section("Models")):
            for model in self.__models__:
                model.addLatexToDoc(doc, **kwargs)

    def onlyEvolving(self, *args: Variable) -> Self:
        """Produce a model collection from this one only evolving a given list of variables"""
        newCollection = ModelCollection()

        for model in self.__models__:
            if any(arg.name in model.evolvedVars for arg in args):
                newCollection.add(model.onlyEvolving(*args))

        return cast(Self, newCollection)

    def filterByGroup(
        self, groups: Optional[List[int]] = None, general: bool = False
    ) -> Self:
        """Filter all models by term group and return new collection

        Args:
            group (Optional[List[int]]): Group indices (Fortran 1-indexing). Defaults to [1]
            general (bool, optional): If true, will filter with respect to general groups, otherwise filters with respect to implicit groups. Defaults to False.
        """

        newCollection = ModelCollection()

        for model in self.__models__:
            if len(model.filterByGroup(groups, general).evolvedVars):
                newCollection.add(model.filterByGroup(groups, general))

        return cast(Self, newCollection)

    def checkModels(self, varCont: VariableContainer):
        for model in self.__models__:
            model.checkTerms(varCont)

    def numGroups(self) -> Tuple[int, int]:
        """Calculate the total number of implicit and general groups needed by this model collection"""
        numImplicitGroups = max(
            max([1] + model.activeImplicitGroups) for model in self.__models__
        )
        numGeneralGroups = max(
            max([1] + model.activeGeneralGroups) for model in self.__models__
        )

        return numImplicitGroups, numGeneralGroups

    def getTermsThatEvolveVar(self, var: Variable) -> List[Tuple[str, str]]:
        """Get a list of (model,term) tuples containing the model and term names evolving a given variable

        Args:
            var (Variable): Evolved variable to get tuples for
        """
        result = []
        for model in self.models:
            if var.name in model.evolvedVars:
                for term in model.ddt[var].termNames:
                    result.append((model.name, term))

        return result

    def registerDerivs(self, container: Textbook):
        for model in self.models:
            model.registerDerivs(container)


class VarlikeModelboundData(ModelboundData):
    """Variable-like modelbound data"""

    def __init__(self) -> None:
        self.__variables__: List[Variable] = []

    @property
    def varNames(self):
        return [v.name for v in self.__variables__]

    def __getitem__(self, key: str):
        if key not in self.varNames:
            raise KeyError()
        return self.__variables__[self.varNames.index(key)]

    def addVar(
        self,
        *args: Variable,
    ) -> None:

        for var in args:
            assert var.derivation is not None, (
                "Variable "
                + var.name
                + " does not have a derivation associated to it - cannot add to VarlikeModelboundData"
            )

            if var.dims is not None:
                assert "t" not in cast(List[str], var.dims), (
                    "Variable "
                    + var.name
                    + " has time dimension - cannot add to VarlikeModelboundData"
                )

            assert var.name not in self.varNames, (
                "Variable " + var.name + " already in VarlikeModelboundData"
            )

            self.__variables__.append(var)

    def dict(self) -> dict:
        properties = {
            "modelboundDataType": "varlikeData",
            "dataNames": self.varNames,
        }

        for var in self.__variables__:
            isDerivedFromOtherData = False
            if any(arg in self.varNames for arg in var.derivationArgs):
                assert all(arg in self.varNames for arg in var.derivationArgs), (
                    "Modelbound variable "
                    + var.name
                    + " has some derivationArgs in the containing modelbound data, but not all. If the variable was not meant to depend on other modelbound data consider renaming some of the variables in the modelbound data object."
                )
                isDerivedFromOtherData = True
            properties[var.name] = {
                "isDistribution": var.isDistribution,
                "isScalar": var.isScalar,
                "isSingleHarmonic": var.isSingleHarmonic,
                "isDerivedFromOtherData": isDerivedFromOtherData,
                "derivationPriority": var.priority,
                "ruleName": var.derivation.name,
                "requiredVarNames": var.derivationArgs,
            }
        return properties

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        with doc.create(tex.Itemize()) as itemize:
            for var in self.__variables__:
                itemize.add_item(tex.NoEscape(f"${var.latex(latexRemap)}$"))

    def registerDerivs(self, container: Textbook):
        for var in self.__variables__:
            container.register(var.derivation, ignoreDuplicates=True)


class LBCModelboundData(ModelboundData):
    """Modelbound data needed for the Logical BC for kinetic electrons"""

    def __init__(
        self,
        grid: Grid,
        ionCurrent: Variable,
        distFun: Variable,
        density: Variable,
        densityDual: Optional[Variable] = None,
        densityOnBoundary: Optional[Variable] = None,
        totalCurrent: Optional[Variable] = None,
        bisTol: float = 1e-12,
        leftBoundary=False,
    ):
        """Modelbound data needed for the logical boundary condition for the electrons in kinetic mode. Matches the electron current with ion current, potentially taking into account a target total current as well.

        Args:
            grid (Grid): Used grid object
            ionCurrent (Variable): Ion current the electrons need to match (up to totalCurrent values, see below)- scalar variable
            distFun (Variable): Electron distribution variable
            density (Variable): Electron density variable
            densityDual (Optional[Variable], optional): Dual electron density variable - needed when using staggered grids. Defaults to None.
            densityOnBoundary (Optional[Variable], optional): Value of the electron density on the boundary - if not present will only extrapolate the distribution to the cell centre closes to the boundary, assuming it stays constant from that point to the boundary - scalar variable. Defaults to None.
            totalCurrent (Optional[Variable], optional): Total current variable - scalar. Defaults to None, effectively setting the total current to 0.
            bisTol (float, optional): Bisection tolerance for the solver trying to match the electron flux to the ion flux. Defaults to 1e-12.
            leftBoundary (bool, optional): True if on left boundary. Defaults to False.
        """
        assert ionCurrent.isScalar, "ionCurrent in LBCModelboundData must be a scalar"
        self.__ionCurrent__ = ionCurrent

        assert (
            distFun.isDistribution
        ), "distFun in LBCModelboundData must be a distribution"

        assert density.isFluid, "density must be a fluid variable in LBCModelboundData"

        if densityDual is not None:
            assert (
                densityDual.isOnDualGrid
            ), "densityDual must be on the dual grid in LBCModelboundData"
            assert (
                densityDual.isFluid
            ), "densityDual must be a fluid variable in LBCModelboundData"

        if densityOnBoundary is not None:
            assert (
                densityOnBoundary.isScalar
            ), "densityOnBoundary must be a scalar variable in LBCModelboundData"

        if totalCurrent is not None:
            assert (
                totalCurrent.isScalar
            ), "totalCurrent must be a scalar variable in LBCModelboundData"
        self.__totalCurrent__ = totalCurrent

        self.__bisTol__ = bisTol
        self.__leftBoundary__ = leftBoundary

        self.__deriv__ = GenericDerivation(
            ("left" if leftBoundary else "right") + "DistExt",
            0,
            {
                "type": "distScalingExtrapDerivation",
                "extrapolateToBoundary": isinstance(densityOnBoundary, Variable),
                "staggeredVars": isinstance(densityDual, Variable),
                "leftBoundary": leftBoundary,
            },
            "\\text{DistExt}_" + "L" if leftBoundary else "R",
        )
        self.__reqVars__ = [distFun.name, density.name]
        if densityDual is not None:
            self.__reqVars__.append(densityDual.name)
        if densityOnBoundary is not None:
            self.__reqVars__.append(densityOnBoundary.name)

        self.__grid__ = grid

    def dict(self):
        mbData = {
            "modelboundDataType": "modelboundLBCData",
            "ionCurrentVarName": self.__ionCurrent__.name,
            "totalCurrentVarName": (
                self.__totalCurrent__.name
                if self.__totalCurrent__ is not None
                else "none"
            ),
            "bisectionTolerance": self.__bisTol__,
            "leftBoundary": self.__leftBoundary__,
        }
        mbData.update(
            {"ruleName": self.__deriv__.name, "requiredVarNames": self.__reqVars__}
        )

        return mbData

    @property
    def varNames(self):
        return ["gamma", "potential", "coVel", "shTemp"]

    def __getitem__(self, key):
        if key not in self.varNames:
            raise KeyError()
        return Variable(key, self.__grid__, isDerived=True, isScalar=True)

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        doc.append(
            "Logical boundary condition data on "
            + ("left" if self.__leftBoundary__ else "right")
            + " boundary"
        )
        with doc.create(tex.Itemize()) as itemize:
            for var in self.varNames:
                itemize.add_item(tex.NoEscape(f"${self[var].latex(latexRemap)}$"))

    def registerDerivs(self, container: Textbook):
        container.register(self.__deriv__, ignoreDuplicates=True)
