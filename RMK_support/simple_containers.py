from typing import Union, List, Dict, cast, Tuple
from .variable_container import VariableContainer
import numpy as np
import warnings
from abc import ABC, abstractmethod


class Term(ABC):

    def __init__(
        self,
        evolvedVar: str,
        implicitGroups: List[int] = [],
        generalGroups: List[int] = [],
    ) -> None:
        super().__init__()
        self.__evolvedVar__ = evolvedVar
        self.__implicitGroups__ = implicitGroups
        self.__generalGroups__ = generalGroups

    @property
    def evolvedVar(self):
        return self.__evolvedVar__

    @property
    def implicitGroups(self):
        return self.__implicitGroups__

    @property
    def generalGroups(self):
        return self.__generalGroups__

    @abstractmethod
    def dict(self) -> dict:
        pass

    def checkTerm(self, varCont: VariableContainer):

        assert self.__evolvedVar__ in varCont.dataset.data_vars.keys(), (
            "Evolved variable "
            + self.__evolvedVar__
            + " not registered in used variable container"
        )


class Species:
    """Contains species data"""

    def __init__(
        self,
        name: str,
        speciesID: int,
        atomicA: float = 1.0,
        charge: float = 0.0,
        associatedVars: List[str] = [],
    ) -> None:
        """Species initialization.

        Args:
            name (str): Species name
            speciesID (int): Species ID
            atomicA (float): Species mass in amu. Defaults to 1
            charge (float): Species charge in e. Defaults to 0
            associatedVars (List[str], optional): List of variables associated with this species. Defaults to [].
        """
        # Assertions
        assert atomicA > 0, "Zero mass species are not allowed"

        self.__name__ = name
        self.__speciesID__ = speciesID
        self.__atomicA__ = atomicA
        self.__charge__ = charge
        self.__associatedVars__ = associatedVars

    @property
    def name(self):
        return self.__name__

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
    def associatedVars(self):
        return self.__associatedVars__

    def dict(self) -> dict:
        """Return Species objects as dictionary entry for JSON output

        Returns:
            dict: ReMKiT1D-ready dictionary form of species data
        """

        speciesData = {
            "ID": self.speciesID,
            "atomicMass": self.atomicA,
            "charge": self.charge,
            "associatedVars": self.associatedVars,
        }

        return speciesData


class VarData:
    """Contains custom matrix fluid model required row and column variable data"""

    def __init__(
        self,
        reqRowVars: List[str] = [],
        reqRowPowers: List[float] = [],
        reqColVars: List[str] = [],
        reqColPowers: List[float] = [],
        reqMBRowVars: List[str] = [],
        reqMBRowPowers: List[float] = [],
        reqMBColVars: List[str] = [],
        reqMBColPowers: List[float] = [],
    ) -> None:
        """VarData constructor

        Args:
            reqRowVars (List[str], optional): List of required row variables. Defaults to [].
            reqRowPowers (List[float], optional): List of required row variable powers. Defaults to [].
            reqColVars (List[str], optional): List of required column variables. Defaults to [].
            reqColPowers (List[float], optional): List of required column variable powers. Defaults to [].
            reqMBRowVars (List[str], optional): List of required row modelbound data names. Defaults to [].
            reqMBRowPowers (List[float], optional): List of required  row modelbound data powers. Defaults to [].
            reqMBColVars (List[str], optional): List of required column modelbound data names. Defaults to [].
            reqMBColPowers (List[float], optional): List of required column modelbound data powers. Defaults to [].
        """

        # Fill empty power lists with ones
        if len(reqRowVars) > 0 and len(reqRowPowers) == 0:
            reqRowPowers = [1.0] * len(reqRowVars)

        if len(reqColVars) > 0 and len(reqColPowers) == 0:
            reqColPowers = [1.0] * len(reqColVars)

        if len(reqMBRowVars) > 0 and len(reqMBRowPowers) == 0:
            reqMBRowPowers = [1.0] * len(reqMBRowVars)

        if len(reqMBColVars) > 0 and len(reqMBColPowers) == 0:
            reqMBColPowers = [1.0] * len(reqMBColVars)

        # Assertions in case power lists are specified

        assert len(reqRowVars) == len(
            reqRowPowers
        ), "reqRowPowers must have same length as reqRowVars"
        assert len(reqColVars) == len(
            reqColPowers
        ), "reqColPowers must have same length as reqColVars"
        assert len(reqMBRowVars) == len(
            reqMBRowPowers
        ), "reqMBRowPowers must have same length as reqMBRowVars"
        assert len(reqMBColVars) == len(
            reqMBColPowers
        ), "reqMBColPowers must have same length as reqMBColVars"

        self.__reqRowVars___ = reqRowVars
        self.__reqRowPowers___ = reqRowPowers
        self.__reqColVars___ = reqColVars
        self.__reqColPowers___ = reqColPowers
        self.__reqMBRowVars___ = reqMBRowVars
        self.__reqMBRowPowers___ = reqMBRowPowers
        self.__reqMBColVars___ = reqMBColVars
        self.__reqMBColPowers___ = reqMBColPowers

    def checkRowColVars(
        self, varCont: VariableContainer, rowVarOnDual=False, colVarOnDual=False
    ):
        """Check whether required variables exist in the variable container and are on the correct grids

        Args:
            varCont (VariableContainer): Variable container used to check
            rowVarOnDual (bool, optional): True if the row variables should be on the dual grid. Defaults to False.
            colVarOnDual (bool, optional): True if the column variables should be on the dual grid. Defaults to False.
        """

        for var in self.__reqRowVars___:
            assert var in varCont.dataset.data_vars.keys(), (
                "Required row variable " + var + " not found in used variable container"
            )

            if not varCont.dataset.data_vars[var].attrs["isScalar"]:
                if (
                    varCont.dataset.data_vars[var].attrs["isOnDualGrid"]
                    is not rowVarOnDual
                ):
                    warnings.warn(
                        "Variable "
                        + var
                        + " appears in required row variables for evolved variable on "
                        + ("dual" if rowVarOnDual else "regular")
                        + " grid but doesn't live on that grid"
                    )

        for var in self.__reqColVars___:
            assert var in varCont.dataset.data_vars.keys(), (
                "Required column variable "
                + var
                + " not found in used variable container"
            )

            assert not varCont.dataset.data_vars[var].attrs["isScalar"], (
                "Error: Required column variable " + var + " is a scalar"
            )

            if varCont.dataset.data_vars[var].attrs["isOnDualGrid"] is not colVarOnDual:
                warnings.warn(
                    "Variable "
                    + var
                    + " appears in required column variables for implicit variable on "
                    + ("dual" if colVarOnDual else "regular")
                    + " grid but doesn't live on that grid"
                )

    def dict(self):
        """Returns dictionary form of VarData to be used in json output

        Returns:
            dict: Dictionary form of VarData to be added to individual custom term properties
        """
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


class CustomNormConst:
    """Custom normalization constant data"""

    def __init__(
        self, multConst=1.0, normNames: List[str] = [], normPowers: List[float] = []
    ) -> None:
        """CustomNormConst constructor

        Args:
            multConst (float, optional): Multiplicative constant component. Defaults to 1.0.
            normNames (List[str], optional): Names of required normalization values. Defaults to [].
            normPowers (List[float], optional): Powers corresponding to required normalization values. Defaults to [].
        """

        # Fill empty norm power list
        if len(normNames) > 0 and len(normPowers) == 0:
            normPowers = [1.0] * len(normNames)

        # Assertions
        assert len(normNames) == len(
            normPowers
        ), "normPowers must have same length as normNames"

        self.__multConst__ = multConst
        self.__normNames__ = normNames
        self.__normPowers__ = normPowers

    def dict(self):
        """Returns dictionary form of CustomNormConst to be used in json output

        Returns:
            dict: Dictionary form of CustomNormConst to be added to individual custom term properties
        """
        cNormConst = {
            "multConst": self.__multConst__,
            "normNames": self.__normNames__,
            "normPowers": self.__normPowers__,
        }

        return cNormConst


class TimeSignalData:
    """Container for custom term time dependence options"""

    def __init__(
        self, signalType="none", period=0.0, params: List[float] = [], realPeriod=False
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
        self.__params__ = params
        self.__realPeriod__ = realPeriod

    def dict(self):
        """Returns dictionary form of TimeSignalData to be used in json output

        Returns:
            dict: Dictionary form of TimeSignalData to be added to individual custom term properties
        """
        tData = {
            "timeSignalType": self.__signalType__,
            "timeSignalPeriod": self.__period__,
            "timeSignalParams": self.__params__,
            "realTimePeriod": self.__realPeriod__,
        }

        return tData


def diagonalStencil(
    evolvedXCells: List[int] = [],
    evolvedHarmonics: List[int] = [],
    evolvedVCells: List[int] = [],
) -> dict:
    """Return diagonal stencil properties

    Args:
        evolvedXCells (List[int], optional): List of evolved x cells. Defaults to [], resulting in all allowed cells.
        evolvedHarmonics (List[int], optional): List of evolved harmonics, used only if the row variable is a distribution. Defaults to [], resulting in all harmonics.
        evolvedVCells (List[int], optional): List of evolved velocity cells, used only if the row variable is a distribution. Defaults to [], resulting in all velocity cells.

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "diagonalStencil",
        "evolvedXCells": evolvedXCells,
        "evolvedHarmonics": evolvedHarmonics,
        "evolvedVCells": evolvedVCells,
    }

    return stencil


class GeneralMatrixTerm(Term):
    """General (custom) matrix term options used to construct custom models"""

    def __init__(
        self,
        evolvedVar: str,
        implicitVar="",
        spatialProfile: List[float] = [],
        harmonicProfile: List[float] = [],
        velocityProfile: List[float] = [],
        evaluatedTermGroup=0,
        implicitGroups=[1],
        generalGroups: List[int] = [],
        customNormConst: Union[CustomNormConst, float, int] = CustomNormConst(),
        timeSignalData=TimeSignalData(),
        varData=VarData(),
        stencilData=diagonalStencil(),
        skipPattern=False,
        fixedMatrix=False,
        copyTermName: Union[str, None] = None,
    ) -> None:
        """GeneralMatrixTerm option container constructor.

        General matrix terms are of the form :math:`LHS=M_{ij}u_j` where the indices correspond to the evolved (row) and implicit (column) variables, and u is the implixit variable. The matrix M has the following form: :math:`M_{ij} = c*X_i*H_i*V_i*T_i*R_i*S_{ij}*C_j`, where this constructor sets the individual components.

        Args:
            evolvedVar (str): Name of evolved implicit variable
            implicitVar (str, optional): Name of column implicit variable. Defaults to evolvedVar value.
            spatialProfile (List[float], optional): Spatial profile - X in the above formula (should conform to x-grid size if supplied). Defaults to [].
            harmonicProfile (List[float], optional): Harmonic profile - H in the above formula (should conform to h-grid size if supplied), used only if evolved variable is a distribution. Defaults to [].
            velocityProfile (List[float], optional): Velocity profile - V in the above formula (should conform to v-grid size if supplied), used only if evolved variable is a distribution. Defaults to [].
            evaluatedTermGroup (int, optional): Term group in parent model to be optionally evaluated as additional row variable (multiplying R in the above formula). Defaults to 0.
            implicitGroups (list, optional): Implicit term groups of parent model to which this term belongs to. Defaults to [1].
            generalGroups (list, optional): General term groups of parent model to which this term belongs to. Defaults to [1].
            customNormConst (Union[CustomNormConst,float,int], optional): Custom normalization constant options - corresponds to the constant c in the above formula. Defaults to CustomNormConst(), if a float/int is passed will use that as the normalization constant.
            timeSignalData (TimeSignalData, optional): Time dependence options - responsible for non-trivial components of the T in the above formula. Defaults to TimeSignalData().
            varData (VarData, optional): Required row and column variable data - corresponding to R and C in the above formula. Defaults to VarData().
            stencilData (dict, optional): Stencil data options - sets S in the above formula. See routines ending in Stencil in simple_containers.py. Defaults to {}, but should be supplied.
            skipPattern (bool, optional): Set to true if the matrix pattern should be skipped during PETSc preallocation (useful when the same pattern has been added already). Defaults to False.
            copyTermName (Union[str,None], optional): Name of term whose matrix is to be copied and multiplied element-wise with this term's stencil. They must have the shame sparsity pattern, i.e. the same stencil shape. Defaults to None.
        """

        super().__init__(evolvedVar, implicitGroups, generalGroups)

        if len(implicitVar) == 0:
            implicitVar = evolvedVar

        self.__implicitVar__ = implicitVar
        self.__spatialProfile__ = spatialProfile
        self.__harmonicProfile__ = harmonicProfile
        self.__velocityProfile__ = velocityProfile
        self.__evaluatedTermGroup__ = evaluatedTermGroup
        self.__customNormConst__ = (
            customNormConst
            if isinstance(customNormConst, CustomNormConst)
            else CustomNormConst(customNormConst)
        )
        self.__timeSignalData__ = timeSignalData
        self.__varData__ = varData
        self.__stencilData__ = stencilData
        self.__skipPattern__ = skipPattern
        self.__fixedMatrix__ = fixedMatrix
        self.__copyTermName__ = copyTermName

    def checkTerm(self, varCont: VariableContainer):
        """Perform consistency check on term

        Args:
            varCont (VariableContainer): Variable container to be used with this term
        """

        super().checkTerm(varCont)

        rowVarOnDual = varCont.dataset[self.evolvedVar].attrs["isOnDualGrid"]

        assert self.__implicitVar__ in varCont.dataset.data_vars.keys(), (
            "Implicit variable "
            + self.__implicitVar__
            + " not registered in used variable container"
        )

        colVarOnDual = varCont.dataset[self.__implicitVar__].attrs["isOnDualGrid"]

        self.__varData__.checkRowColVars(varCont, rowVarOnDual, colVarOnDual)

    def dict(self):
        """Returns dictionary form of GeneralMatrixTerm to be used in json output

        Returns:
            dict: Dictionary form of GeneralMatrixTerm to be used as individual custom term properties
        """
        gTerm = {
            "termType": "matrixTerm",
            "evolvedVar": self.evolvedVar,
            "implicitVar": self.__implicitVar__,
            "spatialProfile": self.__spatialProfile__,
            "harmonicProfile": self.__harmonicProfile__,
            "velocityProfile": self.__velocityProfile__,
            "evaluatedTermGroup": self.__evaluatedTermGroup__,
            "implicitGroups": self.implicitGroups,
            "generalGroups": self.generalGroups,
            "customNormConst": self.__customNormConst__.dict(),
            "timeSignalData": self.__timeSignalData__.dict(),
            "varData": self.__varData__.dict(),
            "stencilData": self.__stencilData__,
            "skipPattern": self.__skipPattern__,
            "fixedMatrix": self.__fixedMatrix__,
        }

        if self.__copyTermName__ is not None:
            gTerm["multCopyTermName"] = self.__copyTermName__

        return gTerm


class DerivationTerm(Term):
    """Derivation-based explicit term options used to construct custom models. The result of evaluating a derivation term is the result of the derivation optionally multiplied by a modelbound variable. Does not support evolving distributions."""

    def __init__(
        self,
        evolvedVar: str,
        derivationRule: dict,
        mbVar: Union[str, None] = None,
        generalGroups=[1],
    ) -> None:
        """Derivation term constructor

        Args:
            evolvedVar (str): Name of the evolved variable. Distributions not supported.
            derivationRule (dict): Derivation rule containing name and required variables.
            mbVar (Union[str,None], optional): Optional modelbound variable. Defaults to None.
            generalGroups (list, optional): General groups this term belongs to within its model. Defaults to [1].
        """

        super().__init__(evolvedVar, [], generalGroups)

        self.__derivationRule__ = derivationRule
        self.__mbVar__ = mbVar

    def checkTerm(self, varCont: VariableContainer):
        """Perform consistency check on term, including the required variables

        Args:
            varCont (VariableContainer): Variable container to be used with this term
        """

        super().checkTerm(varCont)

        requiredVars = self.__derivationRule__["requiredVarNames"]

        for name in requiredVars:
            assert name in varCont.dataset.data_vars.keys(), (
                "Required derivation variable "
                + name
                + " not registered in used variable container"
            )

    def dict(self):
        """Returns dictionary form of DerivationTerm to be used in json output

        Returns:
            dict: Dictionary form of DerivationTerm to be used as individual custom term properties
        """
        gTerm = {
            "termType": "derivationTerm",
            "evolvedVar": self.evolvedVar,
            "generalGroups": self.generalGroups,
        }

        if self.__mbVar__ is not None:
            gTerm["requiredMBVarName"] = self.__mbVar__

        gTerm.update(self.__derivationRule__)

        return gTerm


class TermGenerator:
    """Term generator class used to track the term groups into which the generators will put their turns"""

    def __init__(
        self,
        implicitGroups: List[int] = [1],
        generalGroups: List[int] = [],
        options: Dict[str, object] = {},
    ) -> None:
        self.__implicitGroups__ = implicitGroups
        self.__generalGroups__ = generalGroups
        self.__options__ = options

    @property
    def implicitGroups(self) -> List[int]:
        return self.__implicitGroups__

    @property
    def generalGroups(self) -> List[int]:
        return self.__generalGroups__

    def dict(self) -> dict:
        """Produce dictionary form of TermGenerator

        Returns:
            dict: Dictionary ready to be added to the config file
        """
        tgDict: Dict[str, object] = {
            "implicitGroups": self.__implicitGroups__,
            "generalGroups": self.__generalGroups__,
        }

        tgDict.update(self.__options__)

        return tgDict


class CustomModel:
    """Custom model object property container"""

    def __init__(self, modelTag: str) -> None:
        self.__modelTag__ = modelTag
        self.__termTags__: List[str] = []
        self.__terms__: List[Term] = []
        self.__modelboundData__: Dict[str, object] = {}
        self.__termGeneratorTags__: List[str] = []
        self.__termGeneratorProperties__: Dict[str, object] = {}
        self.__activeImplicitGroups__: List[int] = []
        self.__activeGeneralGroups__: List[int] = []

    @property
    def mbData(self):
        return self.__modelboundData__

    @property
    def activeImplicitGroups(self):
        return self.__activeImplicitGroups__

    @property
    def activeGeneralGroups(self):
        return self.__activeGeneralGroups__

    def addTerm(self, termTag: str, term: Term):
        self.__termTags__.append(termTag)
        self.__terms__.append(term)
        self.__activeImplicitGroups__ += term.implicitGroups
        self.__activeImplicitGroups__ = list(set(self.__activeImplicitGroups__))
        self.__activeGeneralGroups__ += term.generalGroups
        self.__activeGeneralGroups__ = list(set(self.__activeGeneralGroups__))

    def addTermGenerator(
        self, generatorTag: str, generatorProperties: Union[dict, TermGenerator]
    ):
        self.__termGeneratorTags__.append(generatorTag)

        if isinstance(generatorProperties, dict):
            propertiesDict = generatorProperties
            warnings.warn(
                "Adding term generator as dictionary. This is deprecated and provided only for legacy scripts. Useful checks are disabled. Use at own risk."
            )

        if isinstance(generatorProperties, TermGenerator):
            propertiesDict = generatorProperties.dict()
            self.__activeImplicitGroups__ += generatorProperties.implicitGroups
            self.__activeImplicitGroups__ = list(set(self.__activeImplicitGroups__))
            self.__activeGeneralGroups__ += generatorProperties.generalGroups
            self.__activeGeneralGroups__ = list(set(self.__activeGeneralGroups__))

        self.__termGeneratorProperties__[generatorTag] = propertiesDict

    def setModelboundData(self, mbData: dict):
        self.__modelboundData__ = mbData

    def checkTerms(self, varCont: VariableContainer):
        """Check terms in this model for consistency

        Args:
            varCont (VariableContainer): Variable container to be used in this check
        """
        print("Checking terms in model " + self.__modelTag__ + ":")
        for i, term in enumerate(self.__terms__):
            print("   Checking term " + self.__termTags__[i])
            term.checkTerm(varCont)

    def dict(self):
        """Returns dictionary form of CustomModel to be used in json output

        Returns:
            dict: Dictionary form of CustomModel to be used to update model properties
        """
        cModel = {
            self.__modelTag__: {
                "type": "customModel",
                "termTags": self.__termTags__,
                "modelboundData": self.__modelboundData__,
                "termGenerators": {"tags": self.__termGeneratorTags__},
            }
        }

        cModel[self.__modelTag__]["termGenerators"].update(
            self.__termGeneratorProperties__
        )
        for i in range(len(self.__termTags__)):
            cModel[self.__modelTag__].update(
                {self.__termTags__[i]: self.__terms__[i].dict()}
            )

        return cModel


def derivationRule(derivationName: str, requiredVars: List[str]) -> dict:
    """Returns derivation rule dictionary for ReMKiT1D JSON format

    Args:
        derivationName (str): Name of derivation
        requiredVars (List[str]): List of variables required by the derivation (ordered)

    Returns:
        dict: Dictionary representing derivation rule
    """

    rule = {"ruleName": derivationName, "requiredVarNames": requiredVars}

    return rule


def simpleDerivation(multConst: float, varPowers: List[float]) -> dict:
    """Returns property dictionary for a simple derivation object which calculates its value as multConst * prod(vars**powers)

    Args:
        multConst (float): Multiplicative constant
        varPowers (List[float]): Powers to raise passed variables to.

    Returns:
        dict: Dictionary representing derivation properties
    """
    deriv = {"type": "simpleDerivation", "multConst": multConst, "varPowers": varPowers}

    return deriv


def polyDerivation(
    constCoeff: float, polyPowers: List[float], polyCoeffs: List[float]
) -> dict:
    """Returns property dictionary for polynomial function derivation which calculates its value from a single variable
    as constCoeff + sum(polyCoeffs*var**polyPowers). Can also accept multiple variables, one for each polynomial term and use then.

    Args:
        constCoeff (float): Constant coefficient in polynomial expression
        polyPowers (List[float]): Powers corresponding to each term
        polyCoeffs (List[float]): Polynomial coefficients corresponding to each term


    Returns:
        dict: Dictionary representing derivation properties
    """
    deriv = {
        "type": "polynomialFunctionDerivation",
        "constantPolynomialCoefficient": constCoeff,
        "polynomialPowers": polyPowers,
        "polynomialCoefficients": polyCoeffs,
    }

    return deriv


def additiveDerivation(
    derivTags: List[str],
    resultPower: float,
    derivIndices: List[List[int]],
    linCoeffs: List[float] = [],
) -> dict:
    """Returns property dictionary for additive composite derivation which sums up the results of each derivation in derivTags and
    raises the corresponding result to resultPower.

    Args:
        derivTags (List[str]): List of derivations whose output should be added
        resultPower (float): Power to raise the result of the addition
        derivIndices (List[List[int]]]): List of index lists corresponding to each derivation in derivTags.
        linCoeffs (List[List[int]]]): List linear coefficients corresponding to each derivation in derivTags. Defaults to [] resulting in a list of ones.

    Returns:
        dict: Dictionary representing derivation properties

    Usage:
        Given a passed set of variables to the additive derivation object, each individual derivation in derivTags is passed a subset
        of the variables determined by its entry in derivIndices. For example:

        Variables passed to additive derivation as required variables: ["n","T","q"]
        derivTags: ["deriv1","deriv2"] (assumes these derivations are already registered)
        derivIndices: [[1,2],[1,3,2]]

        The above will result in deriv1 getting access to ["n","T"] and "deriv2" to ["n","q","T"]. Note the order.
    """

    assert len(derivTags) == len(
        derivIndices
    ), "derivTags and derivIndices in additiveDerivation must be of same size"
    if len(linCoeffs) != 0:
        assert len(derivTags) == len(
            linCoeffs
        ), "derivTags and linCoeffs in additiveDerivation must be of same size"

    deriv = {
        "type": "additiveDerivation",
        "derivationTags": derivTags,
        "resultPower": resultPower,
        "linearCoefficients": linCoeffs,
    }

    for i, tag in enumerate(derivTags):
        deriv[tag] = {"derivationIndices": derivIndices[i]}

    return deriv


def multiplicativeDerivation(
    innerDerivation: str,
    innerDerivationIndices: List[int],
    innerDerivationPower=1.0,
    outerDerivation: Union[None, str] = None,
    outerDerivationIndices: Union[None, List[int]] = None,
    outerDerivationPower=1.0,
    funcName: Union[None, str] = None,
) -> dict:
    """Returns property dictionary for multiplicative composite derivation which multiplies the outputs of an inner derivation and an optional outer derivation,
    optionally raising them to corresponding powers and applying a function to the inner derivation.

    Args:
        innerDerivation (str): Tag of inner derivation
        innerDerivationIndices (List[int]): Indices of required variables passed to inner derivation
        innerDerivationPower (float, optional): Power to raise the result of inner derivation to. Defaults to 1.0.
        outerDerivation (Union[None,str], optional): Outer derivation tag. Defaults to None.
        outerDerivationIndices (Union[None,List[int]], optional): Indices of required variables passed to outer derivation. Defaults to None.
        outerDerivationPower (float, optional): Power to raise the result of the outer derivation to. Defaults to 1.0.
        funcName (Union[None,str], optional): Name of the Fortran function to apply to inner derivation. Defaults to None.

    Returns:
        dict: Dictionary repesenting derivation properties

    Usage:

        Allowed funcNames: exp,log,sin,cos,abs,tan,atan,asin,acos,sign,erf,erfc
    """

    if outerDerivation is not None:
        assert (
            outerDerivationIndices is not None
        ), "If multiplicative derivation has an outer derivation, outerDerivationIndices must be passed"

    deriv = {
        "type": "multiplicativeDerivation",
        "innerDerivation": innerDerivation,
        "innerDerivIndices": innerDerivationIndices,
        "innerDerivPower": innerDerivationPower,
        "outerDerivation": "none" if outerDerivation is None else outerDerivation,
        "outerDerivIndices": (
            [] if outerDerivationIndices is None else outerDerivationIndices
        ),
        "outerDerivPower": outerDerivationPower,
        "innerDerivFuncName": "none" if funcName is None else funcName,
    }

    return deriv


def boundedExtrapolationDerivation(
    extrapolationProperties: dict,
    expectUpperBoundVar=False,
    expectLowerBoundVar=False,
    ignoreUpperBound=False,
    ignoreLowerBound=False,
    fixedLowerBound=0.0,
    fixedUpperBound: Union[None, float] = None,
) -> dict:
    """Returns bounded extrapolation derivation which calculates a scalar variable based on extrapolation properties at a boundary and
    bounds the value optionally.
    The first expected variable name is the interpolated variable.
    The second is the upper bound if no lower bound variable is expected, otherwise it is the lower bound. The third is the upper bound (if expected).
    Bounds are all expected to have positive values, and this will be ensured by taking their absolute values.
    If applied to the left boundary, will apply -abs(lowerBound) as upper bound and -abs(upperBound) as lower bound, i.e. the bounds are reflected around 0.

    Args:
        extrapolationProperties (dict): Properties of extrapolation used
        expectUpperBoundVar (bool, optional): True if an upper bound variable is expected (can be either scalar or fluid). Defaults to False.
        expectLowerBoundVar (bool, optional): True if a lower bound variable is expected (can be either scalar or fluid). Defaults to False.
        ignoreUpperBound (bool, optional): True if the upper bound value should be ignored. Defaults to False.
        ignoreLowerBound (bool, optional): True if the lower bound value should be ignored (allowes for sign change). Defaults to False.
        fixedLowerBound (float, optional): Fixed lower bound value - ignored if lower bound variable expected. Defaults to 0.0.
        fixedUpperBound (Union[None,float], optional): Fixed upper bound value - ignored if upper bound variable expected. Defaults to None.

    Returns:
        dict: Dictionary representing derivation properties
    """

    deriv = {
        "type": "boundedExtrapolationDerivation",
        "expectUpperBoundVar": expectUpperBoundVar,
        "expectLowerBoundVar": expectLowerBoundVar,
        "ignoreUpperBound": ignoreUpperBound,
        "ignoreLowerBound": ignoreLowerBound,
        "fixedLowerBound": fixedLowerBound,
        "extrapolationStrategy": extrapolationProperties,
    }

    if fixedUpperBound is not None:
        deriv["fixedUpperBound"] = fixedUpperBound

    return deriv


def linExtrapolation(
    leftBoundary=False, staggeredVars=False, expectedHaloWidth: Union[None, int] = None
) -> dict:
    """Returns extrapolation properties for a linear extrapolation

    Args:
        leftBoundary (bool, optional): True if this is an extrapolation for the left boundary. Defaults to False.
        staggeredVars (bool, optional): True if extrapolating staggered variables. Defaults to False.
        expectedHaloWidth (Union[None,Int], optional): Expected x halo width (should be 0 for some modelbound data). Defaults to None, which uses the mpi value in ReMKiT1D.

    Returns:
        dict: Extrapolation property dictionary
    """

    extrap = {
        "type": "linExtrapolation",
        "leftBoundary": leftBoundary,
        "staggeredVars": staggeredVars,
    }

    if expectedHaloWidth is not None:
        extrap["expectedHaloWidth"] = expectedHaloWidth

    return extrap


def logExtrapolation(
    leftBoundary=False, staggeredVars=False, expectedHaloWidth: Union[None, int] = None
) -> dict:
    """Returns extrapolation properties for a logarithmic extrapolation

    Args:
        leftBoundary (bool, optional): True if this is an extrapolation for the left boundary. Defaults to False.
        staggeredVars (bool, optional): True if extrapolating staggered variables. Defaults to False.
        expectedHaloWidth (Union[None,Int], optional): Expected x halo width (should be 0 for some modelbound data). Defaults to None, which uses the mpi value in ReMKiT1D.

    Returns:
        dict: Extrapolation property dictionary
    """

    extrap = {
        "type": "logExtrapolation",
        "leftBoundary": leftBoundary,
        "staggeredVars": staggeredVars,
    }

    if expectedHaloWidth is not None:
        extrap["expectedHaloWidth"] = expectedHaloWidth

    return extrap


def linLogExtrapolation(
    leftBoundary=False, staggeredVars=False, expectedHaloWidth: Union[None, int] = None
) -> dict:
    """Returns extrapolation properties for a logarithmic combined with a linear extrapolation of the second to last point

    Args:
        leftBoundary (bool, optional): True if this is an extrapolation for the left boundary. Defaults to False.
        staggeredVars (bool, optional): True if extrapolating staggered variables. Defaults to False.
        expectedHaloWidth (Union[None,Int], optional): Expected x halo width (should be 0 for some modelbound data). Defaults to None, which uses the mpi value in ReMKiT1D.

    Returns:
        dict: Extrapolation property dictionary
    """

    extrap = {
        "type": "linlogExtrapolation",
        "leftBoundary": leftBoundary,
        "staggeredVars": staggeredVars,
    }

    if expectedHaloWidth is not None:
        extrap["expectedHaloWidth"] = expectedHaloWidth

    return extrap


def coldIonIJIntegralDerivation(index: int, isJInt=False) -> dict:
    """Returns derivation properties for derivation which returns value of the cold ion Shkarofsky I/J integral used in electron-ion collisions for l>0. Expects the passed variable to be ion flow velocity.

    Args:
        index (int): Integral index
        isJInt (bool, optional): True if the integral is a J integral, otherwise its the I integral. Defaults to False.

    Returns:
        dict: Derivation property dictionary
    """

    deriv = {
        "type": "coldIonIJIntegralDerivation",
        "isJIntegral": isJInt,
        "index": index,
    }

    return deriv


def ijIntegralDerivation(index: int, isJInt=False) -> dict:
    """Returns derivation properties for derivation which returns value of Shkarofsky I/J integrals for each passed harmonic. Can handle
    both distributions and single harmonic variables

    Args:
        index (int): Integral index
        isJInt (bool, optional): True if the integral is a J integral, otherwise its the I integral. Defaults to False.

    Returns:
        dict: Derivation property dictionary
    """

    deriv = {"type": "IJIntegralDerivation", "isJIntegral": isJInt, "index": index}

    return deriv


def harmonicExtractorDerivation(index: int) -> dict:
    """Returns derivation properties for derivation which extracts a single harmonic with given index from a distribution variable
    and returns it as a single harmonic variable

    Args:
        index (int): Index of the extracted harmonic

    Returns:
        dict: Derivation property dictionary
    """

    deriv = {"type": "harmonicExtractorDerivation", "index": index}

    return deriv


def distScalingExtrapolationDerivation(
    extrapolateToBoundary=False, staggeredVars=False, leftBoundary=False
) -> dict:
    """Returns properties for a matrix valued derivation that extrapolates a distribution variable through scaling.
    Expects 1-4 variables. The first is the extrapolated distribution, the second the density vector. The third is the staggered density
    if the variables are staggered. The value of the density at the boundary (scalar) is either the third or the fourth variable, and should be present
    only if extrapolating to the boundary.

    Args:
        extrapolateToBoundary (bool, optional): True if extrapolating to the very boundary, otherwise extrapolates to last cell centre. Defaults to False.
        staggeredVars (bool, optional): True if the distribution has staggered odd l harmonics. Defaults to False.
        leftBoundary (bool, optional): True if the extrapolation is associated to the left domain boundary. Defaults to False.

    Returns:
        dict: Derivation property dictionary
    """

    deriv = {
        "type": "distScalingExtrapDerivation",
        "extrapolateToBoundary": extrapolateToBoundary,
        "staggeredVars": staggeredVars,
        "leftBoundary": leftBoundary,
    }

    return deriv


def ddvDerivation(
    includedHarmonics: List[int],
    innerV: Union[None, List[List[float]]] = None,
    outerV: Union[None, List[List[float]]] = None,
    vifAtZero: Union[None, List[List[float]]] = None,
) -> dict:
    """Return derivation properties for a velocity derivative of a distribution variable. Calculates v_o * d(v_i * f)/dv.

    Args:
        includedHarmonics (List[int]): List of harmonics with non-trivial innerV/outerV/vifAtZero. If its of length one assumes only that harmonic should be used and returns a single harmonic variable
        innerV (Union[None,list[List[float]]], optional): Inner velocity vector for each included harmonic (at right cell boundaries in v space). Defaults to None, which results in vectors of ones.
        outerV (Union[None,list[List[float]]], optional): Outer velocity vector for each included harmonics. Defaults to None, which results in vectors of ones.
        vifAtZero (Union[None,list[List[float]]], optional): Extrapolation coefficents of v_i*f at zero in the form A1*f(v1)+A2*f(v2) where A's are given for each included harmonic. Defaults to None, which results in [0,0]

    Returns:
        dict: Derivation property dictionary
    """
    deriv: Dict[str, object] = {"type": "ddvDerivation"}
    if len(includedHarmonics) == 1:
        deriv["targetH"] = includedHarmonics[0]
        if outerV is not None:
            deriv["outerV"] = outerV[0]
        if innerV is not None:
            deriv["innerV"] = innerV[0]
        if vifAtZero is not None:
            deriv["vifAtZero"] = vifAtZero[0]
    else:
        for ind, harmonic in enumerate(includedHarmonics):
            if outerV is not None:
                cast(Dict[str, object], deriv["outerV"])["h=" + str(harmonic)] = outerV[
                    ind
                ]
            if innerV is not None:
                cast(Dict[str, object], deriv["innerV"])["h=" + str(harmonic)] = innerV[
                    ind
                ]
            if vifAtZero is not None:
                cast(Dict[str, object], deriv["vifAtZero"])["h=" + str(harmonic)] = (
                    vifAtZero[ind]
                )

    return deriv


def d2dv2Derivation(
    includedHarmonics: List[int],
    innerV: Union[None, List[List[float]]] = None,
    outerV: Union[None, List[List[float]]] = None,
    vidfdvAtZero: Union[None, List[List[float]]] = None,
) -> dict:
    """Return derivation properties for a second velocity derivative of a distribution variable. Calculates v_o * d(v_i * df/dv)/dv.

    Args:
        includedHarmonics (List[int]): List of harmonics with non-trivial innerV/outerV/vidfdvAtZero. If its of length one assumes only that harmonic should be used and returns a single harmonic variable
        innerV (Union[None,list[List[float]]], optional): Inner velocity vector for each included harmonic (at right cell boundaries in velocity space). Defaults to None, which results in vectors of ones.
        outerV (Union[None,list[List[float]]], optional): Outer velocity vector for each included harmonics. Defaults to None, which results in vectors of ones.
        vidfdvAtZero (Union[None,list[List[float]]], optional): Extrapolation coefficents of v_i*df/dv at zero in the form A1*f(v1)+A2*f(v2) where A's are given for each included harmonic. Defaults to None, which results in [0,0]

    Returns:
        dict: Derivation property dictionary
    """
    deriv: Dict[str, object] = {"type": "d2dv2Derivation"}
    if len(includedHarmonics) == 1:
        deriv["targetH"] = includedHarmonics[0]
        if outerV is not None:
            deriv["outerV"] = outerV[0]
        if innerV is not None:
            deriv["innerV"] = innerV[0]
        if vidfdvAtZero is not None:
            deriv["vidfdvAtZero"] = vidfdvAtZero[0]
    else:
        for ind, harmonic in enumerate(includedHarmonics):
            if outerV is not None:
                cast(Dict[str, object], deriv["outerV"])["h=" + str(harmonic)] = outerV[
                    ind
                ]
            if innerV is not None:
                cast(Dict[str, object], deriv["innerV"])["h=" + str(harmonic)] = innerV[
                    ind
                ]
            if vidfdvAtZero is not None:
                cast(Dict[str, object], deriv["vidfdvAtZero"])["h=" + str(harmonic)] = (
                    vidfdvAtZero[ind]
                )

    return deriv


def momentDerivation(
    momentHarmonic: int,
    momentOrder: int,
    multConst=1.0,
    gVec: Union[List[float], None] = None,
    varPowers: List[float] = [],
) -> dict:
    """Return derivation properties of custom moment derivation. First step in the derivation is taking the corresponding moment of the corresponding
    harmonic in the distribution which is expected to be the first passed variable, optinally multiplied by a constan velocity space vector gVec (4*pi*int(f*g*v**(momentOrder+2)dv)). The results is then optionally multiplied by a constant and a product of fluid variables raised to powers (as in SimpleDerivation).

    Args:
        momentHarmonic (int): Harmonic index to take moment of
        momentOrder (int): Order of the moment
        multConst (float, optional): Multiplicative constant. Defaults to 1.0.
        gVec (Union[List[float], optional): Constant optional velocity space vector. Defaults to None, giving a vector of ones.
        varPowers (List[float], optional): Optional fluid variable powers to multiply the moment result with. Defaults to [].

    Returns:
        dict: Derivation property dictionary
    """

    deriv = {
        "type": "momentDerivation",
        "momentHarmonic": momentHarmonic,
        "momentOrder": momentOrder,
        "multConst": multConst,
        "varPowers": varPowers,
    }

    if gVec is not None:
        deriv["gVector"] = gVec

    return deriv


def vContractionDerivation(
    targetHarmonic: int, gVec: List[float], expNHarmonics=0
) -> dict:
    """Return derivation properties of velocity space contration derivation. Takes the target harmonic of passed distribution variable and returns its velocity space
    dot product with g

    Args:
        targetHarmonic (int): Harmonic index to contract
        gVec (List[float]): Velocity space vector to contract the distribution harmonic with
        expNHarmonics (int): Expected variable number of harmonics. Set to 1 for single harmonic variable. Defaults to 0, expecting a distribution.
    Returns:
        dict: Derivation property dictionary
    """

    deriv = {
        "type": "velocityContractionDerivation",
        "targetH": targetHarmonic,
        "gVector": gVec,
        "expectedNumberOfHarmonics": expNHarmonics,
    }

    return deriv


def locValExtractorDerivation(targetX: int) -> dict:
    """Return derivation properties for a derivation object which extracts a single location value from a fluid variable and stores it as a scalar

    Args:
        targetX (int): Global x cell index to extract value from

    Returns:
        dict: Derivation property dictionary
    """

    deriv = {"type": "locValExtractorDerivation", "targetX": targetX}

    return deriv


def vTensorProdDerivation(power=1.0, gVec: Union[List[float], None] = None) -> dict:
    """Return derivation properties of a tensor product with velocity vector derivation. Expects 1 or 2 fluid variables. If both variables
    passed will return single harmonic where the i-th x and j-th v coordinate value are (gVec(j)+secondVar(i))**power*firstVar(i). If only one
    variable is passed secondVar is set to 0. Returns a single harmonic variable.

    Args:
        power (float, optional): Power to raise the shifted velocity vector to. Defaults to 1.0
        gVec (Union[List[float],None], optional): Velocity vector to use instead of velocity grid. Defaults to None.

    Returns:
        dict: Derivation property dictionary
    """

    deriv = {"type": "velocityTensorProdDerivation", "power": power, "gVector": gVec}

    return deriv


def generalizedIntPolyDerivation(
    polyPowers: List[List[int]],
    polyCoeffs: List[float],
    multConst: float = 1.0,
    funcName: Union[None, str] = None,
) -> dict:
    """Return a derivation object which calculates a multConst * fun(sum c_i prod vars**powers), where prod vars**powers is shorthand for
    the product of passed variables raised to powers corresponding to polynomial coefficient c_i. For allowed functions se multiplicativeDerivation.

    Args:
        polyPowers (List[List[int]]]): Powers corresponding to each coefficient and passed variable
        polyCoeffs (List[float]): Polynomial coefficients (c_i above)
        multConst (float, optional): Multiplicative constant in front of derivation. Defaults to 1.0.
        funcName (Union[None,str], optional): Optional function applied to polynomial result. Defaults to None.

    Returns:
        dict: Derivation property dictionary
    """

    polyPowersDict = {}
    for i, powers in enumerate(polyPowers):
        polyPowersDict["index" + str(i + 1)] = powers

    deriv = {
        "type": "generalizedIntPowerPolyDerivation",
        "multConst": multConst,
        "polynomialPowers": polyPowersDict,
        "polynomialCoefficients": polyCoeffs,
    }

    if funcName is not None:
        deriv["functionName"] = funcName

    return deriv


def rangeFilterDerivation(
    derivName: str,
    controlIndices: List[int],
    controlRanges: List[List[float]],
    derivIndices: Union[None, List[int]] = None,
) -> dict:
    """Return composite derivation object which wraps another derivation with range-based filtering, zeroing out all values where
    passed variables corresponding to controlIndices are outside ranges specified by controlRanges. If derivIndices aren't present all
    passed variables are passed to the derivation in that order.

    Args:
        derivName (str): Name of the wrapped derivation
        controlIndices (List[int]): Indices of passed variables corresponding to control variables
        controlRanges (list[List[float]]): Ranges (all length 2) corresponding to each control variable
        derivIndices (Union[None,List[int]], optional): Optional subset of passed variables passed to the wrapped derivation. Defaults to None, passing all variables.

    Returns:
        dict: Derivation property dictionary
    """

    controlRangesDict = {}
    for i, range in enumerate(controlRanges):
        controlRangesDict["index" + str(i + 1)] = range

    deriv = {
        "type": "rangeFilterDerivation",
        "ruleName": derivName,
        "controlIndices": controlIndices,
        "controlRanges": controlRangesDict,
    }

    if derivIndices is not None:
        deriv["derivationIndices"] = derivIndices

    return deriv


def nDInterpolationDerivation(
    grids: List[np.ndarray], data: np.ndarray, gridNames: Union[List[str], None] = None
) -> dict:
    """Returns a derivation object that will perform n-dimensional linear interpolation on passed data. The number of entries in the grid list must be the dimensionality of the data ndarray, and each individual entry must have the same size as the corresponding dimension of the data. The optional grid names are only used in the construction of the JSON format for ReMKiT1D.

    Args:
        grids (List[np.ndarray]): Values of each of the grids associated with individual data dimensions
        data (np.ndarray): Data to interpolate over
        gridNames (Union[List[str],None], optional): Optional grid names. Defaults to None, resulting in numbered grids.


    Returns:
        dict: Derivation property dictionary
    """

    assert len(grids) == len(
        np.shape(data)
    ), "grids must have the same length as the dimensionality of the passed data"

    dataShape = np.shape(data)

    if gridNames is not None:
        assert len(grids) == len(
            gridNames
        ), "gridNames must have the same length as the passed grid list"
        usedNames = gridNames
    else:
        usedNames = ["grid" + str(i) for i in range(len(grids))]

    for i, grid in enumerate(grids):
        assert len(grid) == dataShape[i], (
            usedNames[i]
            + " does not conform to the corresponding dimension of interpolation data"
        )

    deriv = {
        "type": "nDLinInterpDerivation",
        "data": {"dims": list(dataShape), "values": data.flatten(order="F").tolist()},
        "grids": {"names": usedNames},
    }

    for i, name in enumerate(usedNames):
        cast(Dict[str, object], deriv["grids"])[name] = grids[i].tolist()

    return deriv


def groupEvaluatorManipulator(
    modelTag: str, evalTermGroup: int, resultVarName: str, priority=4
) -> dict:
    """Return manipulator properties for a group evaluator manipulator, which evaluates a term group in a specific model and stores it in a
    variable

    Args:
        modelTag (str): Tag of model whose term should be evaluated
        evalTermGroup (int): Group which should be evaluated
        resultVarName (str): Name of variable into which to store the evaluation
        priority (int, optional): Manipulator priority (0-4). Defaults to 4.

    Returns:
        dict: Manipulator property dictionary
    """

    manip = {
        "type": "groupEvaluator",
        "modelTag": modelTag,
        "evaluatedTermGroup": evalTermGroup,
        "resultVarName": resultVarName,
        "priority": priority,
    }

    return manip


def termEvaluatorManipulator(
    modelTermTags: List[Tuple[str, str]],
    resultVarName: str,
    priority=4,
    accumulate=False,
    update=False,
) -> dict:
    """Return manipulator properties for a term evaluator manipulator, which evaluates a set of model,term pairs and stores the result
    in a given variable

    Args:
        modelTermTag (List[Tuple[str,str]]): Pairs of model,term tags to be evaluated
        resultVarName (str): Name of variable into which to store the evaluation
        priority (int, optional): Manipulator priority (0-4). Defaults to 4.
        accumulate (bool, optional): If true, will accumulate the values into the result variable instead of overwriting. Defaults to False.
        update (bool, optional): If true will independently request updates for the evaluated terms/models. Defaults to False.

    Returns:
        dict: Manipulator property dictionary
    """

    models, terms = zip(*modelTermTags)
    manip = {
        "type": "termEvaluator",
        "evaluatedModelNames": list(models),
        "evaluatedTermNames": list(terms),
        "resultVarName": resultVarName,
        "priority": priority,
        "update": update,
        "accumulate": accumulate,
    }

    return manip


def extractorManipulator(
    modelTag: str, modelboundDataName: str, resultVarName: str, priority=4
) -> dict:
    """Return manipulator properties for an extractor manipulator, which extracts a modelbound variable from a specific model and stores it in
    a variable

    Args:
        modelTag (str): Tag of model hosting the relevant modelbound data
        modelboundDataName (str): Name of the modelbound variable which should be extracted
        resultVarName (str): Name of variable into which to store the extracted modelbound variable valu
        priority (int, optional): Manipulator priority (0-4). Defaults to 4.

    Returns:
        dict: Manipulator property dictionary
    """

    manip = {
        "type": "modelboundDataExtractor",
        "modelTag": modelTag,
        "modelboundDataName": modelboundDataName,
        "resultVarName": resultVarName,
        "priority": priority,
    }

    return manip


def staggeredDivStencil() -> dict:
    """Return basic staggered divergence stencil

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {"stencilType": "staggeredDifferenceStencil"}

    return stencil


def staggeredGradStencil() -> dict:
    """Return basic staggered gradient stencil

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {"stencilType": "staggeredDifferenceStencil", "ignoreJacobian": True}

    return stencil


def boundaryStencilDiv(
    fluxJacVar: str, lowerBoundVar: Union[None, str] = None, isLeft=False
) -> dict:
    """Boundary stencil for divergence terms with extrapolation and lower jacobian bound. Effectively gives the missing boundary div(fluxJacVar * implicitVar) component, handling bounds and extrapolation.

    Args:
        fluxJacVar (str): Name of flux jacobian
        lowerBoundVar (Union[None,str], optional): Name of lower bound variable. Defaults to None - lower bound = 0.
        isLeft (bool, optional): Set to true if stencil is for left boundary. Defaults to False.

    Returns:
        dict: ReMKiT1D boundary stencil option dictionary
    """

    stencil = {
        "stencilType": "boundaryStencil",
        "fluxJacVar": fluxJacVar,
        "leftBoundary": isLeft,
    }

    if lowerBoundVar is not None:
        stencil["lowerBoundVar"] = lowerBoundVar

    return stencil


def boundaryStencilGrad(isLeft=False) -> dict:
    """Boundary stencil for gradient terms with extrapolation

    Args:
        isLeft (bool, optional): Set to true if stencil is for left boundary. Defaults to False.

    Returns:
        dict: ReMKiT1D boundary stencil option dictionary
    """

    stencil = {
        "stencilType": "boundaryStencil",
        "leftBoundary": isLeft,
        "ignoreJacobian": True,
    }

    return stencil


def centralDiffStencilDiv(fluxJacVar: str = "none") -> dict:
    """Return central differenced divergence stencil

    Args:
        fluxJacVar (str, optional): Optional flux jacobian to be interpolated on cell boundaries. Defaults to "none".

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "centralDifferenceInterpolated",
        "interpolatedVarName": fluxJacVar,
    }

    return stencil


def centralDiffStencilGrad() -> dict:
    """Return central differenced gradient stencil

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {"stencilType": "centralDifferenceInterpolated", "ignoreJacobian": True}

    return stencil


def upwindedDiv(fluxJacVar: str) -> dict:
    """Return upwinded divergence stencil

    Args:
        fluxJacVar (str): Flux jacobian variable used to upwind the evolved variable

    Returns:
        dict: Stencil property dictionary
    """
    stencil = {"stencilType": "upwindedDifference", "fluxJacVar": fluxJacVar}

    return stencil


def diffusionStencil(
    ruleName: str, reqVarNames: List[str], doNotInterpolate=False
) -> dict:
    """Return diffusion stencil in x which assumes that both the implicit and evolved variables live on the regular grid.

    Args:
        ruleName (str): Name of derivation used to calculate the diffusion coefficent
        reqVarNames (List[str]): Variable names required for the derivation
        doNotInterpolate (bool, optional): If true will assume that the rule already calculates the diffusion coefficient on cell boundaries. Defaults to False.
    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "diffusionStencil",
        "ruleName": ruleName,
        "requiredVarNames": reqVarNames,
        "doNotInterpolateDiffCoeff": doNotInterpolate,
    }

    return stencil


def momentStencil(momentOrder: int, momentHarmonic: int) -> dict:
    """Return stencil representing the evolution of a fluid quantity by taking a moment of a distribution harmonic

    Args:
        momentOrder (int): Order of the moment
        momentHarmonic (int): Harmonic index to take the moment of

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "momentStencil",
        "momentOrder": momentOrder,
        "momentHarmonic": momentHarmonic,
    }

    return stencil


def kineticSpatialDiffStencil(rowHarmonic: int, colHarmonic: int) -> dict:
    """Return stencil representing the evolution of rowHarmonic due to spatial gradients of colHarmonic. If the harmonics represent
    l-numbers of different parities and the variables are staggered the difference will be calculated using forward/backwards staggered difference,
    otherwise it will be calculated using central difference with interpolation at spatial cell faces.

    Args:
        rowHarmonic (int): Index of row (evolved) harmonic
        colHarmonic (int): Indef of column (implict) harmonic

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "kineticSpatialDiffStencil",
        "rowHarmonic": rowHarmonic,
        "colHarmonic": colHarmonic,
    }

    return stencil


def ddvStencil(
    rowHarmonic: int,
    colHarmonic: int,
    fixedC: Union[List[float], None] = None,
    fixedInterp: Union[List[float], None] = None,
    cfAtZero: Union[List[float], None] = None,
    modelboundC="none",
    modelboundInterp="none",
) -> dict:
    """Return stencil representing the evolution of rowHarmonic due to d(C*f_l)/dv, where l is the column harmonic. C is assumed to
    be defined on right velocity cell boundaries, and f_l is interpolated to those boundaries using either standard or user-defined interpolation
    coefficients (also defined at right cell boundaries). f_{n+1/2} is given by (1-interp(n))*f_{n} + interp(n)*f_{n+1}, where n is a velocity cell index.

    Args:
        rowHarmonic (int): Index of row (evolved) harmonic
        colHarmonic (int): Index of column (implicit) harmonic
        fixedC (Union[List[float],None], optional): Fixed C coefficent, defined at velocity cell boundaries. Defaults to None, giving a vector of ones.
        fixedInterp (Union[List[float],None], optional): Fixed interpolation coefficients used to interpolate f to velocity cell boundaries. Defaults to None.
        cfAtZero (Union[List[float],None], optional): Extrapolation coefficients (length 2) of C*f at zero in the form A1*f(v1)+A2*f(v2) where both A's are fixed. Defaults to None, resulting in zeros.
        modelboundC (str, optional): Name of modelbound C variable - if present will be used over the fixed value. Defaults to "none".
        modelboundInterp (str, optional): Name of modelbound interpolation coefficient variable - if present will be used over the fixed value. Defaults to "none".

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "ddvStencil",
        "modelboundC": modelboundC,
        "modelboundInterp": modelboundInterp,
        "rowHarmonic": rowHarmonic,
        "colHarmonic": colHarmonic,
    }

    if fixedC is not None:
        stencil["fixedC"] = fixedC

    if fixedInterp is not None:
        stencil["fixedInterp"] = fixedInterp

    if cfAtZero is not None:
        stencil["cfAtZero"] = cfAtZero

    return stencil


def velDiffusionStencil(
    rowHarmonic: int,
    colHarmonic: int,
    fixedA: Union[List[float], None] = None,
    adfAtZero: Union[List[float], None] = None,
    modelboundA="none",
) -> dict:
    """Return stencil representing the evolution of rowHarmonic due to d(A*d/f_l/dv)/dv, where l is the column harmonic. A is assumed to
    be defined on right velocity cell boundaries.

    Args:
        rowHarmonic (int): Index of row (evolved) harmonic
        colHarmonic (int): Index of column (implicit) harmonic
        fixedC (Union[List[float],None], optional): Fixed A coefficent, defined at velocity cell boundaries. Defaults to None, giving a vector of ones.
        adfAtZero (Union[List[float],None], optional): Extrapolation coefficients (length 2) of A*df/dv at zero in the form A1*f(v1)+A2*f(v2) where both A's are fixed. Defaults to None, resulting in zeros.
        modelboundA (str, optional): Name of modelbound A variable - if present will be used over the fixed value. Defaults to "none".

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "vDiffusionStencil",
        "modelboundA": modelboundA,
        "rowHarmonic": rowHarmonic,
        "colHarmonic": colHarmonic,
    }

    if fixedA is not None:
        stencil["fixedA"] = fixedA

    if adfAtZero is not None:
        stencil["adfAtZero"] = adfAtZero

    return stencil


def ijIntegralStencil(
    rowHarmonic: int, colHarmonic: int, integralIndex: int, isJIntegral=False
) -> dict:
    """Return stencil evolving rowHarmonic through Shkarofsky I or J integrals of colHarmonic

    Args:
        rowHarmonic (int): Index of row (evolved) harmonic
        colHarmonic (int): Index of column (implict) harmonic
        integralIndex (int): Index of I/J integral
        isJIntegral (bool, optional): True if the integral is the J integral. Defaults to False.

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "shkarofskyIJStencil",
        "JIntegral": isJIntegral,
        "rowHarmonic": rowHarmonic,
        "colHarmonic": colHarmonic,
        "integralIndex": integralIndex,
    }

    return stencil


def fixedEnergyBoltzmannStencil(
    rowHarmonic: int,
    transitionIndex: int,
    fixedEnergyIndex: int,
    absorptionTerm=False,
    detailedBalanceTerm=False,
) -> dict:
    """Return Boltzmann collision operator for given harmonic and transition

    Args:
        rowHarmonic (int): Evolved and implicit harmonic index
        transitionIndex (int): Index of the transition in CRM modelbound data which this stencil is modelling
        fixedEnergyIndex (int): Index of the energy value of this transition in the CRM inelastic data
        absorptionTerm (bool, optional): Set to true if this stencil represents the absorption term as opposed to the emission term. Defaults to False.
        detailedBalanceTerm (bool, optional): Set to true if this stencil is associated with a detailed balance transiton. Defaults to False.

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "boltzmannStencil",
        "rowHarmonic": rowHarmonic,
        "fixedEnergyIndex": fixedEnergyIndex,
        "transitionIndex": transitionIndex,
        "absorptionTerm": absorptionTerm,
        "detailedBalanceTerm": detailedBalanceTerm,
    }

    return stencil


def variableEnergyBoltzmannStencil(
    rowHarmonic: int, transitionIndex: int, absorptionTerm=False, superelasticTerm=False
) -> dict:
    """Return Boltzmann collision operator for given harmonic and transition

    Args:
        rowHarmonic (int): Evolved and implicit harmonic index
        transitionIndex (int): Index of the transition in CRM modelbound data which this stencil is modelling
        absorptionTerm (bool, optional): Set to true if this stencil represents the absorption term as opposed to the emission term. Defaults to False.
        superelasticTerm (bool, optional): Set to true if this stencil is associated with a superelastic (negative transition energy) transiton. Defaults to False.

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "variableBoltzmannStencil",
        "rowHarmonic": rowHarmonic,
        "transitionIndex": transitionIndex,
        "absorptionTerm": absorptionTerm,
        "superelasticTerm": superelasticTerm,
    }

    return stencil


def scalingLBCStencil(
    rowHarmonic: int,
    colHarmonic: int,
    derivRule: dict,
    decompHarmonics: Union[List[float], None] = None,
    leftBoundary=False,
):
    """Return the electron logical boundary condition stencil v*f_b/dx acting on rowHarmonic due to the decomposition of colHarmonic, which
    extrapolates the distribution function through a scaling derivation

    Args:
        rowHarmonic (int): Index of row (evolved) harmonic
        colHarmonic (int): Index of column (implicit) harmonic
        derivRule (dict): distScalingExtrapolation properties used by this stencil
        decompHarmonics (Union[List[float],None], optional): List of harmonics included in the decomposition of column harmonic
                                                             Note: These are the actual implcit harmonics, and in case of a staggered distribution on the right boundary must all have the same l-number parity . Defaults to None, resulting in all harmonics.
        leftBoundary (bool, optional): Set to true if this stencil corresponds to the left boundary. Defaults to False.
    """

    stencil = {
        "stencilType": "scalingLogicalBoundaryStencil",
        "rowHarmonic": rowHarmonic,
        "colHarmonic": colHarmonic,
        "leftBoundary": leftBoundary,
    }

    if decompHarmonics is not None:
        stencil["includedDecompHarmonics"] = decompHarmonics
    stencil.update(derivRule)

    return stencil


def termMomentStencil(colHarmonic: int, momentOrder: int, termName: str) -> dict:
    """Return stencil which represent taking the moment of a given a term which evolves a single harmonic of a distribution variable.
        The term whose moment is taken should be local in space and cannot perform interpolation if implicit in a distribution variable.

    Args:
        colHarmonic (int): Harmonic evolved by the targeted term
        momentOrder (int): Order of moment to be taken
        termName (str): Name/tag of term in model whose moment is to be taken

    Returns:
        dict: Stencil property dictionary
    """

    stencil = {
        "stencilType": "termMomentStencil",
        "momentOrder": momentOrder,
        "colHarmonic": colHarmonic,
        "termName": termName,
    }

    return stencil


def customFluid1DStencil(
    xStencil: List[int],
    fixedColumnVecs: List[np.ndarray],
    varContColumnVars: Union[List[str], None] = None,
    mbDataColumnVars: Union[List[str], None] = None,
) -> dict:
    """Generate a custom 1D fluid stencil based on a relative stencil and column vector data. NOTE: Does not check the variables or fixed vector for length validity.

    Args:
        xStencil (List[int]): Relative stencil (eg. [-1,0,1] will generate a tridiagonal stencil)
        fixedColumnVecs (list[np.ndarray]): List of vectors, one for each entry in the stencil, corresponding to that column.
        varContColumnVars (Union[List[str],None], optional): List of global fluid variables to multiply the fixed column vectors with, one for each stencil entry (use "none" if any one column should not have a variable associated). Defaults to None.
        mbDataColumnVars (Union[List[str],None], optional): List of modelbound fluid variables to multiply the fixed column vectors with, one for each stencil entry (use "none" if any one column should not have a variable associated). Defaults to None.

    Returns:
        dict: Stencil property dictionary
    """
    assert len(fixedColumnVecs) == len(
        xStencil
    ), "fixedColumnVecs must be of same length as xStencil in customFluid1DStencil"

    stencil = {"stencilType": "customFluid1DStencil", "xStencil": xStencil}

    for i, vec in enumerate(fixedColumnVecs):
        stencil.update({"columnVector" + str(i + 1): vec.tolist()})

    if varContColumnVars is not None:
        assert len(varContColumnVars) == len(
            xStencil
        ), "varContColumnVars must be of same length as xStencil in customFluid1DStencil"
        stencil.update({"columnVarContVars": varContColumnVars})
    else:
        stencil.update({"columnVarContVars": ["none" for _ in xStencil]})
    if mbDataColumnVars is not None:
        assert len(mbDataColumnVars) == len(
            xStencil
        ), "mbDataColumnVars must be of same length as xStencil in customFluid1DStencil"
        stencil.update({"columnMBDataVars": mbDataColumnVars})
    else:
        stencil.update({"columnMBDataVars": ["none" for _ in xStencil]})

    return stencil


def rkIntegrator(order: int) -> dict:
    """Return integrator properties for RK integrator of given order

    Args:
        order (int): RK order (1-4 supported)

    Returns:
        dict: Integrator property dictionary
    """

    integ = {"type": "RK", "order": order}

    return integ


def picardBDEIntegrator(
    maxNonlinIters=100,
    nonlinTol=1.0e-12,
    absTol=1.0,
    convergenceVars: List[str] = [],
    associatedPETScGroup=1,
    use2Norm=False,
    internalStepControl=False,
    initialNumInternalSteps=1,
    stepMultiplier=2,
    stepDecrament=1,
    minNonlinIters=5,
    maxBDERestarts=3,
    relaxationWeight: float = 1.0,
) -> dict:
    """Return integrator properties for Backward Euler integrator with Picard (fixed point) iterations

    Args:
        maxNonlinIters (int, optional): Maximum allowed nonlinear (Picard/fixed point) iterations. Defaults to 100.
        nonlinTol (float, optional): Relative convergence tolerance on 2-norm. Defaults to 1.0e-12.
        absTol (float, optional): Absolute tolerance in machine precision units (epsilon in Fortran - 2.22e-16 for double precision). Defaults to 1.0.
        convergenceVars (List[str], optional): Variables used to check for convergence. Defaults to [], which results in all implicit variables.
        associatedPETScGroup (int, optional): PETSc object group this integrator is associated with. Defaults to 1.
        use2Norm (bool, optional): True if 2-norm should be used (benefits distributions). Defaults to False.
        internalStepControl (bool, optional): True if integrator is allowed to control its internal steps based on convergence. Defaults to False.
        initialNumInternalSteps (int, optional): Initial number of integrator substeps. Defaults to 1.
        stepMultiplier (int, optional): Factor by which to multiply current number of substeps when solve fails. Defaults to 2.
        stepDecrament (int, optional): How much to reduce the current number of substeps if nonlinear iterations are below minNonlinIters. Defaults to 1.
        minNonlinIters (int, optional): Number of nonlinear iterations under which the integrator should attempt to reduce the number of internal steps. Defaults to 5.
        maxBDERestarts (int, optional): Maximum number of solver restarts with step splitting. Defaults to 3. Note that there is a hard limit of 10.
        relaxationWeight (float, optional): Relaxation weight for the Picard iteration (relaxatioWeight * newValues + (1-relaxationWeight)*oldValues). Defaults to 1.0.


    Returns:
        dict: Integrator property dictionary
    """

    integ = {
        "type": "BDE",
        "maxNonlinIters": maxNonlinIters,
        "nonlinTol": nonlinTol,
        "absTol": absTol,
        "convergenceVars": convergenceVars,
        "associatedPETScGroup": associatedPETScGroup,
        "use2Norm": use2Norm,
        "relaxationWeight": relaxationWeight,
        "internalStepControl": {
            "active": internalStepControl,
            "startingNumSteps": initialNumInternalSteps,
            "stepMultiplier": stepMultiplier,
            "stepDecrament": stepDecrament,
            "minNumNonlinIters": minNonlinIters,
            "maxBDERestarts": maxBDERestarts,
        },
    }

    return integ


def CVODEIntegrator(
    relTol=1e-5,
    absTol=1e-10,
    maxGMRESRestarts=0,
    CVODEBBDPreParams: List[int] = [0, 0, 0, 0],
    useAdamsMoulton=False,
    useStabLimitDet=False,
    maxOrder=5,
    maxInternalStep=500,
    minTimestep: float = 0.0,
    maxTimestep: float = 0.0,
    initTimestep: float = 0.0,
) -> dict:
    """Return dictionary with CVODE integrator properties. See https://sundials.readthedocs.io/en/latest/index.html

    Args:
        relTol (float, optional): CVODE solver relative tolerance. Defaults to 1e-5.
        absTol (float, optional): CVODE solver absolute tolerance. Defaults to 1e-10.
        maxGMRESRestarts (int, optional): SPGMR maximum number of restarts. Defaults to 0.
        CVODEBBDPreParams (List[int], optional): BBD preconditioner parameters in order [mudq,mldq,mukeep,mlkeep]. Defaults to [0,0,0,0].
        useAdamsMoulton (bool, optional): If true will use Adams Moulton method instead of the default BDF. Defaults to False.
        useStabLimitDet (bool, optional): If true will use stability limit detection. Defaults to False.
        maxOrder (int, optional): Maximum integrator order (set to BDF default, AM default is 12). Defaults to 5.
        maxInternalStep (int, optional): Maximum number of internal CVODE steps per ReMKiT1D timestep. Defaults to 500.
        minTimestep (float, optional): Minimum allowed internal timestep. Defaults to 0.0.
        maxTimestep (float, optional): Maximum allowed internal timestep. Defaults to 0.0, resulting in no limit.
        initTimestep (float, optional): Initial internal timestep. Defaults to 0.0, letting CVODE decide.

    Returns:
        dict: Integrator property dictionary
    """

    assert len(CVODEBBDPreParams) == 4, "CVODEBBDPreParams must be size 4"
    integ = {
        "type": "CVODE",
        "relTol": relTol,
        "absTol": absTol,
        "maxRestarts": maxGMRESRestarts,
        "CVODEPreBBDParams": CVODEBBDPreParams,
        "CVODEUseAdamsMoulton": useAdamsMoulton,
        "CVODEUseStabLimDet": useStabLimitDet,
        "CVODEMaxOrder": maxOrder,
        "CVODEMaxInternalSteps": maxInternalStep,
        "CVODEMaxStepSize": maxTimestep,
        "CVODEMinStepSize": minTimestep,
        "CVODEInitStepSize": initTimestep,
    }

    return integ


class IntegrationStep:
    """Class containing integration step data"""

    def __init__(
        self,
        integratorTag: str,
        globalStepFraction=1.0,
        allowTimeEvolution=False,
        useInitialInput=False,
        defaultUpdateGroups=[1],
        defaultEvaluateGroups=[1],
        defaultUpdateModelData=True,
    ) -> None:
        """Integration step initialization

        Args:
            integratorTag (str): Tag of the used integrator
            globalStepFraction (float, optional): Fraction of the global timestep used for this step. Defaults to 1.0.
            allowTimeEvolution (bool, optional): If true allows the evolution of the time variable by this step. Defaults to False.
            useInitialInput (bool, optional): If true, will use the initial values of variables at the global timestep instead of the values at the end of the previous integration step. Defaults to False.
            defaultUpdateGroups (list, optional): Default updated groups for models integrated by this step. Defaults to [1].
            defaultEvaluateGroups (list, optional): Default evaluated groups for models integrated by this step. Defaults to [1].
            defaultUpdateModelData (bool, optional): Default model data output option. Defaults to True, which results in updating model data during this step.
        """

        self.__integratorTag__ = integratorTag
        self.__globalStepFraction__ = globalStepFraction
        self.__allowTimeEvolution__ = allowTimeEvolution
        self.__useInitialInput__ = useInitialInput
        self.__defaultUpdateGroups__ = defaultUpdateGroups
        self.__defaultEvaluateGroups__ = defaultEvaluateGroups
        self.__defaultUpdateModelData__ = defaultUpdateModelData

        self.__evolvedModels__: List[str] = []
        self.__evolvedModelProperties__: Dict[str, object] = {}

    @property
    def integratorTag(self):
        return self.__integratorTag__

    @property
    def globalStepFraction(self):
        return self.__globalStepFraction__

    @property
    def allowTimeEvolution(self):
        return self.__allowTimeEvolution__

    @property
    def useInitialInput(self):
        return self.__useInitialInput__

    @property
    def defaultUpdateGroups(self):
        return self.__defaultUpdateGroups__

    @property
    def defaultEvaluateGroups(self):
        return self.__defaultEvaluateGroups__

    @property
    def defaultUpdateModelData(self):
        return self.__defaultUpdateModelData__

    @property
    def evolvedModels(self):
        return self.__evolvedModels__

    @property
    def evolvedModelProperties(self):
        return self.__evolvedModelProperties__

    def addModel(
        self,
        modelTag: str,
        updateGroups: Union[None, List[int]] = None,
        evaluateGroups: Union[None, List[int]] = None,
        updateModelData: Union[None, bool] = None,
    ) -> None:
        """Add a model to be integrated during this step

        Args:
            modelTag (str): Tag of the model to be added
            updateGroups (Union[None,List[int]], optional): Optional non-default updated groups for this model. Defaults to None.
            evaluateGroups (Union[None,List[int]], optional): Optional non-default evaluated groups for this model. Defaults to None.
            updateModelData (Union[None,bool], optional): Optional non-default model data update option for this model. Defaults to None.
        """

        self.__evolvedModels__.append(modelTag)

        self.__evolvedModelProperties__[modelTag] = {
            "groupIndices": (
                self.__defaultEvaluateGroups__
                if evaluateGroups is None
                else evaluateGroups
            ),
            "internallyUpdatedGroups": (
                self.__defaultUpdateGroups__ if updateGroups is None else updateGroups
            ),
            "internallyUpdateModelData": (
                self.__defaultUpdateModelData__
                if updateModelData is None
                else updateModelData
            ),
        }

    def dict(self) -> dict:
        """Return ReMKiT1D-readable dictionary object

        Returns:
            dict: Integration step property dictionary
        """

        step = {
            "integratorTag": self.__integratorTag__,
            "evolvedModels": self.__evolvedModels__,
            "globalStepFraction": self.__globalStepFraction__,
            "allowTimeEvolution": self.__allowTimeEvolution__,
            "useInitialInput": self.__useInitialInput__,
        }

        step.update(self.__evolvedModelProperties__)

        return step


class VarlikeModelboundData:
    """Variable-like modelbound data class used for easier instantiation of varlike modelbound data for models"""

    def __init__(self) -> None:
        self.__properties__: Dict[str, object] = {
            "modelboundDataType": "varlikeData",
            "dataNames": [],
        }

    def addVariable(
        self,
        name: str,
        derivationRule: dict,
        isDistribution=False,
        isScalar=False,
        isSingleHarmonic=False,
        isDerivedFromOtherData=False,
        priority=0,
    ) -> None:
        """Add modelbound variable to modelbound data properties

        Args:
            name (str): Name of the variable
            derivationRule (dict): Rule used to derive the variable
            isDistribution (bool, optional): True if the variable is a distribution. Defaults to False.
            isScalar (bool, optional): True if the variable is a scalar. Defaults to False.
            isSingleHarmonic (bool, optional): True if the variable is a single harmonic distribution. Defaults to False.
            isDerivedFromOtherData (bool, optional): True if the variable is derived from other modelbound variables instead of the global variables. Defaults to False.
            priority (int, optional): Derivation priority. Defaults to 0.
        """

        cast(List[str], self.__properties__["dataNames"]).append(name)
        self.__properties__[name] = {
            "isDistribution": isDistribution,
            "isScalar": isScalar,
            "isSingleHarmonic": isSingleHarmonic,
            "isDerivedFromOtherData": isDerivedFromOtherData,
            "derivationPriority": priority,
        }
        cast(Dict[str, object], self.__properties__[name]).update(derivationRule)

    def dict(self) -> dict:
        return self.__properties__


def scalingTimestepController(
    reqVarNames: List[str],
    reqVarPowers: List[str],
    multConst=1.0,
    rescaleTimestep=True,
    useMaxVal=False,
) -> dict:
    """Return properties of scaling timestep controller using variable powers

    Args:
        reqVarNames (List[str]): Names of required variables to calculate the scaling factor/timestep
        reqVarPowers (List[str]): Powers corresponding to required variables
        multConst (float, optional): Optional multiplicative constant. Defaults to 1.0.
        rescaleTimestep (bool, optional): Set to false if this controller should ignore the global/passed timestep and calculate a fixed value. Defaults to True.
        useMaxVal (bool, optional): Use maximum value of scaling factor along x instead of the minimum value. Defaults to False.

    Returns:
        dict: Timestep controller property dictionary
    """

    timestepControllerOptions = {
        "rescaleTimestep": rescaleTimestep,
        "requiredVarNames": reqVarNames,
        "requiredVarPowers": reqVarPowers,
        "multConst": multConst,
        "useMaxVal": useMaxVal,
    }

    return timestepControllerOptions


def modelboundLBCData(
    ionCurrentVar: str,
    derivationRule: dict,
    totalCurrentVar="none",
    bisTol: float = 1e-12,
    leftBoundary=False,
) -> dict:
    """Return dictionary representing modelbound date for the logical boundary condition for electrons

    Args:
        ionCurrentVar (str): Scalar variable name representing the ion current at the boundary
        derivationRule (dict): distScalingExtrapolation derivation rule used for this boundary
        totalCurrentVar (str, optional): Scalar variable name resenting the total current at the boundary. Defaults to "none", resulting in 0 total current.
        bisTol (float, optional): Bisection tolerance for the false position method used in LBC calculation. Defaults to 1e-12.
        leftBoundary (bool, optional): Set to true if this data handles the left boundary. Defaults to False.

    Returns:
        dict: Modelbound data property dictionary
    """

    mbData = {
        "modelboundDataType": "modelboundLBCData",
        "ionCurrentVarName": ionCurrentVar,
        "totalCurrentVarName": totalCurrentVar,
        "bisectionTolerance": bisTol,
        "leftBoundary": leftBoundary,
    }

    mbData.update(derivationRule)

    return mbData
