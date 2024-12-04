from typing import Union, List, Dict, cast, Tuple,Optional
from .variable_container import VariableContainer,Variable,MultiplicativeArgument
from .derivations import Derivation, Textbook, GenericDerivation
from typing_extensions import Self
from .grid import Profile, Grid
import numpy as np
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy,copy
from math import isclose
import pylatex as tex
from .tex_parsing import numToScientificTex

#TODO: docs

class ModelboundData(ABC):

    @abstractmethod
    def addLatexToDoc(self,doc:tex.Document,**kwargs):
        pass

    @abstractmethod
    def dict(self) -> dict:
        pass

    @abstractmethod
    def __getitem__(self,key:str) -> Variable:
        pass

    @property
    @abstractmethod
    def varNames(self) -> List[str]:
        pass

    def registerDerivs(self,container:Textbook):
        pass

class Term(ABC):

    def __init__(
        self,
        name: str,
        evolvedVar: Union[Variable,None]=None,
        implicitGroups: List[int] = [],
        generalGroups: List[int] = [],
    ) -> None:
        super().__init__()
        self.__name__ = name
        self.__evolvedVar__ = evolvedVar
        self.__implicitGroups__ = implicitGroups
        self.__generalGroups__ = generalGroups

    @property 
    def name(self):
        return self.__name__

    @name.setter
    def name(self,name:str):
        self.__name__ = name

    def rename(self,name:str):
        newTerm = deepcopy(self)
        newTerm.__name__ = name 
        return newTerm
    
    @property
    def evolvedVar(self):
        return self.__evolvedVar__

    @evolvedVar.setter
    def evolvedVar(self,name:Variable):
        self.__evolvedVar__ = name

    def withEvolvedVar(self,name:str) -> Self:
        self.evolvedVar = name
        return self

    @property
    def implicitGroups(self):
        return self.__implicitGroups__

    @property
    def generalGroups(self):
        return self.__generalGroups__

    def regroup(self,implicitGroups:List[int]=[],generalGroups:List[int]=[]):
        self.__implicitGroups__ = implicitGroups 
        self.__generalGroups__ = generalGroups 
        return self

    @abstractmethod
    def dict(self) -> dict:
        pass

    def checkTerm(self, varCont: VariableContainer, mbData:Union[ModelboundData,None] = None):

        assert len(self.implicitGroups) or len(self.generalGroups), "Both implicit and general groups empty in term "+self.name

        assert self.__evolvedVar__ is not None, "Term evolvedVar not set"

        assert self.__evolvedVar__.name in varCont.varNames, (
            "Evolved variable "
            + self.__evolvedVar__.name
            + " not registered in used variable container"
        )

    @abstractmethod 
    def latex(self,*args,**kwargs) -> str:
        pass

    @abstractmethod
    def __add__(self,rhs:Self):
        pass

    @abstractmethod
    def registerDerivs(self,container:Textbook):
        pass

class TermCollection:

    def __init__(self,evolvedVar:Variable,modelName:str,latexName:str,derivativeTex:str="\\partial"):
        self.__evolvedVar__ = evolvedVar 
        self.__modelName__ = modelName
        self.__latexName__ = latexName
        self.__derivativeTex__ = derivativeTex
        self.__terms__:List[Term] = []

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
        return sum([term.implicitGroups for term in self.__terms__], [])

    @property
    def activeGeneralGroups(self):
        return sum([term.generalGroups for term in self.__terms__], [])

    def __getitem__(self,key:str):
        if key not in self.termNames: 
            raise KeyError()
        return self.__terms__[self.termNames.index(key)]

    def __setitem__(self,key:str,term:Term):
        if key not in self.termNames: 
            self.__terms__.append(term)
        else:
            self.__terms__[self.termNames.index(key)] = term

    def __delitem__(self,key:str):
        if key not in self.termNames: 
            raise KeyError()
        del self.__terms__[self.termNames.index(key)]

    def dict(self):
        return {term.name:term.dict() for term in self.__terms__}

    def addLatexToDoc(self,doc:tex.Alignat,latexRemap:Dict[str,str]={}):
        evolvedVarTex= latexRemap[self.__evolvedVar__.name] if self.__evolvedVar__.name in latexRemap else "\\text{"+self.__evolvedVar__.name.replace("_","\_")+"}"
        doc.append("\\left(\\frac{"+self.__derivativeTex__+" "+evolvedVarTex+"}{"+self.__derivativeTex__+" t}\\right)_{"+self.__latexName__+"} &= ")

        for i,term in enumerate(self.__terms__):
            buffer = ("+ " if i>0 else " ") +term.latex(latexRemap=latexRemap)
            if i == 0:
                doc.append( (buffer[1:] if buffer.startswith("+  -") else buffer)+"\\\\")
            else:
                doc.extend(["&"+( buffer[1:] if buffer.startswith("+  -") else buffer)+"\\\\"])

    def __add__(self,rhs:Union[Term,Self]) -> Self:
        assert isinstance(rhs,(Term,TermCollection)), "cannot add non-Terms to term collection"
        if isinstance(rhs,Term):
            assert rhs.name not in self.termNames, "duplicate term name in TermCollection - use Term.rename()"
            assert rhs.evolvedVar is None or rhs.evolvedVar.name == self.__evolvedVar__.name, "Cannot add Term to TermCollection if it evolves a variable different from the collection"
            newCollection = deepcopy(self)
            newCollection.__terms__.append(deepcopy(rhs))
            newCollection.__terms__[-1].evolvedVar=self.__evolvedVar__
            
        if isinstance(rhs,TermCollection):
            newCollection = deepcopy(self)
            for term in rhs.terms:
                newCollection+=term.withEvolvedVar(self.evolvedVar)

        return newCollection

    def __sub__(self,rhs:Union[Term]):

        assert isinstance(rhs,(Term,TermCollection)), "cannot add (negative) non-Terms to term collection"
        if isinstance(rhs,Term):
            assert rhs.name not in self.termNames, "duplicate term name in TermCollection - use Term.rename()"
            assert rhs.evolvedVar is None or rhs.evolvedVar.name == self.__evolvedVar__.name, "Cannot add (negative) Term to TermCollection if it evolves a variable different from the collection"
            newCollection = deepcopy(self)
            newCollection.__terms__.append(deepcopy(-1*rhs))
            newCollection.__terms__[-1].evolvedVar=self.__evolvedVar__
            
        if isinstance(rhs,TermCollection):
            newCollection = deepcopy(self)
            for term in rhs.terms:
                newCollection-=term.withEvolvedVar(self.evolvedVar)

        return newCollection

    def filterByGroup(self,group:int,general:bool=False):

        newCollection = TermCollection(self.evolvedVar,modelName=self.modelName,latexName=self.__latexName__,derivativeTex=self.__derivativeTex__)

        for term in self.__terms__:
            if general: 
                if group in term.generalGroups:
                    newCollection+=term 
            else: 
                if group in term.implicitGroups:
                    newCollection+=term 

        return newCollection

    def checkTerms(self,varCont:VariableContainer,mbData:Union[ModelboundData,None]=None):

        for term in self.__terms__:
            print("   Checking term " + term.name)
            term.checkTerm(varCont,mbData)

    def registerDerivs(self,container:Textbook):
        for term in self.__terms__:
            term.registerDerivs(container)
            
class VarData:

    def __init__(
        self,
        reqRowVars: MultiplicativeArgument = MultiplicativeArgument(),
        reqColVars: MultiplicativeArgument = MultiplicativeArgument(),
        reqMBRowVars: MultiplicativeArgument = MultiplicativeArgument(),
        reqMBColVars: MultiplicativeArgument = MultiplicativeArgument(),
    ) -> None:

        self.__reqRowVars___ = list(reqRowVars.argMultiplicity.keys())
        self.__reqRowPowers___ = list(reqRowVars.argMultiplicity.values())
        self.__reqColVars___ = list(reqColVars.argMultiplicity.keys())
        self.__reqColPowers___ = list(reqColVars.argMultiplicity.values())
        self.__reqMBRowVars___ = list(reqMBRowVars.argMultiplicity.keys())
        self.__reqMBRowPowers___ = list(reqMBRowVars.argMultiplicity.values())
        self.__reqMBColVars___ = list(reqMBColVars.argMultiplicity.keys())
        self.__reqMBColPowers___ = list(reqMBColVars.argMultiplicity.values())

    def checkRowColVars(
        self, varCont: VariableContainer, rowVarOnDual=False, colVarOnDual=False, mbData:Union[ModelboundData,None] = None
    ):
        """Check whether required variables exist in the variable container and are on the correct grids

        Args:
            varCont (VariableContainer): Variable container used to check
            rowVarOnDual (bool, optional): True if the row variables should be on the dual grid. Defaults to False.
            colVarOnDual (bool, optional): True if the column variables should be on the dual grid. Defaults to False.
        """

        for var in self.__reqRowVars___ :
            assert var in varCont.varNames, (
                "Required row variable " + var + " not found in used variable container"
            )

            if not varCont[var].isScalar:
                if (
                    varCont[var].isOnDualGrid
                    is not rowVarOnDual
                ):
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
            assert mbData is not None, "No modelbound data available when modelbound variables required by a term"

            for var  in self.__reqMBRowVars___:
                assert var in mbData.varNames, (
                    "Required row variable " + var + " not found in used modelbound data"
                )

                if not mbData[var].isScalar:
                    if (
                        mbData[var].isOnDualGrid
                        is not rowVarOnDual
                    ):
                        warnings.warn(
                            "Variable "
                            + var
                            + " appears in modelbound required row variables for evolved variable on "
                            + ("dual" if rowVarOnDual else "regular")
                            + " grid but doesn't live on that grid - ignore if evolved variable is a distribution"
                        )

            for var in self.__reqMBColVars___:
                assert var in mbData.varNames, (
                    "Required column variable "
                    + var
                    + " not found in used modelbound"
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

#TODO: rework this class
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

    def latex(self) -> str:
        return "T"


class AbstractStencil(ABC):

    @abstractmethod
    def latex(self,arg:MultiplicativeArgument,**kwargs) -> str:
        pass

    @abstractmethod
    def dict(self) -> dict:
        pass

    @abstractmethod 
    def __call__(self, *args:MultiplicativeArgument, **kwds) -> Term:
        pass 

    def registerDerivs(self,container:Textbook):
        pass

class MatrixTerm(Term):
    def __init__(
        self,
        name:str,
        stencil:AbstractStencil,
        evolvedVar:Union[Variable,None] =None,
        implicitVar:Union[Variable,None] =None,
        profiles:Dict[str,Profile] = {},
        R:MultiplicativeArgument=MultiplicativeArgument(),
        modelboundR:MultiplicativeArgument=MultiplicativeArgument(),
        C:MultiplicativeArgument=MultiplicativeArgument(),
        modelboundC:MultiplicativeArgument=MultiplicativeArgument(),
        T:Union[TimeSignalData,None]=None,
        implicitGroups=[1],
        generalGroups: List[int] = [],
        **kwargs,
    ) -> None:

        super().__init__(name,evolvedVar, implicitGroups, generalGroups)

        self.__stencil__ = stencil

        self.__implicitVar__ = implicitVar
        assert all([k in ["X","H","V"] for k in profiles.keys()]), "Profiles in MatrixTerm constructor must have keys X,H, or V"
        self.__profiles__ = profiles

        self.__R__ = R 
        self.__C__ = C 
        self.__modelboundR__ = modelboundR
        self.__modelboundC__ = modelboundC
        self.__T__ = T
        self.__skipPattern__ = kwargs.get("skipPattern",False)
        self.__fixedMatrix__ = kwargs.get("fixedMatrix",False)
        self.__copyTermName__: Union[str, None] = kwargs.get("copyTermName",None)
        self.__evaluatedTermGroup__:int = kwargs.get("evaluatedTermGroup",0)
        self.__constLatex__:Union[str,None] = kwargs.get("constLatex",None)

    @property
    def implicitVar(self):
        return self.__implicitVar__ if self.__implicitVar__ is not None else self.evolvedVar

    @property 
    def stencil(self):
        return self.__stencil__

    @property
    def constLatex(self):
        return self.__constLatex__ 

    @constLatex.setter
    def constLatex(self,expression:str):
        self.__constLatex__ = expression

    @property
    def fixedMatrix(self):
        return self.__fixedMatrix__

    def withFixedMatrix(self,fixed=True):
        newTerm = deepcopy(self)
        newTerm.__fixedMatrix__ = fixed 
        return newTerm

    @property
    def skipPattern(self):
        return self.__skipPattern__

    def withSkippingPattern(self,skip=True):
        newTerm = deepcopy(self)
        newTerm.__skipPattern__ = skip 
        return newTerm

    @property 
    def multConst(self):
        return self.__R__.scalar*self.__modelboundR__.scalar

    def checkTerm(self, varCont: VariableContainer, mbData:Union[ModelboundData,None] = None):
        """Perform consistency check on term

        Args:
            varCont (VariableContainer): Variable container to be used with this term
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

        vData = VarData(self.__R__,self.__C__,self.__modelboundR__,self.__modelboundC__)

        vData.checkRowColVars(varCont, rowVarOnDual, colVarOnDual,mbData)

    def dict(self):
        """Returns dictionary form of GeneralMatrixTerm to be used in json output

        Returns:
            dict: Dictionary form of GeneralMatrixTerm to be used as individual custom term properties
        """
        assert self.evolvedVar is not None, "Called dict() on MatrixTerm without setting evolved variable"
        assert self.__implicitVar__ is not None, "Called dict() on MatrixTerm without setting implicit variable"
    
        gTerm = {
            "termType": "matrixTerm",
            "evolvedVar": self.evolvedVar.name,
            "implicitVar": self.__implicitVar__.name,
            "spatialProfile": self.__profiles__["X"].data.tolist() if "X" in self.__profiles__ else [],
            "harmonicProfile": self.__profiles__["H"].data.tolist() if "H" in self.__profiles__ else [],
            "velocityProfile": self.__profiles__["V"].data.tolist() if "V" in self.__profiles__ else [],
            "evaluatedTermGroup": self.__evaluatedTermGroup__,
            "implicitGroups": self.implicitGroups,
            "generalGroups": self.generalGroups,
            "customNormConst": {"multConst":self.multConst},
            "timeSignalData": self.__T__.dict() if self.__T__ is not None else TimeSignalData().dict(),
            "varData": VarData(self.__R__,self.__C__,self.__modelboundR__,self.__modelboundC__).dict(),
            "stencilData": self.__stencil__.dict(),
            "skipPattern": self.__skipPattern__,
            "fixedMatrix": self.__fixedMatrix__,
        }

        if self.__copyTermName__ is not None:
            gTerm["multCopyTermName"] = self.__copyTermName__

        return gTerm

    def latex(self, *args, **kwargs):
        latexRemap:Dict[str,str] = kwargs.get("latexRemap",{})
        result = " " +self.__stencil__.latex(self.implicitVar*self.__C__*self.__modelboundC__,latexRemap=latexRemap)
        result = " " + self.__modelboundR__.latex(latexRemap) + result 
        result = " " + self.__R__.latex(latexRemap) + result 
        for key in self.__profiles__:
            result = " " + self.__profiles__[key].latex() + result 
        if self.__T__ is not None: result = " " + self.__T__.latex() + result 
        constRepr = numToScientificTex(self.multConst,removeUnity=True)
        result = " " + (constRepr if self.constLatex is None else self.constLatex) + result
        return result

    def __neg__(self):
        newMat = deepcopy(self)
        newMat.__R__*=-1
        return newMat

    def __rmul__(self,lhs:Union[float,int,Profile,TimeSignalData,Variable,MultiplicativeArgument]) -> Self:
        if isinstance(lhs,(int,float)):
            newMat = deepcopy(self)
            newMat.__R__*=lhs 
        if isinstance(lhs,Profile):
            newMat = deepcopy(self)
            if lhs.dim not in newMat.__profiles__:
                newMat.__profiles__[lhs.dim] = lhs
            else: 
                newMat.__profiles__[lhs.dim] = Profile(lhs.data*newMat.__profiles__[lhs.dim].data,lhs.dim,latexName=lhs.latex()+newMat.__profiles__[lhs.dim].latex())
        if isinstance(lhs,TimeSignalData):
            assert self.__T__ is None, "Cannot multiply MatrixTerm that already has explicit time dependece by a TimeSignal"
            newMat = deepcopy(self)
            newMat.__T__=lhs 
        if isinstance(lhs,(Variable,MultiplicativeArgument)):
            newMat = deepcopy(self)
            newMat.__R__= lhs*newMat.__R__

        return newMat

    def __matmul__(self,rhs:Union[Variable,MultiplicativeArgument]) -> Self: 
        if isinstance(rhs,(Variable,MultiplicativeArgument)):
            newMat = deepcopy(self)
            newMat.__modelboundR__= rhs*newMat.__modelboundR__ 

        return newMat

    def __add__(self,rhs:Term) -> TermCollection:
        newCollection = TermCollection(self.evolvedVar,"","")
        newCollection+=self 
        newCollection+=rhs 

        return newCollection

    def __sub__(self,rhs:Term) -> TermCollection:
        newCollection = TermCollection(self.evolvedVar,"","")
        newCollection+=self 
        newCollection+=-rhs 

        return newCollection

    def registerDerivs(self, container: Textbook):
        self.__stencil__.registerDerivs(container)


class Stencil(AbstractStencil):

    def __init__(self,latexTemplate:Union[str,None]=None,properties:Union[Dict[str,object],None]=None):
        super().__init__()
        if latexTemplate is not None:
            assert "$0" in latexTemplate, "Stencil latexTemplate argument must be specified"
        self.__latexTemplate__ = latexTemplate
        self.__properties__ = properties

    def dict(self)->dict:
        assert self.__properties__ is not None, "Stencil default dict properties not provided"
        return self.__properties__
    
    def __call__(self, *args:Union[MultiplicativeArgument,Variable], **kwargs) -> MatrixTerm:

        assert len(args) == 1 or len(args) == 2, "Stencil __call__ must have 1 or 2 args"
        C = args[0] if isinstance(args[0],MultiplicativeArgument) else MultiplicativeArgument((args[0],1.0))
        modelboundC = MultiplicativeArgument() if len(args)==1 else (args[1] if isinstance(args[1],MultiplicativeArgument) else MultiplicativeArgument((args[1],1.0)))
        assert len(C.args) > 0, "First arg in Stencil call must have at least one variable"
        for arg in args:
            if isinstance(arg,MultiplicativeArgument):
                assert arg.scalar == 1, "Stencils must act on MultiplicativeArguments without a non-trivial scalar component"
        return MatrixTerm("unnamed_term",self,implicitVar=C.firstArg,C=C/C.firstArg,modelboundC=modelboundC)

    def latex(self, arg, **kwargs):
        assert self.__latexTemplate__ is not None, "latex() on Stencil called without a default latexTemplate set"
        latexRemap:Dict[str,str] = kwargs.get("latexRemap",{})
        return self.__latexTemplate__.replace("$0",arg.latex(latexRemap))

class DiagonalStencil(Stencil):

    def __init__(self, evolvedXCells: Optional[List[int]] = None,
                       evolvedHarmonics: Optional[List[int]] = None,
                       evolvedVCells: Optional[List[int]] = None):
        properties = {
            "stencilType": "diagonalStencil",
            "evolvedXCells": evolvedXCells if evolvedXCells is not None else [],
            "evolvedHarmonics": evolvedHarmonics if evolvedHarmonics is not None else [],
            "evolvedVCells": evolvedVCells if evolvedVCells is not None else [],
        }
        super().__init__("$0",properties)

class DerivationTerm(Term):
    """Derivation-based explicit term options used to construct custom models. The result of evaluating a derivation term is the result of the derivation optionally multiplied by a modelbound variable. Does not support evolving distributions."""

    def __init__(
        self,
        name:str,
        derivation: Derivation,
        derivationArgs:List[str],
        evolvedVar: Union[Variable,None]=None,
        mbVar: Union[Variable, None] = None,
        generalGroups=[1],
    ) -> None:
        """Derivation term constructor

        Args:
            evolvedVar (str): Name of the evolved variable. Distributions not supported.
            derivationRule (dict): Derivation rule containing name and required variables.
            mbVar (Union[str,None], optional): Optional modelbound variable. Defaults to None.
            generalGroups (list, optional): General groups this term belongs to within its model. Defaults to [1].
        """

        super().__init__(name,evolvedVar, [], generalGroups)

        self.__derivation__ = derivation
        self.__derivationArgs__ = derivation.fillArgs(derivationArgs)
        self.__mbVar__ = mbVar

    def checkTerm(self, varCont: VariableContainer, mbData:Union[ModelboundData,None] = None):
        """Perform consistency check on term, including the required variables

        Args:
            varCont (VariableContainer): Variable container to be used with this term
        """

        super().checkTerm(varCont)

        for name in self.__derivationArgs__:
            assert name in varCont.varNames, (
                "Required derivation variable "
                + name
                + " not registered in used variable container"
            )

        if self.__mbVar__ is not None:
            assert mbData is not None, "Modelbound variable present in derivation term when no mbData passed"
            assert self.__mbVar__.name in mbData.varNames, "Variable "+self.__mbVar__.name +" not in modelbound data"

            if (
                    self.__mbVar__.isOnDualGrid
                    is not self.evolvedVar.isOnDualGrid
                ):
                    warnings.warn(
                        "Variable "
                        + self.__mbVar__.name
                        + " appears in required row variables for evolved variable on "
                        + ("dual" if self.evolvedVar.isOnDualGrid else "regular")
                        + " grid but doesn't live on that grid"
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

        gTerm.update({"ruleName": self.__derivation__.name, "requiredVarNames": self.__derivationArgs__})

        return gTerm

    def latex(self, *args, **kwargs):
        latexRemap:Dict[str,str] = kwargs.get("latexRemap",{})
        remappedArgs = (latexRemap[arg] if arg in latexRemap else "\\text{"+arg.replace("_","\_")+"}" for arg in self.__derivationArgs__)
        result=" "+self.__derivation__.latex(*remappedArgs)
        if self.__mbVar__ is not None:
            result = latexRemap[self.__mbVar__.name] if self.__mbVar__.name in latexRemap else "\\text{"+self.__mbVar__.name.replace("_","\_")+"} "+result 
        return result

    def __add__(self,rhs:Term) -> TermCollection:
        newCollection = TermCollection(self.evolvedVar,"","")
        newCollection+=self 
        newCollection+=rhs 

        return newCollection

    def registerDerivs(self, container: Textbook):
        container.register(self.__derivation__,ignoreDuplicates=True)

class TermGenerator(ABC):
    """Term generator class used to track the term groups into which the generators will put their turns"""

    def __init__(
        self,
        name: str,
        implicitGroups: List[int] = [1],
        generalGroups: List[int] = [],
    ) -> None:
        super().__init__
        self.__name__ = name
        self.__implicitGroups__ = implicitGroups
        self.__generalGroups__ = generalGroups

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
        """Produce dictionary form of TermGenerator

        Returns:
            dict: Dictionary ready to be added to the config file
        """
        tgDict: Dict[str, object] = {
            "implicitGroups": self.__implicitGroups__,
            "generalGroups": self.__generalGroups__,
        }

        return tgDict

    @abstractmethod
    def addLatexToDoc(self,doc:tex.Document,**kwargs):
        pass

    @property
    @abstractmethod
    def evolvedVars(self) -> List[str]:
        pass 

    @abstractmethod
    def onlyEvolving(self,*args:Variable) -> Self:
        pass
class DDT:

    def __init__(self,modelName:str,modelLatexName:str):
        self.__termCollections__:List[TermCollection] = []
        self.__modelName__ = modelName
        self.__modelLatexName__ = modelLatexName

    def __getitem__(self, var:Variable):
        if var.name not in self.evolvedVars: 
            self.__termCollections__.append(TermCollection(var,self.__modelName__,self.__modelLatexName__
            ))
            return self.__getitem__(var)
        ind = self.evolvedVars.index(var.name)
        return self.__termCollections__[ind]
    
    def __setitem__(self, var:Variable,tc:TermCollection):
        if var.name not in self.evolvedVars: 
            self.__termCollections__.append(TermCollection(var,self.__modelName__,self.__modelLatexName__
            ))
            self.__setitem__(var,tc)
            return
        ind = self.evolvedVars.index(var.name)
        self.__termCollections__[ind] = tc

    def __delitem__(self,var:Variable):
        if var.name not in self.evolvedVars: 
            raise KeyError()
        del self.__termCollections__[self.evolvedVars.index(var.name)]

    @property
    def evolvedVars(self):
        return [tc.evolvedVar.name for tc in self.__termCollections__]

    def registerDerivs(self,container:Textbook):
        for tc in self.__termCollections__:
            tc.registerDerivs(container)

class Model:
    """Model object property container"""

    def __init__(self, name: str,latexName: Optional[str]=None,isIntegrable=True) -> None:
        self.__name__ = name
        self.__latexName__ = latexName if latexName is not None else "\\text{"+name.replace("_","\_")+"}"
        self.__modelboundData__: Union[ModelboundData,None] = None
        self.__termGenerators__:List[TermGenerator] = []
        self.__isIntegrable__ = isIntegrable        

        self.ddt = DDT(name,self.__latexName__)

    @property
    def name(self):
        return self.__name__

    def rename(self,name:str,latexName:Union[str,None]=None):
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
            evolvedVars+=tg.evolvedVars
        
        return list(set(evolvedVars))

    @property
    def mbData(self):
        return self.__modelboundData__

    @property
    def activeImplicitGroups(self):
        activeGroups = sum([tc.activeImplicitGroups for tc in self.ddt.__termCollections__], [])
        activeGroups += sum(
            [
                tg.implicitGroups
                for tg in self.__termGenerators__
            ],
            [],
        )

        return list(set(activeGroups))

    @property
    def activeGeneralGroups(self):
        activeGroups = sum([tc.activeGeneralGroups for tc in self.ddt.__termCollections__], [])
        activeGroups += sum(
            [
                tg.generalGroups
                for tg in self.__termGenerators__
            ],
            [],
        )

        return list(set(activeGroups))

    @property
    def isIntegrable(self):
        return self.__isIntegrable__

    @isIntegrable.setter
    def isIntegrable(self,integrable:bool):
        self.__isIntegrable__ = integrable

    def onlyEvolving(self,*args:Variable) ->Self: 
        newModel = Model(self.name,self.__latexName__,self.isIntegrable)

        for arg in args:
            if arg.name in self.evolvedVars:
                newModel.ddt[arg] = self.ddt[arg]

        argNames = [arg.name for arg in args]
        for tg in self.__termGenerators__:
            if any(name in tg.evolvedVars for name in argNames):
                newModel.addTermGenerator(tg.onlyEvolving(*args))

        newModel.setModelboundData(self.mbData)

        return newModel

    def addTerm(self, termTag: str, term: Term):
        assert term.evolvedVar is not None, "Cannot add Term without evolved variable directly"
        
        self.ddt[term.evolvedVar] += term.rename(termTag)

    def addTermGenerator(
        self, generator:TermGenerator
    ):
        assert generator.name not in [tg.name for tg in self.__termGenerators__], "Term generator with name "+generator.name + " already in model "+self.name

        self.__termGenerators__.append(generator)

    def setModelboundData(self, mbData: ModelboundData):
        self.__modelboundData__ = mbData

    def addLatexToDoc(self,doc:tex.Document,**kwargs):
        latexRemap:Dict[str,str] = kwargs.get("latexRemap",{})
        with doc.create(tex.Subsection(tex.NoEscape("$"+self.__latexName__+"$"))):
            with doc.create(tex.Subsubsection("Equation contributions")):
                for tc in self.ddt.__termCollections__:
                    if len(tc.termNames):
                        with doc.create(tex.Alignat(numbering=False,escape=False)) as agn:
                            tc.addLatexToDoc(agn,latexRemap)

            if self.__modelboundData__ is not None:
                with doc.create(tex.Subsubsection("Modelbound data")):
                    self.__modelboundData__.addLatexToDoc(doc,**kwargs)

            if len(self.__termGenerators__):
                with doc.create(tex.Subsubsection("Term generators")):
                    for tg in self.__termGenerators__:
                        tg.addLatexToDoc(doc,**kwargs)

    def checkTerms(self, varCont: VariableContainer):
        """Check terms in this model for consistency

        Args:
            varCont (VariableContainer): Variable container to be used in this check
        """
        print("Checking terms in model " + self.name + ":")
        for tc in self.ddt.__termCollections__:
            tc.checkTerms(varCont,self.mbData)

    def dict(self):
        """Returns dictionary form of CustomModel to be used in json output

        Returns:
            dict: Dictionary form of CustomModel to be used to update model properties
        """
        cModel = {
                "type": "customModel",
                "termTags": sum([tc.termNames for tc in self.ddt.__termCollections__],[]),
                "termGenerators": {
                    "tags": [tg.name for tg in self.__termGenerators__]
                },
            }
        if self.__modelboundData__ is not None:
            cModel["modelboundData"] = self.__modelboundData__.dict()
        cModel["termGenerators"].update(
            {tg.name: tg.dict() for tg in self.__termGenerators__}
        )
        for tc in self.ddt.__termCollections__:
            cModel.update(tc.dict())

        return cModel

    def registerDerivs(self,container:Textbook):
        self.ddt.registerDerivs(container)
        if self.mbData is not None:
            self.__modelboundData__.registerDerivs(container)

class ModelCollection: 

    def __init__(self,*args:Model):
        self.__models__:List[Model] = list(args)

    @property
    def modelNames(self):
        return [model.name for model in self.__models__]

    @property 
    def models(self):
        return self.__models__

    def __getitem__(self,key:str):
        if key not in self.modelNames: 
            raise KeyError()
        return self.models[self.modelNames.index(key)]

    def __setitem__(self,key:str,model:Model):
        if key not in self.modelNames: 
            self.models.append(model)
        else:
            self.models[self.modelNames.index(key)] = model

    def __delitem__(self,key:str):
        if key not in self.modelNames: 
            raise KeyError()
        del self.models[self.modelNames.index(key)]

    def add(self,*args:Model):
        for arg in args:
            assert arg.name not in self.modelNames, "Attempted to add duplicate model to ModelCollection - use key access if you wish to overwrite an existing model"

        self.__models__+=list(args)

    def dict(self) -> dict: 

        modelDict = {"tags":[m.name for m in self.__models__]}

        for model in self.__models__:
            modelDict.update({model.name:model.dict()})

        return modelDict 

    def addLatexToDoc(self,doc:tex.Document,**kwargs):
        with doc.create(tex.Section("Models")):
            for model in self.__models__:
                model.addLatexToDoc(doc,**kwargs)

    def onlyEvolving(self,*args:Variable) -> Self:
        newCollection = ModelCollection()

        for model in self.__models__:
            if any(arg.name in model.evolvedVars for arg in args):
                newCollection.add(model.onlyEvolving(*args))

        return newCollection

    def checkModels(self,varCont:VariableContainer):
        for model in self.__models__:
            model.checkTerms(varCont)

    def numGroups(self) -> Tuple[int,int]:

        numImplicitGroups = max(max([1]+model.activeImplicitGroups) for model in self.__models__)
        numGeneralGroups = max(max([1]+model.activeGeneralGroups) for model in self.__models__)

        return numImplicitGroups,numGeneralGroups

    def getTermsThatEvolveVar(self,var:Variable) -> List[Tuple[str,str]]:

        result = []
        for model in self.models:
            if var.name in model.evolvedVars:
                for term in model.ddt[var].termNames:
                    result.append((model.name,term))

        return result

    def registerDerivs(self,container:Textbook):
        for model in self.models:
            model.registerDerivs(container)

class VarlikeModelboundData(ModelboundData):
    """Variable-like modelbound data class used for easier instantiation of varlike modelbound data for models"""

    def __init__(self) -> None:
        self.__variables__:List[Variable] = []

    @property 
    def varNames(self):
        return [v.name for v in self.__variables__] 

    def __getitem__(self,key:str):
        if key not in self.varNames: 
            raise KeyError()
        return self.__variables__[self.varNames.index(key)]
    
    def addVar(
        self,
        *args:Variable,
    ) -> None:

        for var in args:
            assert var.derivation is not None, "Variable "+var.name+" does not have a derivation associated to it - cannot add to VarlikeModelboundData"
            assert "t" not in var.dims, "Variable "+var.name+" has time dimension - cannot add to VarlikeModelboundData"

            assert var.name not in self.varNames, "Variable "+var.name+" already in VarlikeModelboundData" 

            self.__variables__.append(var)

    def dict(self) -> dict:
        properties = {
            "modelboundDataType": "varlikeData",
            "dataNames": self.varNames,
        }

        for var in self.__variables__:
            isDerivedFromOtherData = False
            if any(arg in self.varNames for arg in var.derivationArgs):
                assert all(arg in self.varNames for arg in var.derivationArgs), "Modelbound variable "+var.name+" has some derivationArgs in the containing modelbound data, but not all. If the variable was not meant to depend on other modelbound data consider renaming some of the variables in the modelbound data object." 
                isDerivedFromOtherData = True
            properties[var.name] = {
                "isDistribution": var.isDistribution,
                "isScalar": var.isScalar,
                "isSingleHarmonic": var.isSingleHarmonic,
                "isDerivedFromOtherData": isDerivedFromOtherData,
                "derivationPriority": var.priority,
                "ruleName": var.derivation.name, 
                "requiredVarNames": var.derivationArgs
            }
        return properties

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        latexRemap:Dict[str,str] = kwargs.get("latexRemap",{})
        with doc.create(tex.Itemize()) as itemize:
            for var in self.__variables__:
                itemize.add_item(tex.NoEscape(f"${var.latex(latexRemap)}$"))

    def registerDerivs(self, container: Textbook):
        for var in self.__variables__:
            container.register(var.derivation,ignoreDuplicates=True)

class LBCModelboundData(ModelboundData):

    def __init__(self,grid:Grid,
                    ionCurrent:Variable, 
                    distFun:Variable,
                    density:Variable,
                    densityDual:Union[Variable,None]=None,
                    densityOnBoundary:Union[Variable,None]=None,
                    totalCurrent:Union[Variable,None]=None,
                    bisTol: float=1e-12,
                    leftBoundary=False):

        assert ionCurrent.isScalar, "ionCurrent in LBCModelboundData must be a scalar"
        self.__ionCurrent__ = ionCurrent
        
        assert distFun.isDistribution, "distFun in LBCModelboundData must be a distribution"
        
        assert density.isFluid, "density must be a fluid variable in LBCModelboundData"
        
        if densityDual is not None:
            assert densityDual.isOnDualGrid, "densityDual must be on the dual grid in LBCModelboundData"
            assert densityDual.isFluid, "densityDual must be a fluid variable in LBCModelboundData"
        
        if densityOnBoundary is not None:
            assert densityOnBoundary.isScalar, "densityOnBoundary must be a scalar variable in LBCModelboundData"
        
        if totalCurrent is not None:
            assert totalCurrent.isScalar, "totalCurrent must be a scalar variable in LBCModelboundData"
        self.__totalCurrent__ = totalCurrent
        
        self.__bisTol__ = bisTol
        self.__leftBoundary__ = leftBoundary

        self.__deriv__ = GenericDerivation(("left" if leftBoundary else "right")+"DistExt",0,{"type": "distScalingExtrapDerivation",
        "extrapolateToBoundary": isinstance(densityOnBoundary,Variable),
        "staggeredVars": isinstance(densityDual,Variable),
        "leftBoundary": leftBoundary},"\\text{DistExt}_"+"L"if leftBoundary else "R")
        self.__reqVars__ = [distFun.name,density.name]
        if densityDual is not None:
            self.__reqVars__.append(densityDual.name)
        if densityOnBoundary is not None:
            self.__reqVars__.append(densityOnBoundary.name)

        self.__grid__ = grid

    def dict(self):
        mbData = {
            "modelboundDataType": "modelboundLBCData",
            "ionCurrentVarName": self.__ionCurrent__.name,
            "totalCurrentVarName": self.__totalCurrent__.name if self.__totalCurrent__ is not None else "none",
            "bisectionTolerance": self.__bisTol__,
            "leftBoundary": self.__leftBoundary__,
        }
        mbData.update({"ruleName": self.__deriv__.name, "requiredVarNames": self.__reqVars__})
        
        return mbData

    @property
    def varNames(self):
        return ["gamma","potential","coVel","shTemp"]

    def __getitem__(self, key):
        if key not in self.varNames: 
            raise KeyError()
        return Variable(key,self.__grid__,isDerived=True,isScalar=True)

    def addLatexToDoc(self, doc, **kwargs):
        latexRemap:Dict[str,str] = kwargs.get("latexRemap",{})
        doc.append("Logical boundary condition data on "+("left" if self.__leftBoundary__ else "right") + " boundary")
        with doc.create(tex.Itemize()) as itemize:
                for var in self.varNames:
                    itemize.add_item(tex.NoEscape(f"${self[var].latex(latexRemap)}$"))

    def registerDerivs(self, container: Textbook):
        container.register(self.__deriv__,ignoreDuplicates=True)
