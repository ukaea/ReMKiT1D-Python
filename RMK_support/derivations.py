from typing import Union, List, Dict, cast, Tuple, Type,Callable
from typing_extensions import Self
import numpy as np
from abc import ABC, abstractmethod
from .import calculation_tree_support as ct
from copy import copy,deepcopy
from .grid import Grid,Profile
import pylatex as tex
from math import isclose
from .tex_parsing import numToScientificTex
from scipy import special
from scipy.interpolate import RegularGridInterpolator

#TODO: docs

class DerivBase(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod 
    def rename(self,name:str) -> Self:
        pass

class DerivationArgument(ABC):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def apply(cls,deriv:Type[DerivBase],*args: Self):
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
        associatedVars: List[DerivationArgument] = [],
        latexName: Union[str,None] = None
    ) -> None:

        assert atomicA > 0, "Species mass must be positive"

        self.__name__ = name
        self.__speciesID__ = speciesID
        self.__atomicA__ = atomicA
        self.__charge__ = charge
        self.__associatedVars__ = copy(associatedVars)
        self.__latexName__ = latexName

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
    def associatedVarNames(self):
        return [var.name for var in self.__associatedVars__]

    def associateVar(self,*args:DerivationArgument):
        for arg in args:
            if arg.name not in self.associatedVarNames:
                self.__associatedVars__.append(arg)

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

    def latex(self):
        return self.__latexName__ if self.__latexName__ is not None else "\\text{"+self.name.replace("_","\_")+"}"

class SpeciesContainer:

    def __init__(self,*args:Species):
        self.__species__ = list(args)

    @property 
    def speciesNames(self):
        return [species.name for species in self.__species__]

    def dict(self) -> dict: 

        speciesDict = {"names":self.speciesNames}

        for species in self.__species__:
            speciesDict.update({species.name:species.dict()})

        return speciesDict

    def addLatexToDoc(self,doc:tex.Document,**kwargs):
        if len(self.__species__) > 0:
            latexRemap:Dict[str,str] = kwargs.get("latexRemap",{})
            with doc.create(tex.Section("Species")):
                with doc.create(tex.Itemize()) as itemize:
                    for species in self.__species__:
                        associatedVarNames = ["$"+latexRemap[varName]+"$" if varName in latexRemap else "$\\text{"+varName.replace("_","\_")+"}$" for varName in species.associatedVarNames]
                        itemize.add_item(tex.NoEscape("$"+species.latex()+"$"+f": ID: {species.speciesID}; A: {species.atomicA:.4e}; Z: {species.charge:.2f}; Associated vars: "+",".join(associatedVarNames)))

    def __getitem__(self,key:str):
        if key not in self.speciesNames: 
            raise KeyError()
        return self.__species__[self.speciesNames.index(key)]

    def __setitem__(self,key:str,species:Species):
        if key not in self.speciesNames: 
            self.__species__.append(species)
        else:
            self.__species__[self.speciesNames.index(key)] = species

    def __delitem__(self,key:str):
        if key not in self.speciesNames: 
            raise KeyError()
        del self.__species__[self.speciesNames.index(key)]

    def add(self,*args:Species):
        for arg in args:
            assert arg.name not in self.speciesNames, "Attempted to add duplicate species to SpeciesContainer - use key access if you wish to overwrite an existing species"

        self.__species__+=list(args)
class DerivationContainer(ABC):

    @abstractmethod
    def register(self,deriv:DerivBase,ignoreDuplicates=False) -> None: 
        pass
class Derivation(DerivBase):

    def __init__(self, name: str, numArgs: int, latexTemplate: Union[str, None] = None, latexName: Union[str,None] = None, container:Union[DerivationContainer,None]=None) -> None:
        super().__init__()
        self.__name__ = name
        assert numArgs >= 0, "Negative number of arguments in Derivation"
        self.__numArgs__ = numArgs
        self.__latexName__ = latexName if latexName is not None else "\\text{"+name.replace("_","\_")+"}"
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
    def name(self,name:str):
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
            raise Exception(
                "latexTemplate not provided in Derivation and latex() function not overwritten"
            )
        expression = copy(self.__latexTemplate__)
        for i, name in enumerate(args):
            expression = expression.replace(f"${i}", name)

        return expression

    def evaluate(
        self,*args:np.ndarray
    ) -> np.ndarray:
        raise Exception("Derivation evaluation not implemented")

    def __call__(self, *args):
        if (len(args)):
            if isinstance(args[0],str):
                return self.latex(*args)
            if isinstance(args[0],np.ndarray):
                return self.evaluate(*args)
            if issubclass(type(args[0]),DerivationArgument):
                return type(args[0]).apply(self,*args)
            raise TypeError("Unsupported argument in Derivation __call__")
            
        else:
            return self.dict()

    def registerComponents(self,container:DerivationContainer):
        pass

    def fillArgs(self,*args:str) -> List[str]:
        return list(args)

    @property 
    def resultProperties(self) -> Dict[str,object]:
        return {}

    @property 
    def enclosedArgs(self) -> int:
        return 0

class GenericDerivation(Derivation):

    def __init__(
        self,
        name: str,
        numArgs: int,
        properties: Dict[str, object],
        latexTemplate: Union[str, None] = None,
        resultProperties:Dict[str,object]={},
        container:Union[DerivationContainer,None]=None
    ) -> None:
        """Generic derivation wrapper taking in the ReMKiT1D dictionary representation

        Args:
            name (str): Name of the derivation
            properties (Dict[str,object]): Dictionary representation of the ReMKiT1D derivation
        """
        super().__init__(name, numArgs, latexTemplate,container=container)
        self.__properties__ = properties
        self.__resultProperties__ = resultProperties

    def dict(self) -> dict:
        return self.__properties__

    def latex(self, *args) -> str:
        assert (
            len(args) == self.numArgs
        ), "latex() called with args not conforming to the number of expected arguments"
        if self.latexTemplate is None:
            expression =  self.latexName + "\\left("
            for name in args:
                expression += name + ","
            return expression[:-1] + "\\right)"
        else:
            return super().latex(*args)

    @property
    def resultProperties(self):
        return self.__resultProperties__

class NodeDerivation(Derivation):

    def __init__(self, name:str, node: ct.Node, latexTemplate: Union[str, None] = None, container:Union[DerivationContainer,None]=None) -> None:
        super().__init__(name,len(ct.getLeafVars(node)), latexTemplate, container=container)
        self.__node__ = node

    def dict(self) -> dict:
        return ct.treeDerivation(self.__node__)

    def evaluate(
        self,*args
    ) -> np.ndarray:
        assert (
            len(args) == self.numArgs
        ), "evaluate() called with args not conforming to the number of expected arguments"
        
        return self.__node__.evaluate(dict(zip(ct.getLeafVars(self.__node__),args)))

    def latex(self, *args:str) -> str:
        assert (
            len(args) == self.numArgs
        ), "latex() called with args not conforming to the number of expected arguments"
        if self.latexTemplate is None:
            remap = dict(zip(ct.getLeafVars(self.__node__),args))
            return self.__node__.latex(remap)
        else:
            return super().latex(*args)

    @property 
    def node(self):
        return self.__node__

class SimpleDerivation(Derivation):

    def __init__(
        self,
        name:str,
        multConst: float,
        varPowers: List[float],
        container:Union[DerivationContainer,None]=None
    ) -> None:
        """Simple derivation object which calculates its value as multConst * prod(vars**powers)

        Args:
            multConst (float): Multiplicative constant
            varPowers (List[float]): Powers to raise passed variables to.
        """
        super().__init__(name,len(varPowers),container=container)
        self.__multConst__ = multConst
        self.__varPowers__ = varPowers

    def dict(self) -> dict:
        return {
            "type": "simpleDerivation",
            "multConst": self.__multConst__,
            "varPowers": self.__varPowers__,
        }

    def evaluate(
        self, *args:np.ndarray
    ) -> np.ndarray:
        assert len(args) == len(
            self.__varPowers__
        ), "SimpleDerivation args must conform to the number of variable powers"
        val = self.__multConst__ * args[0] ** self.__varPowers__[0]
        for i in range(1, len(args)):
            val *= args[i] ** self.__varPowers__[i]
        return val

    def latex(self, *args:str) -> str:
        expression = numToScientificTex(self.__multConst__,removeUnity=True)
        numerator = ""
        denominator = ""

        for i,arg in enumerate(args):
            if self.__varPowers__[i] > 0:
                 if isclose(self.__varPowers__[i],1.0,rel_tol=1e-4):
                    numerator+=" "+arg
                 else: 
                    power =f"{round(self.__varPowers__[i])}" if isclose(self.__varPowers__[i],round(self.__varPowers__[i]),rel_tol=1e-2) else  f"{self.__varPowers__[i]:.2f}"
                    numerator+=" "+arg+"^{"+power+"}"
            else:
                if isclose(self.__varPowers__[i],-1.0,rel_tol=1e-4):
                    denominator+=" "+arg
                else:
                    power =f"{round(-self.__varPowers__[i])}" if isclose(self.__varPowers__[i],round(self.__varPowers__[i]),rel_tol=1e-2) else  f"{-self.__varPowers__[i]:.2f}"
                    denominator+=" "+arg+"^{"+power+"}"
        if len(denominator):
            return expression+"\\frac{"+numerator+"}{"+denominator+"}"
        return expression + numerator

class BuiltInDerivation(Derivation):

    def __init__(self, name: str, numArgs: int, latexTemplate: Union[str,None] = None, latexName: Union[str,None] = None) -> None:
        super().__init__(name, numArgs, latexTemplate, latexName)

    def dict(self) -> dict:
        """No-op since these are built-in derivations
        """
        return {}

class InterpolationDerivation(BuiltInDerivation):

    def __init__(self, grid:Grid, ontoDual=True, onDistribution=False):
        if onDistribution and not ontoDual:
            raise ValueError("InterpolationDerivation on distributions must be ontoDual")
        name = "gridToDual" if ontoDual else "dualToGrid"
        if onDistribution:
            name = "distributionInterp"
        super().__init__(name, 1)
        self.__grid__ = grid 

    def evaluate(self, *args:np.ndarray):
        assert len(args) == 1, "InterpolationDerivation evaluate() must have exactly one argument"
        if self.name == "distributionInterp":
            return self.__grid__.distFullInterp(args[0])
        if self.name == "gridToDual":
            return self.__grid__.gridToDual(args[0])
        if self.name == "dualToGrid":
            return self.__grid__.dualToGrid(args[0])

    def latex(self, *args: str):
        assert len(args) == 1, "InterpolationDerivation latex() must have exactly one argument"

        if self.name == "gridToDual":
            return "\\mathcal{I}\\left("+args[0]+"\\rightarrow \\text{dual}\\right)"
        if self.name == "dualToGrid":
            return "\\mathcal{I}\\left("+args[0]+"\\rightarrow \\text{grid}\\right)"
        if self.name == "distributionInterp":
            return "\\mathcal{I}\\left("+args[0]+"\\rightarrow \\text{grid/dual}\\right)"


class Textbook(DerivationContainer):

    def __init__(self,grid:Grid,tempDerivSpeciesIDs: List[int] = [],
        ePolyCoeff=1.0,
        ionPolyCoeff=1.0,
        electronSheathGammaIonSpeciesID=-1,
        removeLogLeiDiscontinuity=False) -> None:
        super().__init__()

        self.__grid__ = grid
        self.__tempDerivSpeciesIDs__ = copy(tempDerivSpeciesIDs)
        self.__ePolyCoeff__ = ePolyCoeff
        self.__ionPolyCoeff__ = ionPolyCoeff
        self.__electronSheathGammaIonSpeciesID__ =electronSheathGammaIonSpeciesID
        self.__removeLogLeiDiscontinuity__ = removeLogLeiDiscontinuity

        self.__derivations__:List[Derivation] = []

        #Built-in derivations 
        #TODO: add result properties to all derivations!
        self.__derivations__.append(GenericDerivation("flowSpeedFromFlux",2,{},latexTemplate="\\frac{$0}{$1}"))
        self.__derivations__.append(GenericDerivation("leftElectronGamma",2,{},latexTemplate="\\gamma_{e,\\text{left}}\\left($0,$1\\right)"))
        self.__derivations__.append(GenericDerivation("rightElectronGamma",2,{},latexTemplate="\\gamma_{e,\\text{right}}\\left($0,$1\\right)"))
        self.__derivations__.append(GenericDerivation("densityMoment",1,{},latexTemplate="\\langle $0 \\rangle _{l=0,n=0}"))
        self.__derivations__.append(GenericDerivation("energyMoment",1,{},latexTemplate="\\langle $0 \\rangle _{l=0,n=2}"))

        self.__derivations__.append(GenericDerivation("cclDragCoeff",1,{},latexTemplate="C_{CCL}($0)"))
        self.__derivations__.append(GenericDerivation("cclDiffusionCoeff",2,{},latexTemplate="D_{CCL}($0,$1)"))
        self.__derivations__.append(GenericDerivation("cclWeight",2,{},latexTemplate="\\delta_{CCL}($0,$1)"))

        if grid.lMax > 0:
            self.__derivations__.append(GenericDerivation("fluxMoment",1,{},latexTemplate="\\frac{1}{3}\\langle $0 \\rangle _{l=1,n=1}"))
            self.__derivations__.append(GenericDerivation("heatFluxMoment",1,{},latexTemplate="\\frac{1}{3}\\langle $0 \\rangle _{l=1,n=3}"))
        if grid.lMax > 1:
            self.__derivations__.append(GenericDerivation("viscosityTensorxxMoment",1,{},latexTemplate="\\frac{2}{15}\\langle $0 \\rangle _{l=2,n=2}"))

        self.__derivations__.append(InterpolationDerivation(grid))
        self.__derivations__.append(InterpolationDerivation(grid,False))
        self.__derivations__.append(InterpolationDerivation(grid,onDistribution=True))

        self.__derivations__.append(GenericDerivation("gradDeriv",1,{},latexTemplate="\\nabla\\left($0\\right)"))
        self.__derivations__.append(GenericDerivation("logLee",2,{},"\\text{log}\\Lambda_{e,e}\\left($0,$1\\right)"))

        self.__derivations__.append(GenericDerivation("maxwellianDistribution",2,{},"f_M\\left($0,$1\\right)"))

        self.__builtinDerivNum__ = len(self.__derivations__)

    @property 
    def registeredDerivs(self):
        return [deriv.name for deriv in self.__derivations__]
    
    def __getitem__(self, name:str):
        if name not in self.registeredDerivs: 
            if name.startswith("sonicSpeed"):
                return GenericDerivation(name,2,{},latexTemplate="c_{s,"+name[10:]+"}\\left($0,$1\\right)")
            if name.startswith("tempFromEnergy"):
                return GenericDerivation(name,3,{},latexTemplate="T_{"+name[14:]+"}\\left($0,$1,$2\\right)")
            if name.startswith("logLei"):
                return GenericDerivation(name,2,{},latexTemplate="\\text{log}\\Lambda_{e,"+name[6:]+"}\\left($0,$1\\right)")
            if name.startswith("logLii"):
                names = name.split("_")
                return GenericDerivation(name,4,{},latexTemplate="\\text{log}\\Lambda_{"+names[0][6:]+","+names[1]+"}\\left($0,$1\\right)")

            raise KeyError()
        return self.__derivations__[self.registeredDerivs.index(name)]
    
    def register(self, deriv: Derivation,ignoreDuplicates=False) -> None:
        if not ignoreDuplicates:
            assert deriv.name not in self.registeredDerivs, "Derivation "+deriv.name+" already registered in textbook"
            for prefix in ["sonicSpeed","tempFromEnergy","logLei","logLii"]:
                assert not deriv.name.startswith(prefix), "Derivation "+deriv.name+" starts with reserved prefix"
        try:
            _=self[deriv.name]
        except KeyError:
            deriv.registerComponents(self)
            try:
                _=self[deriv.name]
            except KeyError:
                self.__derivations__.append(deriv)

    def dict(self):

        textbookDict={"standardTextbook":
            {"temperatureDerivSpeciesIDs":self.__tempDerivSpeciesIDs__,
             "electronPolytropicCoeff":self.__ePolyCoeff__,
             "ionPolytropicCoeff":self.__ionPolyCoeff__,
             "electronSheathGammaIonSpeciesID":self.__electronSheathGammaIonSpeciesID__,
             "removeLogLeiDiscontinuity": self.__removeLogLeiDiscontinuity__
            }}

        textbookDict["customDerivations"] = { "tags":self.registeredDerivs[self.__builtinDerivNum__:]}
        for i in range(self.__builtinDerivNum__,len(self.__derivations__)):
            textbookDict["customDerivations"][self.__derivations__[i].name] = self.__derivations__[i].dict()

        return textbookDict

    def addSpeciesForTempDeriv(self,species:Species):
        self.__tempDerivSpeciesIDs__ = list(set(self.__tempDerivSpeciesIDs__).add(species.speciesID))

    @property
    def ePolyCoeff(self):
        return self.__ePolyCoeff__

    @ePolyCoeff.setter
    def ePolyCoeff(self,val:float):
        self.__ePolyCoeff__ =  val

    @property
    def ionPolyCoeff(self):
        return self.__ionPolyCoeff__

    @ionPolyCoeff.setter
    def ionPolyCoeff(self,val:float):
        self.__ionPolyCoeff__ =  val

    def setSheathGammaSpecies(self,species:Species):
        self.__electronSheathGammaIonSpeciesID__ = species.speciesID 

    @property
    def removeLogLeiDiscontinuity(self):
        return self.__removeLogLeiDiscontinuity__

    @removeLogLeiDiscontinuity.setter
    def removeLogLeiDiscontinuity(self,remove:bool):
        self.__removeLogLeiDiscontinuity__ = remove

    def addLatexToDoc(self,doc:tex.Document):
        if len(self.registeredDerivs) > self.__builtinDerivNum__:
            with doc.create(tex.Section("Custom derivations")):
                with doc.create(tex.Itemize()) as itemize:
                        for deriv in self.__derivations__[self.__builtinDerivNum__:]:
                            argTuple = tuple("x_"+str(i) for i in range(deriv.numArgs+deriv.enclosedArgs))
                            itemize.add_item(tex.NoEscape(deriv.name.replace("_","\_")+f": ${deriv.latex(*argTuple)}$"))

class MultiplicativeDerivation(Derivation):

    def __init__(self, name: str, innerDerivation:Derivation, outerDerivation: Union[Derivation,None]=None, innerFunc:Union[str,None]=None,  container: Union[DerivationContainer, None] = None) -> None:
        numArgs = innerDerivation.numArgs 
        if outerDerivation is not None:
            numArgs+=outerDerivation.numArgs
        super().__init__(name, numArgs, container=container)

        self.__innerDeriv__ = innerDerivation
        self.__outerDeriv__:Union[str,None] = outerDerivation 

        funMap = {"exp":np.exp,
                  "log":np.log,
                  "sin":np.sin,
                  "cos":np.cos,
                  "abs":np.abs,
                  "tan":np.tan,
                  "atan":np.arctan,
                  "asin":np.arcsin,
                  "acos":np.arccos,
                  "sign":np.sign,
                  "erf":special.erf,
                  "erfc":special.erfc
                  }

        self.__innerFunc__:Union[str,None] = innerFunc
        self.__fun__:Union[Callable,None] = None
        if self.__innerFunc__ is not None:
            assert self.__innerFunc__ in funMap, "Unsupported function passed to MultiplicativeDerivation"
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
        return self.innerDeriv.enclosedArgs if self.outerDeriv is None else self.innerDeriv.enclosedArgs + self.outerDeriv.enclosedArgs

    def dict(self) -> Dict:
        
        deriv = {
            "type": "multiplicativeDerivation",
            "innerDerivation": self.innerDeriv.name,
            "innerDerivIndices": [i+1 for i in range(self.innerDeriv.numArgs+self.innerDeriv.enclosedArgs)],
            "innerDerivPower": 1.0,
            "outerDerivation": "none" if self.outerDeriv is None else self.outerDeriv.name,
            "outerDerivIndices": (
                [] if self.outerDeriv is None else [i+1+self.innerDeriv.numArgs+self.innerDeriv.enclosedArgs for i in range(self.outerDeriv.numArgs+self.outerDeriv.enclosedArgs)]
            ),
            "outerDerivPower": 1.0,
            "innerDerivFuncName": "none" if self.__innerFunc__ is None else self.__innerFunc__,
        }

        return deriv

    def latex(self, *args:str)->str:
        assert len(args) == self.numArgs+self.enclosedArgs, "latex() on MultiplicativeDerivation called with wrong number of arguments"
        result = "\\left("+self.innerDeriv.latex(*args[:self.innerDeriv.numArgs+self.innerDeriv.enclosedArgs])+"\\right)"

        if self.innerFunc is not None:
            result = self.innerFunc+result

        if self.outerDeriv is not None:
            result+="\\left("+self.outerDeriv.latex(*args[self.innerDeriv.numArgs+self.innerDeriv.enclosedArgs:])+"\\right)"

        return result

    def evaluate(self, *args):
        assert len(args) == self.numArgs, "evaluate() on MultiplicativeDerivation called with wrong number of arguments"
        result = self.innerDeriv.evaluate(*args[:self.innerDeriv.numArgs])

        if self.innerFunc is not None:
            result = self.__fun__(result)

        if self.outerDeriv is not None:
            result*=self.outerDeriv.evaluate(*args[self.innerDeriv.numArgs:])


        return result 

    def fillArgs(self, *args):
        inner = self.innerDeriv.fillArgs(*args[:self.innerDeriv.numArgs])
        outer = []
        if self.outerDeriv is not None:
            outer = self.outerDeriv.fillArgs(*args[self.innerDeriv.numArgs:])
        return inner+outer

    def registerComponents(self, container):
        container.register(self.__innerDeriv__,ignoreDuplicates=True)
        if self.outerDeriv is not None:
            container.register(self.__outerDeriv__,ignoreDuplicates=True)

class AdditiveDerivation(Derivation):

    def __init__(self, name, derivs:List[Derivation],resultPower: Union[int,float]=1,linCoeffs:Union[List[float],None]=None, container = None):
        numArgs = sum(deriv.numArgs for deriv in derivs)
        super().__init__(name, numArgs, container=container)
        
        if linCoeffs is not None:
            assert len(derivs) == len(
                    linCoeffs
            ), "derivs and linCoeffs in AdditiveDerivation must be of same size"

        self.__resultPower__ = resultPower
        self.__linCoeffs__ = linCoeffs if linCoeffs is not None else [1.0 for _ in derivs]
        self.__derivs__:List[Derivation] = []
        for deriv in derivs:
            if deriv.name in [d.name for d in self.__derivs__]:
                derivCopy = deepcopy(deriv)
                derivCopy.name+="_copy"
                while derivCopy.name in [d.name for d in self.__derivs__]:
                    derivCopy.name+="_copy"
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

    def fillArgs(self, *args):

        derivArgs:List[str]=[]
        
        offset = 0
        for deriv in self.__derivs__:
            derivArgs+=deriv.fillArgs(*args[offset:offset+deriv.numArgs])
            offset+=deriv.numArgs 

        return derivArgs
    def dict(self):

        derivIndices:List[List[int]] = []
        offset = 1 
        for deriv in self.__derivs__:
            derivIndices.append([offset+k for k in range(deriv.numArgs)])
            offset+=deriv.numArgs 
        
        derivTags = [deriv.name for deriv in self.__derivs__]
        derivDict = {
            "type": "additiveDerivation",
            "derivationTags": derivTags,
            "resultPower": self.__resultPower__,
            "linearCoefficients": self.__linCoeffs__,
        }

        for i, tag in enumerate(derivTags):
            derivDict[tag] = {"derivationIndices": derivIndices[i]}

        return derivDict

    def latex(self, *args:str)->str:
        assert len(args) == self.numArgs+self.enclosedArgs, "latex() on AdditiveDerivation called with wrong number of arguments"
        result = ""
        offset = 0
        for i,deriv in enumerate(self.__derivs__):
            if self.__linCoeffs__[i] < 0:
                result+=""
            else: 
                result+="+ " if i>0 else " "
            coeffRepr = numToScientificTex(self.__linCoeffs__[i],removeUnity=True)
            result+=coeffRepr+"\\left("+deriv.latex(*args[offset:offset+deriv.numArgs+deriv.enclosedArgs])+"\\right)" if coeffRepr != "" else deriv.latex(*args[offset:offset+deriv.numArgs+deriv.enclosedArgs])
            offset+=deriv.numArgs
        powerRepr = numToScientificTex(self.__resultPower__)
        result = "\\left("+result+"\\right)^{"+powerRepr+"}" if self.__resultPower__!= 1 else result

        return result 

    def evaluate(self, *args):
        assert len(args) == self.numArgs, "evaluate() on AdditiveDerivation called with wrong number of arguments"
        result = 0
        offset = 0
        for i,deriv in enumerate(self.__derivs__):
            result += self.__linCoeffs__[i]*deriv.evaluate(*args[offset:offset+deriv.numArgs])
            offset+=deriv.numArgs
        result = result ** self.__resultPower__ 

        return result 

    def registerComponents(self, container):
        for deriv in self.__derivs__:
            container.register(deriv,ignoreDuplicates=True)

class DerivationClosure(Derivation):

    def __init__(self, deriv:Derivation, *args:DerivationArgument, container:Union[DerivationContainer,None] = None,**kwargs):
        self.__deriv__ = deriv
        self.__args__ = args 
        self.__argPositions__ = kwargs.get("argPositions",tuple(range(len(args))))
        assert len(self.__argPositions__) == len(args), "argPositions in DerivationClosure must conform to the number of args"
        #TODO: add bounds check for argPositions
        super().__init__(deriv.name, deriv.numArgs - len(args), latexTemplate=deriv.latexTemplate, latexName=deriv.latexName)

        if container is not None: 
            container.register(deriv,ignoreDuplicates=True)

    def rename(self, name):
        self.__name__ = name 
        self.__deriv__.name = name 
        return self 
    
    def dict(self):
        return {}

    @property
    def enclosedArgs(self):
        return len(self.__args__) + self.__deriv__.enclosedArgs
    
    def fillArgs(self, *args:str):
        argNameList:List[str] = []

        argCounter = 0
        for i in range(self.__deriv__.numArgs):
            if i in self.__argPositions__:
                argNameList.append(self.__args__[self.__argPositions__.index(i)].name)
            else:
                argNameList.append(args[argCounter])
                argCounter+=1

        argNameList=self.__deriv__.fillArgs(*tuple(argNameList))

        return argNameList

    def latex(self, *args):

        return self.__deriv__.latex(*args)

    def evaluate(self, *args:np.ndarray):

        argValList:List[np.ndarray] = []

        argCounter = 0
        for i in range(self.__deriv__.numArgs):
            if i in self.__argPositions__:
                argValList.append(self.__args__[self.__argPositions__.index(i)].data)
            else:
                argValList.append(args[argCounter])
                argCounter+=1 

        return self.__deriv__.evaluate(*tuple(argValList))

    def __generalAdd__(self,rhs:Self) -> Self:
        addDeriv = AdditiveDerivation(self.name+"_"+rhs.name,[self.__deriv__,rhs.__deriv__])
        posMap = tuple(list(self.__argPositions__) + [self.__deriv__.numArgs+i for i in rhs.__argPositions__])
        return DerivationClosure(addDeriv,*tuple(list(self.__args__)+list(rhs.__args__)),argPositions=posMap)

    def __add__(self,rhs:Self) -> Self: 
        assert self.numArgs == 0 and rhs.numArgs == 0, "Only complete closures can be added"
        if isinstance(self.__deriv__,AdditiveDerivation):
            if isinstance(rhs.__deriv__,AdditiveDerivation):
                if self.__deriv__.resultPower == 1 and rhs.__deriv__.resultPower == 1:
                     addDeriv = AdditiveDerivation(self.name+"_"+rhs.name,self.__deriv__.derivs+rhs.__deriv__.derivs,resultPower=1,linCoeffs=self.__deriv__.linCoeffs+rhs.__deriv__.linCoeffs)
                     posMap = tuple(list(self.__argPositions__) + [self.__deriv__.numArgs+i for i in rhs.__argPositions__])
                     return DerivationClosure(addDeriv,*tuple(list(self.__args__)+list(rhs.__args__)),argPositions=posMap)
                return self.__generalAdd__(rhs)
            if self.__deriv__.resultPower == 1:
                addDeriv = AdditiveDerivation(self.name+"_"+rhs.name,self.__deriv__.derivs+[rhs.__deriv__],resultPower=1,linCoeffs=self.__deriv__.linCoeffs+[1.0])
                posMap = tuple(list(self.__argPositions__) + [self.__deriv__.numArgs+i for i in rhs.__argPositions__])
                return DerivationClosure(addDeriv,*tuple(list(self.__args__)+list(rhs.__args__)),argPositions=posMap)
            return self.__generalAdd__(rhs)
        
        if isinstance(rhs.__deriv__,AdditiveDerivation):
            if rhs.__deriv__.resultPower == 1:
                addDeriv = AdditiveDerivation(self.name+"_"+rhs.name,[self.__deriv__]+rhs.__deriv__.derivs,resultPower=1,linCoeffs=[1.0]+rhs.__deriv__.linCoeffs)
                posMap = tuple(list(self.__argPositions__) + [self.__deriv__.numArgs+i for i in rhs.__argPositions__])
                return DerivationClosure(addDeriv,*tuple(list(self.__args__)+list(rhs.__args__)),argPositions=posMap)
        return self.__generalAdd__(rhs)

    def __generalMul__(self,rhs:Self) -> Self:
        mulDeriv = MultiplicativeDerivation(self.name+"X"+rhs.name,innerDerivation=self.__deriv__,outerDerivation=rhs.__deriv__)
        posMap = tuple(list(self.__argPositions__) + [self.__deriv__.numArgs+i for i in rhs.__argPositions__])
        return DerivationClosure(mulDeriv,*tuple(list(self.__args__)+list(rhs.__args__)),argPositions=posMap)

    def __mul__(self,rhs:Self) -> Self: 
        assert self.numArgs == 0 and rhs.numArgs == 0, "Only complete closures can be multiplied"
        if isinstance(self.__deriv__,MultiplicativeDerivation):
            if self.__deriv__.outerDeriv is None:
                mulDeriv = MultiplicativeDerivation(self.name+"X"+rhs.name,innerDerivation=self.__deriv__.innerDeriv,innerFunc=self.__deriv__.innerFunc,outerDerivation=rhs.__deriv__)
                posMap = tuple(list(self.__argPositions__) + [self.__deriv__.numArgs+i for i in rhs.__argPositions__])
                return DerivationClosure(mulDeriv,*tuple(list(self.__args__)+list(rhs.__args__)),argPositions=posMap)
            return self.__generalMul__(rhs)
        if isinstance(rhs.__deriv__,MultiplicativeDerivation):
            return rhs.__mul__(self) 
        return self.__generalMul__(rhs)

    def __rmul__(self,lhs:Union[float,int]) -> Self: 
        assert isinstance(lhs,(int,float))
        assert self.numArgs == 0, "Only complete closures can be multiplied"
        if isinstance(self.__deriv__,AdditiveDerivation):
            addDeriv = AdditiveDerivation(self.name+"_rmul",self.__deriv__.derivs,resultPower=self.__deriv__.resultPower,linCoeffs=[lhs**(1/self.__deriv__.resultPower)*coeff for coeff in self.__deriv__.linCoeffs])
            return DerivationClosure(addDeriv,*self.__args__,argPositions=self.__argPositions__)
        addDeriv = AdditiveDerivation(self.name+"_rmul",[self.__deriv__],linCoeffs=[lhs])
        return DerivationClosure(addDeriv,*self.__args__,argPositions=self.__argPositions__)

    def __pow__(self,rhs:Union[float,int]) -> Self: 
        assert isinstance(rhs,(int,float))
        assert self.numArgs == 0, "Only complete closures can be raised to powers"
        if isinstance(self.__deriv__,AdditiveDerivation):
            addDeriv = AdditiveDerivation(self.name+"_pow",self.__deriv__.derivs,resultPower=self.__deriv__.resultPower*rhs,linCoeffs=self.__deriv__.linCoeffs)
            return DerivationClosure(addDeriv,*self.__args__,argPositions=self.__argPositions__)
        addDeriv = AdditiveDerivation(self.name+"_pow",[self.__deriv__],linCoeffs=[1.0],resultPower=rhs)
        return DerivationClosure(addDeriv,*self.__args__,argPositions=self.__argPositions__)
    
    def registerComponents(self, container):
        container.register(self.__deriv__,ignoreDuplicates=True)

def funApply(funName:str,closure:DerivationClosure):
    assert closure.numArgs == 0, "Can only apply functions to complete derivation closures"
    if isinstance(closure.__deriv__,MultiplicativeDerivation):
        if closure.__deriv__.outerDeriv is None and closure.__deriv__.innerFunc is None:
            mulDeriv = MultiplicativeDerivation(funName+"_"+closure.name,innerDerivation=closure.__deriv__.innerDeriv,innerFunc=funName)
            return DerivationClosure(mulDeriv,*closure.__args__,argPositions=closure.__argPositions__)
    mulDeriv = MultiplicativeDerivation(funName+"_"+closure.name,innerDerivation=closure.__deriv__,innerFunc=funName)
    return DerivationClosure(mulDeriv,*closure.__args__,argPositions=closure.__argPositions__)

class PolynomialDerivation(GenericDerivation):

    def __init__(self, name: str, constCoeff:Union[float,int], polyCoeffs: np.ndarray, polyPowers:np.ndarray, container: Union[DerivationContainer,None] = None) -> None:
        assert len(polyCoeffs) == len(polyPowers)
        properties = {"type": "polynomialFunctionDerivation",
        "constantPolynomialCoefficient": constCoeff,
        "polynomialPowers": polyPowers.tolist(),
        "polynomialCoefficients": polyCoeffs.tolist()}
        super().__init__(name, len(polyCoeffs), properties, latexTemplate=None, container=container)
        self.__constCoeff__ = constCoeff
        self.__polyCoeffs__ = polyCoeffs
        self.__polyPowers__ = polyPowers 

    def latex(self, *args: str) -> str:
        assert len(args) == self.numArgs, "Unexpected number of arguments in PolynomialDerivation latex() call"
        result = numToScientificTex(self.__constCoeff__) if self.__constCoeff__ != 0 else ""
        for i,arg in enumerate(args):
            result += "" if self.__polyCoeffs__[i] < 0 else (" + " if result != "" else "")
            result += numToScientificTex(self.__polyCoeffs__[i],True)
            powerRepr = numToScientificTex(self.__polyCoeffs__[i])
            if powerRepr == "1.00": 
                result += "\\left("+arg+"\\right)^{"+ powerRepr+"}"
            else: 
                result +=arg 

        return result

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        assert len(args) == self.numArgs, "Unexpected number of arguments in PolynomialDerivation evaluate() call"
        result = self.__constCoeff__
        for i,arg in enumerate(args):
            result+=self.__polyCoeffs__[i]*arg**self.__polyPowers__[i]

        return result

class RangeFilterDerivation(Derivation):

    def __init__(self, name: str, deriv:Derivation, filtering: List[Tuple[DerivationArgument,float,float]],container: Union[DerivationContainer, None] = None) -> None:
        super().__init__(name, deriv.numArgs, container=container)
        self.__deriv__ = deriv 
        for _,range0,range1 in filtering: 
            assert range0<range1,"Ranges in RangeFilterDerivation must be monotonic"
        self.__filtering__ = filtering

    def dict(self) -> Dict:
        controlRangesDict = {}
        for i,fl in enumerate(self.__filtering__):
            _,range0,range1 = fl
            controlRangesDict["index" + str(i + 1)] = [range0,range1]

        deriv = {
            "type": "rangeFilterDerivation",
            "ruleName": self.__deriv__.name,
            "controlIndices": [i+1 for i in range(len(self.__filtering__))],
            "controlRanges": controlRangesDict,
            "derivationIndices":[i+1+len(self.__filtering__) for i in range(self.numArgs)]
        }

        return deriv

    def latex(self, *args: str) -> str:

        assert len(args) == self.numArgs+len(self.__filtering__), "latex() on RangeFilterDerivation called with wrong number of arguments"
        result = "\\left("+self.__deriv__.latex(*args[len(self.__filtering__):])+"\\right)\\Pi\\left["

        for i,fl in enumerate(self.__filtering__):
            _,range0,range1 = fl
            scale = range1-range0
            offset = (range1+range0)/2
            sign = " - "
            if offset < 0:
                offset*=-1
                sign = " + "
            if i >0:
                result+=", "
            result+="\\left("+args[i] + sign + numToScientificTex(offset)+"\\right)/"+numToScientificTex(scale)
        result+="\\right]"

        return result

    def evaluate(self, *args: np.ndarray) -> np.ndarray:

        filterVals = []
        for i,fl in enumerate(self.__filtering__):
            _,range0,range1 = fl
            filterVals.append[np.where(args[i]<range1 and args[i]>range0,np.ones(args[i].shape),np.zeros(args[i].shape))]

        result = self.__deriv__.evaluate(*args[len(self.__filtering__):])
        for filter in filterVals:
            result*= filter

        return result

    def registerComponents(self, container: DerivationContainer):
        container.register(self.__deriv__)

    def fillArgs(self, *args: str) -> List[str]:
        argNames = [var.name for var,_,_ in self.__filtering__]
        argNames+=list(args)
        return argNames

class BoundedExtrapolationDerivation(Derivation):

    def __init__(self, name, extrapolationType:str = "lin", lowerBound:Union[float,DerivationArgument,None]=None,upperBound:Union[float,int,DerivationArgument,None]=None, container = None,**kwargs):
        super().__init__(name, 1, container=container)
        assert extrapolationType in ["lin","log","linlog"], "Unsupported extrapolation type detected"
        self.__extrapType__ = extrapolationType
        if isinstance(lowerBound,(float,int)):
            assert lowerBound >= 0, "Fixed lowerBound must be non-positive"
        if isinstance(upperBound,(float,int)):
            assert upperBound >= 0, "Fixed upperBound must be non-positive"

        if isinstance(lowerBound,(float,int)) and isinstance(upperBound,(float,int)):
            assert upperBound > lowerBound, "Fixed upperBound must be greated than fixed lowerBound"
        self.__lowerBound__=lowerBound
        self.__upperBound__=upperBound

        self.__leftBoundary__:bool = kwargs.get("leftBoundary",False)
        self.__staggeredVars__:bool = kwargs.get("staggeredVars",False)
        self.__expectedHaloWidth__:Union[int,None] = kwargs.get("expectedHaloWidth",None)

    def dict(self):

        deriv = {
            "type": "boundedExtrapolationDerivation",
            "expectUpperBoundVar": isinstance(self.__upperBound__,DerivationArgument),
            "expectLowerBoundVar": isinstance(self.__lowerBound__,DerivationArgument),
            "ignoreUpperBound": self.__upperBound__ is None,
            "ignoreLowerBound": self.__lowerBound__ is None,
            "fixedLowerBound": self.__lowerBound__ if isinstance(self.__lowerBound__,(float,int)) else 0,
            "extrapolationStrategy": {
                "type": self.__extrapType__ + "Extrapolation",
                "leftBoundary": self.__leftBoundary__,
                "staggeredVars": self.__staggeredVars__,
            },
        }

        if isinstance(self.__upperBound__,(float,int)):
            deriv["fixedUpperBound"] = self.__upperBound__

        if isinstance(self.__expectedHaloWidth__,int):
            deriv["extrapolationStrategy"].update({"expectedHaloWidth":self.__expectedHaloWidth__})

        return deriv

    @property
    def enclosedArgs(self):
        return int(isinstance(self.__lowerBound__,DerivationArgument)) + int(isinstance(self.__upperBound__,DerivationArgument))

    def latex(self, *args):
        var = args[0]
        lowerBound = ""
        if isinstance(self.__lowerBound__,(float,int)):
            lowerBound = numToScientificTex(self.__lowerBound__)
        if isinstance(self.__lowerBound__,DerivationArgument):
            lowerBound = args[1]

        upperBound=""
        if isinstance(self.__upperBound__,DerivationArgument):
            upperBound = args[2] if isinstance(self.__lowerBound__,DerivationArgument) else args[1]
        if isinstance(self.__upperBound__,(float,int)):
            upperBound = numToScientificTex(self.__upperBound__)

        side = "L" if self.__leftBoundary__ else "R"
        return "\\text{Extrap}_{"+side+","+self.__extrapType__+"}\\left("+var+"\\right)^{"+upperBound+"}_{"+lowerBound+"}"

    def fillArgs(self, *args):
        newArgs = [args[0]]
        if isinstance(self.__lowerBound__,DerivationArgument):
            newArgs.append(self.__lowerBound__.name)
        if isinstance(self.__upperBound__,DerivationArgument):
            newArgs.append(self.__upperBound__.name)
        return newArgs

    @property
    def resultProperties(self):
        return {"isScalar":True,
                "isDistribution":False,
                "isSingleHarmonic":False}

def coldIonIDeriv(name:str,index:int):
    deriv = {
        "type": "coldIonIJIntegralDerivation",
        "isJIntegral": False,
        "index": index,
    }
    return GenericDerivation(name,1,deriv,"I_{c,"+str(index)+"}\\left($0\\right)")

def coldIonJDeriv(name:str,index:int):
    deriv = {
        "type": "coldIonIJIntegralDerivation",
        "isJIntegral": True,
        "index": index,
    }
    return GenericDerivation(name,1,deriv,"J_{c,"+str(index)+"}\\left($0\\right)")

def shkarofskyIIntegralDeriv(name:str,index:int):
    deriv = {
        "type": "IJIntegralDerivation",
        "isJIntegral": False,
        "index": index,
    }
    return GenericDerivation(name,1,deriv,"I_{"+str(index)+"}\\left($0\\right)")

def shkarofskyJIntegralDeriv(name:str,index:int):
    deriv = {
        "type": "IJIntegralDerivation",
        "isJIntegral": True,
        "index": index,
    }
    return GenericDerivation(name,1,deriv,"J_{"+str(index)+"}\\left($0\\right)")

class HarmonicExtractorDerivation(Derivation):

    def __init__(self, name, grid:Grid, index:int, container = None):
        super().__init__(name, 1, container=container)
        self.__grid__ = grid 
        assert index>0 ,"Harmonic index must be positive"
        assert index<=self.__grid__.numH(),"Harmonic index out of bounds"
        self.__index__ = index 

    def dict(self):
        return {"type": "harmonicExtractorDerivation", "index": self.__index__}

    def latex(self, *args):
        assert len(args) == self.numArgs, "Wrong number of args passed to HarmonicExtractorDerivation latex()"
        return "\\left("+args[0]+"\\right)_{h="+str(self.__index__)+"}"

    def evaluate(self, *args):
        assert len(args) == self.numArgs, "Wrong number of args passed to HarmonicExtractorDerivation evaluate()"
        assert args[0].shape == (self.__grid__.numX(),self.__grid__.numH(),self.__grid__.numV()),"args[0] of HarmonicExtractorDerivation must be a full distribution"

        return args[0][:,self.__index__,:]

    @property
    def resultProperties(self):
        return {"isScalar":False,"isDistribution":False,"isSingleHarmonic":True}

class DDVDerivation(Derivation):

    def __init__(self, name: str, grid:Grid, harmonicIndex:int, innerV:Union[None,Profile]=None, outerV:Union[None,Profile]=None,vifAtZero:Union[None,Tuple[float,float]]=None, container: Union[DerivationContainer,None] = None) -> None:
        super().__init__(name, 1, container=container)
        self.__grid__ = grid 
        self.__harmonicIndex__ = harmonicIndex
        if innerV is not None:
            assert innerV.dim == "V" and len(innerV.data) == grid.numV(), "innerV size does not conform to velocity grid"
        self.__innerV__ = innerV 
        if outerV is not None:
            assert outerV.dim == "V" and len(outerV.data) == grid.numV(), "outerV size does not conform to velocity grid"
        self.__outerV__ = outerV 
        self.__vifAtZero__ = vifAtZero

    def dict(self) -> Dict:
        deriv: Dict[str, object] = {"type": "ddvDerivation",
                                    "targetH":self.__harmonicIndex__}
        if self.__outerV__ is not None:
            deriv["outerV"] = self.__outerV__.data.tolist()
        if self.__innerV__ is not None:
            deriv["innerV"] = self.__innerV__.data.tolist()
        if self.__vifAtZero__ is not None:
            deriv["vifAtZero"] = list(self.__vifAtZero__)

        return deriv

    def latex(self, *args: str) -> str:
        assert self.numArgs == len(args), "latex() on DDVDerivation called with wrong number of arguments"
        outerV = "" if self.__outerV__ is None else "V_o"
        if self.__innerV__ is None:
            return outerV+"\\frac{\\partial {"+args[0]+"}_{h="+str(self.__harmonicIndex__)+"}}{\\partial v}"
        return outerV+"\\frac{\\partial }{\\partial v}\\left(V_i {"+args[0]+"}_{h="+str(self.__harmonicIndex__)+"}\\right)"

    @property 
    def resultProperties(self):
        return {"isScalar":False,
                "isDistribution":False,
                "isSingleHarmonic":True}

class D2DV2Derivation(Derivation):

    def __init__(self, name: str, grid:Grid, harmonicIndex:int, innerV:Union[None,Profile]=None, outerV:Union[None,Profile]=None,vidfdvAtZero:Union[None,Tuple[float,float]]=None, container: Union[DerivationContainer,None] = None) -> None:
        super().__init__(name, 1, container=container)
        self.__grid__ = grid 
        self.__harmonicIndex__ = harmonicIndex
        if innerV is not None:
            assert innerV.dim == "V" and len(innerV.data) == grid.numV(), "innerV size does not conform to velocity grid"
        self.__innerV__ = innerV 
        if outerV is not None:
            assert outerV.dim == "V" and len(outerV.data) == grid.numV(), "outerV size does not conform to velocity grid"
        self.__outerV__ = outerV 
        self.__vidfdvAtZero__ = vidfdvAtZero

    def dict(self) -> Dict:
        deriv: Dict[str, object] = {"type": "d2dv2Derivation",
                                    "targetH":self.__harmonicIndex__}
        if self.__outerV__ is not None:
            deriv["outerV"] = self.__outerV__.data.tolist()
        if self.__innerV__ is not None:
            deriv["innerV"] = self.__innerV__.data.tolist()
        if self.__vidfdvAtZero__ is not None:
            deriv["vidfdvAtZero"] = list(self.__vidfdvAtZero__)

        return deriv

    def latex(self, *args: str) -> str:
        assert self.numArgs == len(args), "latex() on D2DV2Derivation called with wrong number of arguments"

        outerV = "" if self.__outerV__ is None else "V_o"
        if self.__innerV__ is None:
            return outerV+"\\frac{\\partial^2 {"+args[0]+"}_{h="+str(self.__harmonicIndex__)+"}}{\\partial v^2}"
        inner = "\\frac{\\partial {"+args[0]+"}_{h="+str(self.__harmonicIndex__)+"}}{\\partial v}"
        return outerV+"\\frac{\\partial }{\\partial v}\\left(V_i "+inner+"\\right)"

    @property 
    def resultProperties(self):
        return {"isScalar":False,
                "isDistribution":False,
                "isSingleHarmonic":True}

class MomentDerivation(Derivation):

    def __init__(self, name: str, grid:Grid, momentHarmonic:int, momentOrder:int, multConst: float = 1.0,
        varPowers: List[float]=[], gVec: Union[Profile, None] = None, container: Union[DerivationContainer,None] = None) -> None:
        super().__init__(name, 1+len(varPowers), container)
        self.__momentHarmonic__ = momentHarmonic
        self.__momentOrder__ = momentOrder 
        self.__multConst__ = multConst 
        self.__varPowers__ = varPowers 
        self.__grid__ = grid 
        if gVec is not None:
            assert gVec.dim == "V" and len(gVec.data) == grid.numV(), "gVec size does not conform to velocity grid"
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
        assert len(args) == self.numArgs, "latex() on MomentDerivation called with wrong number of arguments"
        expression = numToScientificTex(self.__multConst__,removeUnity=True)
        numerator = ""
        denominator = ""
        if len(args) > 1: 

            for i,arg in enumerate(args[1:]):
                if self.__varPowers__[i] > 0:
                    if isclose(self.__varPowers__[i],1.0,rel_tol=1e-4):
                        numerator+=" "+arg
                    else: 
                        power =f"{round(self.__varPowers__[i])}" if isclose(self.__varPowers__[i],round(self.__varPowers__[i]),rel_tol=1e-2) else  f"{self.__varPowers__[i]:.2f}"
                        numerator+=" "+arg+"^{"+power+"}"
                else:
                    if isclose(self.__varPowers__[i],-1.0,rel_tol=1e-4):
                        denominator+=" "+arg
                    else:
                        power =f"{round(-self.__varPowers__[i])}" if isclose(self.__varPowers__[i],round(self.__varPowers__[i]),rel_tol=1e-2) else  f"{-self.__varPowers__[i]:.2f}"
                        denominator+=" "+arg+"^{"+power+"}"
            if len(denominator):
                expression+="\\frac{"+numerator+"}{"+denominator+"}"
            else:
                expression+= numerator
        g= ""
        if self.__gVec__ is not None:
            g = "g(v)"
        return expression + "\\langle "+g+args[0]+" \\rangle _{h="+str(self.__momentHarmonic__)+",n="+str(self.__momentOrder__)+"}"

    @property
    def resultProperties(self):
        return {"isScalar":False,"isDistribution":False,"isSingleHarmonic":False}
class GenIntPolynomialDerivation(Derivation):

    def __init__(self, name: str,polyPowers: np.ndarray,polyCoeffs: np.ndarray,multConst: float = 1.0,
    funcName: Union[None, str] = None, container: Union[DerivationContainer, None] = None) -> None:
        super().__init__(name, polyPowers.shape[1], container)
        self.__polyPowers__ = polyPowers 
        self.__polyCoeffs__ = polyCoeffs
        self.__multConst__ = multConst 
        self.__funcName__ = funcName

        funMap = {"exp":np.exp,
                  "log":np.log,
                  "sin":np.sin,
                  "cos":np.cos,
                  "abs":np.abs,
                  "tan":np.tan,
                  "atan":np.arctan,
                  "asin":np.arcsin,
                  "acos":np.arccos,
                  "sign":np.sign,
                  "erf":special.erf,
                  "erfc":special.erfc
                  }

        self.__funcName__:Union[str,None] = funcName
        self.__fun__:Union[Callable,None] = None
        if self.__funcName__ is not None:
            assert self.__funcName__ in funMap, "Unsupported function passed to GenIntPolynomialDerivation"
            self.__fun__ = funMap[self.__funcName__]

    def dict(self) -> Dict:
        polyPowersDict = {}
        for i in range(self.__polyPowers__.shape[0]):
            polyPowersDict["index" + str(i + 1)] = self.__polyPowers__[i,:].tolist()

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
        assert len(args) == self.numArgs, "Unexpected number of arguments in GenIntPolynomialDerivation latex() call"
        expression = numToScientificTex(self.__multConst__)
        innerExpr = "\\sum_i c_i "+" ".join([arg+"^{p_{i,"+str(j)+"}}" for j,arg in enumerate(args)])
        if self.__funcName__ is not None:
            return expression + self.__funcName__+"\\left( "+ innerExpr + "\\right)"

        return expression + innerExpr

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        assert len(args) == self.numArgs, "Unexpected number of arguments in GenIntPolynomialDerivation evaluate() call"
        result = 0
        for ind,coeff in np.ndenumerate(self.__polyCoeffs__):
            for j,arg in enumerate(args):
                result+=coeff*arg**self.__polyPowers__[ind[0],j]
        if self.__funcName__ is not None: 
            result = self.__fun__(result)
        return result*self.__multConst__

class LocValExtractorDerivation(Derivation):

    def __init__(self, name: str, grid:Grid, targetX:int, container: Union[DerivationContainer,None] = None) -> None:
        super().__init__(name, 1, container=container)
        self.__grid__ = grid 
        self.__targetX__ = targetX 
        assert targetX > 0 and targetX<= grid.numX() ,"LocValExtractorDerivation targetX out of bounds"

    def dict(self) -> Dict:
        return {"type": "locValExtractorDerivation", "targetX": self.__targetX__}

    def latex(self, *args: str) -> str:
        assert len(args) == self.numArgs, "LocValExtractorDerivation latex() call unexpected number of arguments"
        return "\\left(" + args[0]+"\\right)_{X="+str(self.__targetX__)+"}"

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        assert len(args) == self.numArgs, "LocValExtractorDerivation evaluate() call unexpected number of arguments"
        assert args[0].shape == (self.__grid__.numX(),), "LocValExtractorDerivation evaluate() argument does not conform to the X dimension of the grid"
        return np.array([args[0][self.__targetX__-1]])

    @property 
    def resultProperties(self):
        return {"isScalar":True,
                "isDistribution":False,
                "isSingleHarmonic":False}

class NDInterpolationDerivation(Derivation):

    def __init__(self, name: str, grids: List[np.ndarray], data: np.ndarray, container: Union[DerivationContainer, None] = None) -> None:
        super().__init__(name, len(grids), container)
        self.__grids__ = grids 
        self.__data__ = data 
        dataShape = np.shape(data)
        assert len(grids) == len(dataShape), "The number of orthogonal grids in NDInterpolationDerivation must be the equal to the number of data dimensions"

        for i, grid in enumerate(grids):
            assert len(grid) == dataShape[i], (
                "Grid "+str(i)
                + " does not conform to the corresponding dimension of interpolation data"
            )

    def dict(self) -> Dict:
        deriv = {
        "type": "nDLinInterpDerivation",
        "data": {"dims": list(self.__data__.shape), "values": self.__data__.flatten(order="F").tolist()},
        "grids": {"names": ["grid"+str(i) for i in range(len(self.__grids__))]},
    }
        for i, name in enumerate(["grid"+str(i) for i in range(len(self.__grids__))]):
            cast(Dict[str, object], deriv["grids"])[name] = self.__grids__[i].tolist()

        return deriv 

    def latex(self, *args: str) -> str:
        return "\\mathcal{I}_{ND}\\left("+",".join(args)+"\\right)"

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        
        return RegularGridInterpolator(tuple(self.__grids__),self.__data__,fill_value=0)(np.array(list(zip(*args))))
        