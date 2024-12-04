from typing import Union, List, Dict, cast, Tuple
from typing_extensions import Self
import numpy as np
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
import pylatex as tex
from .tex_parsing import numToScientificTex
from . import model_construction as mc
from .variable_container import Variable, VariableContainer, MultiplicativeArgument

class Integrator(ABC):

    def __init__(self,name:str):
        super().__init__()
        self.__name__ = name 

    @property
    def name(self) -> str:
        return self.__name__ 

    @abstractmethod 
    def dict(self) -> dict: 
        pass 

    @abstractmethod
    def addLatexToDoc(self,doc:tex.Document,**kwargs) -> None:
        pass 


class IntegrationRule:

    def __init__(self,model:mc.Model, updatedGroups:Union[List[int],None]=None,evaluatedGroups:Union[List[int],None]=None,updateModelData=True):

        self.__updatedGroups__ = updatedGroups 
        self.__evaluatedGroups__ = evaluatedGroups
        self.__updateModelData__ = updateModelData 
        self.__model__ = model

    @property 
    def modelName(self):
        return self.__model__.name
    
    @property
    def updatedGroups(self):
        if self.__updatedGroups__ is None:
            return self.__evaluatedGroups__
        return self.__updatedGroups__ 

    @updatedGroups.setter
    def updatedGroups(self,groups:List[int]):
        self.__updatedGroups__ = groups

    @property
    def evaluatedGroups(self):
        return self.__evaluatedGroups__ 

    @evaluatedGroups.setter
    def evaluatedGroups(self,groups:List[int]):
        self.__evaluatedGroups__ = groups

    @property
    def updateModelData(self):
        return self.__updateModelData__ 

    @updateModelData.setter
    def updateModelData(self,update:bool):
        self.__updateModelData__ = update

    def defaultGroups(self,implicitGroups:int):

        if self.evaluatedGroups is None:
            self.evaluatedGroups = self.__model__.activeImplicitGroups + [implicitGroups+g for g in self.__model__.activeGeneralGroups]

    def latex(self) -> str:
        expression = "Evolved model: " + self.__model__.name.replace("_","\_")
        expression+= "\\newline Evaluated groups: "+",".join(str(group) for group in self.evaluatedGroups)
        expression+= "\\newline Updated groups: "+",".join(str(group) for group in self.updatedGroups)
        expression+="\\newline Update modelbound data (if any): "+"Yes" if self.updateModelData else "No"

        return expression

class IntegrationStepBase(ABC): 

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def stepFraction(self):
        pass

    @abstractmethod
    def rename(self,name:str) -> Self:
        pass

    @abstractmethod
    def addLatexToDoc(self,doc:tex.Document,implicitGroups:int,**kwargs):
        pass

class IntegrationStepSequence:

    def __init__(self,*args:IntegrationStepBase):
        self.__steps__:List[IntegrationStepBase] = list(args)

    @property 
    def steps(self):
        return self.__steps__

    @steps.setter
    def steps(self, steps:List[IntegrationStepBase]):
        self.__steps__ = steps

    def __mul__(self,rhs:Union[IntegrationStepBase,Self]) -> Self:
        assert isinstance(rhs,(IntegrationStepBase,Self)), "__mul__ rhs of IntegrationStepSequence must be IntegrationStep or IntegrationSequence"
        if isinstance(rhs,IntegrationStepBase):
            newSequence = deepcopy(self)
            newSequence.__steps__ = [rhs.rename(rhs.name+str(len(self.steps)))]+newSequence.__steps__
        if isinstance(rhs,IntegrationStepSequence):
            newSequence = deepcopy(rhs)
            newSequence.__steps__ += self.__steps__
        return newSequence

    def __rmul__(self,lhs:IntegrationStepBase) -> Self:
        assert isinstance(lhs,IntegrationStepBase), "__rmul__ lhs of IntegrationStepSequence must be IntegrationStep"
        newSequence = deepcopy(self)
        newSequence.__steps__+=[lhs.rename(lhs.name+str(len(self.steps)))]
        return newSequence

    def addLatexToDoc(self,doc:tex.Document,implicitGroups:int,**kwargs):
        doc.append(tex.NoEscape("Scheme: $"+"".join("\\text{"+step.name+"}("+numToScientificTex(step.stepFraction,removeUnity=True)+"dt)" for step in self.steps) +"$"))
        for step in self.steps:
            with doc.create(tex.Subsection(step.name)):
                step.addLatexToDoc(doc,implicitGroups,**kwargs)


    
class IntegrationStep(IntegrationStepBase):
    """Class containing integration step data"""

    def __init__(self, name:str, integrator:Integrator,**kwargs) -> None:
        self.__name__ = name
        self.__integrator__ = integrator
        self.__globalStepFraction__ = kwargs.get("globalStepFraction",1.0)
        self.__allowTimeEvolution__ = kwargs.get("allowTimeEvolution",True)
        self.__useInitialInput__ = kwargs.get("useInitialInput",False)
        self.__rules__:List[IntegrationRule] = []

        selfOuter = self
        class Rules:
            def __getitem__(self, model:mc.Model):
                if model.name not in selfOuter.evolvedModels: 
                    raise KeyError()
                return selfOuter.__rules__[selfOuter.evolvedModels.index(model.name)]
            
            def __setitem__(self, model:mc.Model,rule:IntegrationRule):
                if model.name not in selfOuter.evolvedModels: 
                    selfOuter.__rules__.append(rule)
                    return
                selfOuter.__rules__[selfOuter.evolvedModels.index(model.name)] = rule

            def __delitem__(self,model:mc.Model):
                if model.name not in selfOuter.evolvedModels: 
                    raise KeyError()
                del selfOuter.__rules__[selfOuter.evolvedModels.index(model.name)]

        self.rules = Rules()

    @property 
    def evolvedModels(self):
        return [m.modelName for m in self.__rules__]

    @property
    def name(self):
        return self.__name__ 

    def rename(self,name:str):
        newStep = deepcopy(self)
        newStep.__name__ = name 
        return newStep 

    def __call__(self,step:float):
        newStep = deepcopy(self)
        newStep.__globalStepFraction__ = step 
        return newStep 

    @property 
    def stepFraction(self):
        return self.__globalStepFraction__

    @property
    def timeVariableEvolved(self):
        return self.__allowTimeEvolution__ 

    @property 
    def startingFromZero(self):
        return self.__useInitialInput__

    def disableTimeEvo(self):
        self.__allowTimeEvolution__ = False 
        return self 

    def enableTimeEvo(self):
        self.__allowTimeEvolution__ = True 
        return self 

    def startFromZero(self):
        self.__useInitialInput__ = True 
        return self 

    def startFromLast(self):
        self.__useInitialInput__ = False 
        return self

    @property
    def integrator(self):
        return self.__integrator__

    @integrator.setter
    def integrator(self, integ:Integrator):
        self.__integrator__ = integ 

    def add(self,*args:Union[IntegrationRule,mc.Model,mc.ModelCollection]) -> None:
        for arg in args:
            assert isinstance(arg,(mc.Model,IntegrationRule,mc.ModelCollection)), "__add__ RHS for Integration step must be Model, ModelCollection, or IntegrationRule"
            if isinstance(arg,mc.Model):
                self.rules[arg] = IntegrationRule(arg)
            if isinstance(arg,IntegrationRule):
                self.rules[arg.__model__] = arg 
            if isinstance(arg,mc.ModelCollection):
                for model in arg.models:
                    self.add(model)
                

    def __mul__ (self,rhs:Union[Self,IntegrationStepSequence]) -> IntegrationStepSequence:
        assert isinstance(rhs,(IntegrationStep,IntegrationStepSequence)), "__mul__ rhs for IntegrationStep must be IntegrationStep or IntegrationStepSequence"

        if isinstance(rhs,IntegrationStep):
            return self*(rhs*IntegrationStepSequence())

        return rhs.__rmul__(self)

    def dict(self,implicitGroups:int) -> dict:
        """Return ReMKiT1D-readable dictionary object

        Returns:
            dict: Integration step property dictionary
        """

        step = {
            "integratorTag": self.integrator.name,
            "evolvedModels": self.evolvedModels,
            "globalStepFraction": self.__globalStepFraction__,
            "allowTimeEvolution": self.__allowTimeEvolution__,
            "useInitialInput": self.__useInitialInput__,
        }

        for rule in self.__rules__:
            rule.defaultGroups(implicitGroups)
            step.update({rule.modelName: { "groupIndices": (
                rule.evaluatedGroups
            ),
            "internallyUpdatedGroups": (
                rule.updatedGroups
            ),
            "internallyUpdateModelData": (
                rule.updateModelData
            ) }})

        return step

    def addLatexToDoc(self, doc:tex.Document,implicitGroups:int,**kwargs):
        doc.append(tex.NoEscape("Evolving time: "+("Yes" if self.__allowTimeEvolution__ else "No")))
        doc.append(tex.NoEscape("\\newline Starting from initial state: "+("Yes" if self.__useInitialInput__ else "No")))
        with doc.create(tex.Subsubsection("Integrator")):
            self.integrator.addLatexToDoc(doc,**kwargs)
        with doc.create(tex.Subsubsection("Integration rules")):
            with doc.create(tex.Itemize()) as itemize:
                for rule in self.__rules__:
                    rule.defaultGroups(implicitGroups)
                    itemize.add_item(tex.NoEscape(rule.latex()))
            

class Timestep:

    def __init__(self,timestep:Union[Variable,MultiplicativeArgument,float]):
        self.__timestep__:MultiplicativeArgument = MultiplicativeArgument()*timestep
        self.__max__ = False

    @property
    def usingMaxVal(self):
        return self.__max__ 

    def max(self) -> Self:
        self.__max__ = True 
        return self

    def min(self) -> Self:
        self.__max__ = False 
        return self 

    def dict(self) -> dict:
        return {
        "timestepController": {"active": len(self.__timestep__.args) > 0,
                               "rescaleTimestep": True,
        "requiredVarNames": list(self.__timestep__.argMultiplicity.keys()),
        "requiredVarPowers": [item for _,item in self.__timestep__.argMultiplicity.items()],
        "multConst": 1.0,
        "useMaxVal": self.__max__,},
            "initialTimestep": self.__timestep__.scalar
        }

    def checkConsistency(self, varCont: VariableContainer):

        for _,var in self.__timestep__.args.items():
            assert var.name in varCont.varNames, (
                    "Timestep scaling variable " + var + " not found in used variable container"
                )
    def latex(self,**kwargs):
        latexRemap:Dict[str,str] = kwargs.get("latexRemap",{})
        if len(self.__timestep__.args) > 0:
            if self.__max__:
                return "\\text{max}\\left("+numToScientificTex(self.__timestep__.scalar)+self.__timestep__.latex(latexRemap)+"\\right)"
            return "\\text{min}\\left("+numToScientificTex(self.__timestep__.scalar)+self.__timestep__.latex(latexRemap)+"\\right)"
        return numToScientificTex(self.__timestep__.scalar)

class IntegrationScheme:

    def __init__(self,dt=Union[Timestep,float],steps:Union[IntegrationStepSequence,IntegrationStep]=IntegrationStepSequence()):
        self.__timestep__= dt if isinstance(dt,Timestep) else Timestep(dt)
        self.__stepSequence__:IntegrationStepSequence = steps if isinstance(steps,IntegrationStepSequence) else IntegrationStepSequence(steps)

        self.__mode__ = "fixedNumSteps"
        self.__numTimesteps__ = 1
        self.__outputInterval__ = 1 

        self.__outputPoints__:List[float] = []

    @property 
    def timestep(self):
        return self.__timestep__ 

    @timestep.setter
    def timestep(self,dt=Union[Timestep,float]):
        self.__timestep__ = dt if isinstance(dt,Timestep) else Timestep(dt) 

    @property 
    def steps(self):
        return self.__stepSequence__.steps

    @steps.setter
    def steps(self,seq:IntegrationStepSequence):
        self.__stepSequence__ = seq 

    def setFixedNumTimesteps(self,numTimesteps=1,outputInterval=1):

        self.__mode__ = "fixedNumSteps"
        self.__numTimesteps__ = numTimesteps
        self.__outputInterval__ = outputInterval 

    def setOutputPoints(self,outputPoints:List[float]):
        assert all(point > 0 for point in outputPoints), "All output points must be positive"
        assert all(np.diff(np.array(outputPoints))>0), "outputPoints must be monotonically increasing sequence"

        self.__outputPoints__ = outputPoints 
        self.__mode__ = "outputDriven"

    def dict(self,implicitGroups:int, mpiComm: dict) -> dict: 

        scheme = {"stepTags": [step.name for step in self.steps],
            "integratorTags": list(set(step.integrator.name for step in self.steps))}
        scheme.update(self.__timestep__.dict())
        for step in self.steps:
            scheme[step.name] = {"commData":mpiComm}
            scheme[step.name].update(step.dict(implicitGroups))
            scheme.update({step.integrator.name:step.integrator.dict()})

        timeloop = {"mode":self.__mode__,
                    "numTimesteps":self.__numTimesteps__,
                    "fixedSaveInterval":self.__outputInterval__,
                    "outputPoints":self.__outputPoints__}
        return {"integrator":scheme,"timeloop":timeloop}

    def addLatexToDoc(self,doc:tex.Document,implicitGroups:int,**kwargs):
        with doc.create(tex.Section("Integration scheme")):
            doc.append(tex.NoEscape("$dt ="+self.timestep.latex(**kwargs)+" $ "))
            if self.__mode__=="fixedNumSteps":
                doc.append(tex.NoEscape("\\newline Running for "+str(self.__numTimesteps__)+" steps, outputting every "+str(self.__outputInterval__)))
            else:
                doc.append(tex.NoEscape("\\newline Output points: "+",".join("$"+numToScientificTex(point) +"$" for point in self.__outputPoints__)))
            doc.append(tex.NoEscape("\\newline Total number of implicit groups: "+str(implicitGroups)))
            with doc.create(tex.Subsection("Integration steps")):
                self.__stepSequence__.addLatexToDoc(doc,implicitGroups,**kwargs)
    
class BDEIntegrator(Integrator):

    def __init__(self,name:str,**kwargs):
        super().__init__(name)
        self.__maxNonlinIters__:int = kwargs.get("maxNonlinIters",100)
        self.__nonlinTol__:float = kwargs.get("nonlinTol",1.0e-12)
        self.__absTol__:float = kwargs.get("absTol",1.0)
        self.__convergenceVars__:List[Variable] = kwargs.get("convergenceVars",[])
        self.__associatedPETScGroup__:int = kwargs.get("associatedPETScGroup",1)
        self.__use2Norm__:bool = kwargs.get("use2Norm",False)

        
        self.__internalStepControl__:bool = kwargs.get("internalStepControl",False)

        self.__initialNumInternalSteps__:int = kwargs.get("initialNumInternalSteps",1)
        self.__stepMultiplier__:int=kwargs.get("stepMultiplier",2)
        self.__stepDecrament__=kwargs.get("stepDecrament",1)
        self.__minNonlinIters__=kwargs.get("minNonlinIters",5)
        self.__consolidationInterval__=kwargs.get("consolidationInterval",50)
        self.__maxBDERestarts__=kwargs.get("maxBDERestarts",3)
        self.__relaxationWeight__:float=kwargs.get("relaxationWeight", 1.0)

    def dict(self) -> dict: 

        integ = {
            "type": "BDE",
            "maxNonlinIters": self.__maxNonlinIters__,
            "nonlinTol": self.__nonlinTol__,
            "absTol": self.__absTol__,
            "convergenceVars": [var.name for var in self.__convergenceVars__],
            "associatedPETScGroup": self.__associatedPETScGroup__,
            "use2Norm": self.__use2Norm__,
            "relaxationWeight": self.__relaxationWeight__,
            "internalStepControl": {
                "active": self.__internalStepControl__,
                "startingNumSteps": self.__initialNumInternalSteps__,
                "stepMultiplier": self.__stepMultiplier__,
                "stepDecrament": self.__stepDecrament__,
                "minNumNonlinIters": self.__minNonlinIters__,
                "maxBDERestarts": self.__maxBDERestarts__,
                "BDEConsolidationInterval": self.__consolidationInterval__,
            },
        }

        return integ

    def addLatexToDoc(self, doc, **kwargs):
        doc.append(self.name+": Backwards Euler integrator with fixed-point iterations")
        

class RKIntegrator(Integrator):

    def __init__(self, name: str,order:int):
        self.__order__ = order
        super().__init__(name)

    def dict(self) -> Dict:
        return {"type": "RK", "order": self.__order__}

    def addLatexToDoc(self, doc, **kwargs):
        doc.append(self.name+": Runge-Kutta integrator - order "+str(self.__order__))

class CVODEIntegrator(Integrator):

    def __init__(self, name: str,**kwargs):

        self.__relTol__:float = kwargs.get("relTol",1e-5)
        self.__absTol__:float = kwargs.get("absTol",1e-10)
        self.__maxGMRESRestarts__:int = kwargs.get("maxGMRESRestarts",0)
        self.__CVODEBBDPreParams__:Tuple[int,int,int,int] = kwargs.get("CVODEBBDPreParams", (0, 0, 0, 0))
        self.__useAdamsMoulton__:bool = kwargs.get("useAdamsMoulton",False)
        self.__useStabLimitDet__:bool = kwargs.get("useStabLimitDet",False)
        self.__maxOrder__:int = kwargs.get("maxOrder",5)
        self.__maxInternalStep__:int = kwargs.get("maxInternalStep",500)
        self.__minTimestep__:float = kwargs.get("minTimestep", 0.0)
        self.__maxTimestep__:float = kwargs.get("maxTimestep", 0.0)
        self.__initTimestep__:float = kwargs.get("initTimestep", 0.0)
        super().__init__(name)

    def dict(self) -> Dict:
        return {
        "type": "CVODE",
        "relTol": self.__relTol__,
        "absTol": self.__absTol__,
        "maxRestarts": self.__maxGMRESRestarts__,
        "CVODEPreBBDParams": self.__CVODEBBDPreParams__,
        "CVODEUseAdamsMoulton": self.__useAdamsMoulton__,
        "CVODEUseStabLimDet": self.__useStabLimitDet__,
        "CVODEMaxOrder": self.__maxOrder__,
        "CVODEMaxInternalSteps": self.__maxInternalStep__,
        "CVODEMaxStepSize": self.__maxTimestep__,
        "CVODEMinStepSize": self.__minTimestep__,
        "CVODEInitStepSize": self.__initTimestep__,
    }

    def addLatexToDoc(self, doc, **kwargs):
        doc.append(self.name+": CVODE Integrator")
