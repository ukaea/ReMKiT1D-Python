from typing import Union, List, Dict, cast, Tuple
from typing_extensions import Self
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
import pylatex as tex  # type: ignore
from .tex_parsing import numToScientificTex
from . import model_construction as mc
from .variable_container import Variable, VariableContainer, MultiplicativeArgument
import warnings


class Integrator(ABC):
    """Abstract integrator base class"""

    def __init__(self, name: str):
        super().__init__()
        self.__name__ = name

    @property
    def name(self) -> str:
        return self.__name__

    @abstractmethod
    def dict(self) -> dict:
        pass

    @abstractmethod
    def addLatexToDoc(self, doc: tex.Document, **kwargs) -> None:
        pass


class IntegrationRule:
    """Integration rule, encapsulating how a given model should be evaluated/updated in an integration step"""

    def __init__(
        self,
        model: mc.Model,
        updatedGroups: Union[List[int], None] = None,
        evaluatedGroups: Union[List[int], None] = None,
        updateModelData=True,
    ):
        """Integration rule for a given model

        Args:
            model (mc.Model): Model whose contributions are being integrated
            updatedGroups (Union[List[int], None], optional): Groups in the model that need updating - Fortran 1-indexing with implicit groups indexed first, followed by general groups, i.e. if there are 3 total implicit groups, group 4 will be the first general group (the  total number of implicit groups should be determined at the level of ModelCollection). Defaults to None.
            evaluatedGroups (Union[List[int], None], optional): Groups in the model whose contributions are evaluated and used in equations- Fortran 1-indexing with implicit groups indexed first, followed by general groups . Defaults to None.
            updateModelData (bool, optional): True if the modelbound data should be updated in the step containing this rule. Defaults to True.
        """
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
    def updatedGroups(self, groups: List[int]):
        self.__updatedGroups__ = groups

    @property
    def evaluatedGroups(self):
        return self.__evaluatedGroups__

    @evaluatedGroups.setter
    def evaluatedGroups(self, groups: List[int]):
        self.__evaluatedGroups__ = groups

    @property
    def updateModelData(self):
        return self.__updateModelData__

    @updateModelData.setter
    def updateModelData(self, update: bool):
        self.__updateModelData__ = update

    def defaultGroups(self, implicitGroups: int):
        """Set default evaluated groups to all active implicit and general groups in the model

        Args:
            implicitGroups (int): Total number of implicit groups in the ReMKiT1D simulation
        """
        if self.evaluatedGroups is None:
            self.evaluatedGroups = self.__model__.activeImplicitGroups + [
                implicitGroups + g for g in self.__model__.activeGeneralGroups
            ]

    def latex(self) -> str:
        expression = "Evolved model: " + self.__model__.name.replace("_", r"\_")
        expression += "\\newline Evaluated groups: " + ",".join(
            str(group) for group in self.evaluatedGroups
        )
        expression += "\\newline Updated groups: " + ",".join(
            str(group) for group in self.updatedGroups
        )
        expression += (
            "\\newline Update modelbound data (if any): " + "Yes"
            if self.updateModelData
            else "No"
        )

        return expression


class IntegrationStepBase(ABC):
    """Abstract base class for integration steps"""

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def stepFraction(self):
        pass

    @abstractmethod
    def rename(self, name: str) -> Self:
        pass

    @abstractmethod
    def addLatexToDoc(self, doc: tex.Document, implicitGroups: int, **kwargs):
        pass

    @abstractmethod
    def only(self, *args: mc.Model) -> Self:
        pass


class IntegrationStepSequence:
    """Container class for integration steps, allowing for their chaining using the multiplication operator"""

    def __init__(self, *args: IntegrationStepBase):
        self.__steps__: List[IntegrationStepBase] = list(args)

    @property
    def steps(self):
        return self.__steps__

    @steps.setter
    def steps(self, steps: List[IntegrationStepBase]):
        self.__steps__ = steps

    def __mul__(self, rhs: Union[IntegrationStepBase, Self]) -> Self:
        assert isinstance(
            rhs, (IntegrationStepBase, IntegrationStepSequence)
        ), "__mul__ rhs of IntegrationStepSequence must be IntegrationStep or IntegrationSequence"
        if isinstance(rhs, IntegrationStepBase):
            newSequence = deepcopy(self)
            newSequence.__steps__ = [
                rhs.rename(rhs.name + str(len(self.steps)))
            ] + newSequence.__steps__
        if isinstance(rhs, IntegrationStepSequence):
            newSequence = cast(Self, deepcopy(rhs))
            newSequence.__steps__ += self.__steps__
        return cast(Self, newSequence)

    def __rmul__(self, lhs: IntegrationStepBase) -> Self:
        assert isinstance(
            lhs, IntegrationStepBase
        ), "__rmul__ lhs of IntegrationStepSequence must be IntegrationStep"
        newSequence = deepcopy(self)
        newSequence.__steps__ += [lhs.rename(lhs.name + str(len(self.steps)))]
        return newSequence

    def addLatexToDoc(
        self,
        doc: tex.Document,
        implicitGroups: int,
        models: mc.ModelCollection,
        **kwargs
    ):
        doc.append(
            tex.NoEscape(
                "Scheme: $"
                + "".join(
                    "\\text{"
                    + step.name
                    + "}("
                    + numToScientificTex(step.stepFraction, removeUnity=True)
                    + "dt)"
                    for step in reversed(self.steps)
                )
                + "$"
            )
        )
        for step in self.steps:
            with doc.create(tex.Subsection(step.name)):
                filteredStep = step.only(*models.models)
                filteredStep.addLatexToDoc(doc, implicitGroups, **kwargs)


class Rules:
    """Container for integration rules providing Model-based indexing"""

    def __init__(self) -> None:
        self.__rules__: List[IntegrationRule] = []

    def __getitem__(self, model: mc.Model):
        if model.name not in self.evolvedModels:
            raise KeyError()
        return self.__rules__[self.evolvedModels.index(model.name)]

    def __setitem__(self, model: mc.Model, rule: IntegrationRule):
        if model.name not in self.evolvedModels:
            self.__rules__.append(rule)
            return
        self.__rules__[self.evolvedModels.index(model.name)] = rule

    def __delitem__(self, model: mc.Model):
        if model.name not in self.evolvedModels:
            raise KeyError()
        del self.__rules__[self.evolvedModels.index(model.name)]

    @property
    def evolvedModels(self):
        return [m.modelName for m in self.__rules__]

    def only(self, *args: mc.Model) -> Self:
        """Filter to include only rules for given Models"""
        newRules = Rules()
        argNames = [arg.name for arg in args]
        for rule in self.__rules__:
            if rule.modelName in argNames:
                newRules.__rules__.append(rule)
            else:
                warnings.warn(
                    "Model "
                    + rule.modelName
                    + " excluded from integration rules - not present in filtering models"
                )

        return cast(Self, newRules)


class IntegrationStep(IntegrationStepBase):
    """Class containing integration step data"""

    def __init__(self, name: str, integrator: Integrator, **kwargs) -> None:
        """Integration step comprised of an integrator and integration rules

        Args:
            name (str): Name of the integration step
            integrator (Integrator): Integrator used for this integtion step

        kwargs:

            globalStepFraction (float): The fraction of the global timestep taken by this integration step. This can be set using __call__ and composing this step with others using multiplication. Defaults to 1.0

            allowTimeEvolution (bool): If true, the time variable is evolved at the end of the step, otherwise the value of the time variable at the end of the step is reverted back to the value at the start of the step. Defaults to True

            useInitialInput (bool): If true, and if this step is in a sequence of multiple steps, it will treat as its initial condition the state of the system at the start of the sequence, otherwise it will continue where the previous step in the sequence stopped. Defaults to False.
        """
        self.__name__ = name
        self.__integrator__ = integrator
        self.__globalStepFraction__ = kwargs.get("globalStepFraction", 1.0)
        self.__allowTimeEvolution__ = kwargs.get("allowTimeEvolution", True)
        self.__useInitialInput__ = kwargs.get("useInitialInput", False)

        self.rules = Rules()

    @property
    def evolvedModels(self):
        return self.rules.evolvedModels

    @property
    def name(self):
        return self.__name__

    def rename(self, name: str):
        """Return a copy of this integration step with a different name

        Args:
            name (str): New name for the step
        """
        newStep = deepcopy(self)
        newStep.__name__ = name
        return newStep

    def __call__(self, step: float):
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
        """Disable the time evolution in this step and return it"""
        self.__allowTimeEvolution__ = False
        return self

    def enableTimeEvo(self):
        """Enable the time evolution in this step and return it"""
        self.__allowTimeEvolution__ = True
        return self

    def startFromZero(self):
        """Set this step to start from the initial state at the start of the integration step sequence and return it"""
        self.__useInitialInput__ = True
        return self

    def startFromLast(self):
        """Set this step to start from the state after the previous step in the integration step sequence and return it"""
        self.__useInitialInput__ = False
        return self

    @property
    def integrator(self):
        return self.__integrator__

    @integrator.setter
    def integrator(self, integ: Integrator):
        self.__integrator__ = integ

    def add(self, *args: Union[IntegrationRule, mc.Model, mc.ModelCollection]) -> None:
        """Add any number of integration rules, models, or model collections to this this. If models or model collections are added, their integration rules will be constructed with default options."""
        for arg in args:
            assert isinstance(
                arg, (mc.Model, IntegrationRule, mc.ModelCollection)
            ), "__add__ RHS for Integration step must be Model, ModelCollection, or IntegrationRule"
            if isinstance(arg, mc.Model):
                self.rules[arg] = IntegrationRule(arg)
            if isinstance(arg, IntegrationRule):
                self.rules[arg.__model__] = arg
            if isinstance(arg, mc.ModelCollection):
                for model in arg.models:
                    self.add(model)

    def only(self, *args: mc.Model) -> Self:
        """Filter integration step rules to only include those for given models"""
        newStep = deepcopy(self)
        newStep.rules = self.rules.only(*args)

        return cast(Self, newStep)

    def __mul__(
        self, rhs: Union[Self, IntegrationStepSequence]
    ) -> IntegrationStepSequence:
        assert isinstance(
            rhs, (IntegrationStep, IntegrationStepSequence)
        ), "__mul__ rhs for IntegrationStep must be IntegrationStep or IntegrationStepSequence"

        if isinstance(rhs, IntegrationStep):
            return self * (rhs * IntegrationStepSequence())

        return rhs.__rmul__(self)

    def dict(self, implicitGroups: int) -> dict:

        step = {
            "integratorTag": self.integrator.name,
            "evolvedModels": self.evolvedModels,
            "globalStepFraction": self.__globalStepFraction__,
            "allowTimeEvolution": self.__allowTimeEvolution__,
            "useInitialInput": self.__useInitialInput__,
        }

        for rule in self.rules.__rules__:
            rule.defaultGroups(implicitGroups)
            step.update(
                {
                    rule.modelName: {
                        "groupIndices": (rule.evaluatedGroups),
                        "internallyUpdatedGroups": (rule.updatedGroups),
                        "internallyUpdateModelData": (rule.updateModelData),
                    }
                }
            )

        return step

    def addLatexToDoc(self, doc: tex.Document, implicitGroups: int, **kwargs):
        doc.append(
            tex.NoEscape(
                "Evolving time: " + ("Yes" if self.__allowTimeEvolution__ else "No")
            )
        )
        doc.append(
            tex.NoEscape(
                "\\newline Starting from initial state: "
                + ("Yes" if self.__useInitialInput__ else "No")
            )
        )
        with doc.create(tex.Subsubsection("Integrator")):
            self.integrator.addLatexToDoc(doc, **kwargs)
        with doc.create(tex.Subsubsection("Integration rules")):
            with doc.create(tex.Itemize()) as itemize:
                for rule in self.rules.__rules__:
                    rule.defaultGroups(implicitGroups)
                    itemize.add_item(tex.NoEscape(rule.latex()))


class Timestep:
    """Wrapper for the setting of the integration timestep, including scaling controls"""

    def __init__(self, timestep: Union[Variable, MultiplicativeArgument, float]):
        """Wrapper for the setting of the integration timestep, including scaling controls

        Args:
            timestep (Union[Variable, MultiplicativeArgument, float]): Timestep value - if the timestep is a float, it will be kept constant, otherwise it will be evaluated on the spatial grid and the minimum (or maximum, see below) value will be used as the step value
        """
        self.__timestep__: MultiplicativeArgument = MultiplicativeArgument() * timestep
        self.__max__ = False

    @property
    def usingMaxVal(self):
        return self.__max__

    def max(self) -> Self:
        """Set the Timestep to using the maximum value of the evaluated quantities, and return the Timestep"""
        self.__max__ = True
        return self

    def min(self) -> Self:
        """Set the Timestep to using the minimum value of the evaluated quantities, and return the Timestep"""
        self.__max__ = False
        return self

    def dict(self) -> dict:
        return {
            "timestepController": {
                "active": len(self.__timestep__.args) > 0,
                "rescaleTimestep": True,
                "requiredVarNames": list(self.__timestep__.argMultiplicity.keys()),
                "requiredVarPowers": [
                    item for _, item in self.__timestep__.argMultiplicity.items()
                ],
                "multConst": 1.0,
                "useMaxVal": self.__max__,
            },
            "initialTimestep": self.__timestep__.scalar,
        }

    def checkConsistency(self, varCont: VariableContainer):

        for _, var in self.__timestep__.args.items():
            assert var.name in varCont.varNames, (
                "Timestep scaling variable "
                + var
                + " not found in used variable container"
            )

    def latex(self: Self, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        if len(self.__timestep__.args) > 0:
            if self.__max__:
                return (
                    "\\text{max}\\left("
                    + numToScientificTex(self.__timestep__.scalar)
                    + self.__timestep__.latex(latexRemap)
                    + "\\right)"
                )
            return (
                "\\text{min}\\left("
                + numToScientificTex(self.__timestep__.scalar)
                + self.__timestep__.latex(latexRemap)
                + "\\right)"
            )
        return numToScientificTex(self.__timestep__.scalar)


class IntegrationScheme:
    """Integration scheme to be used in a ReMKiT1D simulation"""

    def __init__(
        self,
        dt: Union[Timestep, float],
        steps: Union[
            IntegrationStepSequence, IntegrationStep
        ] = IntegrationStepSequence(),
    ):
        """Integration scheme to be used in a ReMKiT1D simulation encapsulating both the steps and the rule for calculating the timestep

        The scheme defaults to doing a single step and outputting the values after it. See member functions for setting the mode and options for the timestepping/output.

        Args:
            dt (Union[Timestep, float]): Timestep length.
            steps (Union[ IntegrationStepSequence, IntegrationStep ], optional): Integration step sequence or a single integration step (the only step). Defaults to IntegrationStepSequence() which is an empty sequence.
        """
        self.__timestep__ = dt if isinstance(dt, Timestep) else Timestep(dt)
        self.__stepSequence__: IntegrationStepSequence = (
            steps
            if isinstance(steps, IntegrationStepSequence)
            else IntegrationStepSequence(steps)
        )

        self.__mode__ = "fixedNumSteps"
        self.__numTimesteps__ = 1
        self.__outputInterval__ = 1

        self.__outputPoints__: List[float] = []

    @property
    def timestep(self):
        return self.__timestep__

    @timestep.setter
    def timestep(self, dt=Union[Timestep, float]):
        self.__timestep__ = dt if isinstance(dt, Timestep) else Timestep(dt)

    @property
    def steps(self):
        """Get the individual steps in the step sequence of the scheme"""
        return self.__stepSequence__.steps

    @steps.setter
    def steps(self, seq: IntegrationStepSequence):
        self.__stepSequence__ = seq

    def setFixedNumTimesteps(self, numTimesteps=1, outputInterval=1):
        """Set the mode of the integration scheme to running for a fixed number of timesteps and outputting every set number of timesteps

        Args:
            numTimesteps (int, optional): Number of timesteps to run for. Defaults to 1.
            outputInterval (int, optional): Number of timesteps between outputting variable data. Defaults to 1.
        """
        self.__mode__ = "fixedNumSteps"
        self.__numTimesteps__ = numTimesteps
        self.__outputInterval__ = outputInterval

    def setOutputPoints(self, outputPoints: List[float]):
        """Set the timestepping mode to output at particular values of the time variable. Step lengths are adjusted before output points, but never below the set Timestep values

        Args:
            outputPoints (List[float]): List of positive monotonically increasing values of the time variable at which the code should produce output
        """
        assert all(
            point > 0 for point in outputPoints
        ), "All output points must be positive"
        assert all(
            np.diff(np.array(outputPoints)) > 0
        ), "outputPoints must be monotonically increasing sequence"

        self.__outputPoints__ = outputPoints
        self.__mode__ = "outputDriven"

    def dict(
        self, implicitGroups: int, mpiComm: dict, models: mc.ModelCollection
    ) -> dict:

        scheme: Dict[str, object] = {
            "stepTags": [step.name for step in self.steps],
            "integratorTags": list(set(step.integrator.name for step in self.steps)),
        }
        scheme.update(self.__timestep__.dict())
        for step in self.steps:
            scheme[step.name] = {"commData": mpiComm}
            filteredStep = step.only(*models.models)
            cast(Dict, scheme[step.name]).update(filteredStep.dict(implicitGroups))
            scheme.update({step.integrator.name: step.integrator.dict()})

        timeloop = {
            "mode": self.__mode__,
            "numTimesteps": self.__numTimesteps__,
            "fixedSaveInterval": self.__outputInterval__,
            "outputPoints": self.__outputPoints__,
        }
        return {"integrator": scheme, "timeloop": timeloop}

    def addLatexToDoc(
        self,
        doc: tex.Document,
        implicitGroups: int,
        models: mc.ModelCollection,
        **kwargs
    ):
        with doc.create(tex.Section("Integration scheme")):
            doc.append(tex.NoEscape("$dt =" + self.timestep.latex(**kwargs) + " $ "))
            if self.__mode__ == "fixedNumSteps":
                doc.append(
                    tex.NoEscape(
                        "\\newline Running for "
                        + str(self.__numTimesteps__)
                        + " steps, outputting every "
                        + str(self.__outputInterval__)
                    )
                )
            else:
                doc.append(
                    tex.NoEscape(
                        "\\newline Output points: "
                        + ", ".join(
                            "$" + numToScientificTex(point, decimals=4) + "$"
                            for point in self.__outputPoints__
                        )
                    )
                )
            doc.append(
                tex.NoEscape(
                    "\\newline Total number of implicit groups: " + str(implicitGroups)
                )
            )
            with doc.create(tex.Subsection("Integration steps")):
                self.__stepSequence__.addLatexToDoc(
                    doc, implicitGroups, models, **kwargs
                )


class BDEIntegrator(Integrator):
    """Backwards difference Euler integrator with fixed-point iterations used to solve non-linear systems"""

    def __init__(self, name: str, **kwargs):
        """Backwards difference Euler integrator with fixed-point iterations used to solve non-linear systems

        Args:
            name (str): Name of the integrator

        kwargs:

        maxNonlinIters (int): Maximum allowed nonlinear (Picard/fixed point) iterations. Defaults to 100.

        nonlinTol (float): Relative convergence tolerance on 2-norm. Defaults to 1.0e-12.
        absTol (float): Absolute tolerance in machine precision units (epsilon in Fortran - 2.22e-16 for double precision). Defaults to 1.0.

        convergenceVars (List[str]): Variables used to check for convergence. Defaults to [], which results in all implicit variables.

        associatedPETScGroup (int): PETSc object group this integrator is associated with. Defaults to 1.

        use2Norm (bool): True if 2-norm should be used (benefits distributions). Defaults to False.

        internalStepControl (bool): True if integrator is allowed to control its internal steps based on convergence. Defaults to False.

        initialNumInternalSteps (int): Initial number of integrator substeps. Defaults to 1.

        stepMultiplier (int): Factor by which to multiply current number of substeps when solve fails. Defaults to 2.

        stepDecrament (int): How much to reduce the current number of substeps if nonlinear iterations are below minNonlinIters. Defaults to 1.

        minNonlinIters (int): Number of nonlinear iterations under which the integrator should attempt to reduce the number of internal steps. Defaults to 5.

        maxBDERestarts (int): Maximum number of solver restarts with step splitting. Defaults to 3. Note that there is a hard limit of 10.

        relaxationWeight (float): Relaxation weight for the Picard iteration (relaxationWeight * newValues + (1-relaxationWeight)*oldValues). Defaults to 1.0.
        """
        super().__init__(name)
        self.__maxNonlinIters__: int = kwargs.get("maxNonlinIters", 100)
        self.__nonlinTol__: float = kwargs.get("nonlinTol", 1.0e-12)
        self.__absTol__: float = kwargs.get("absTol", 1.0)
        self.__convergenceVars__: List[Variable] = kwargs.get("convergenceVars", [])
        self.__associatedPETScGroup__: int = kwargs.get("associatedPETScGroup", 1)
        self.__use2Norm__: bool = kwargs.get("use2Norm", False)

        self.__internalStepControl__: bool = kwargs.get("internalStepControl", False)

        self.__initialNumInternalSteps__: int = kwargs.get("initialNumInternalSteps", 1)
        self.__stepMultiplier__: int = kwargs.get("stepMultiplier", 2)
        self.__stepDecrament__ = kwargs.get("stepDecrament", 1)
        self.__minNonlinIters__ = kwargs.get("minNonlinIters", 5)
        self.__consolidationInterval__ = kwargs.get("consolidationInterval", 50)
        self.__maxBDERestarts__ = kwargs.get("maxBDERestarts", 3)
        self.__relaxationWeight__: float = kwargs.get("relaxationWeight", 1.0)

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
        doc.append(
            self.name + ": Backwards Euler integrator with fixed-point iterations"
        )


class RKIntegrator(Integrator):
    """Runge-Kutta integrator of given order (supports 1-4 currently)"""

    def __init__(self, name: str, order: int):
        """Runge-Kutta integrator of given order

        Args:
            name (str): Name of the integrator
            order (int): RK order (1-4 supported)
        """
        self.__order__ = order
        super().__init__(name)

    def dict(self) -> Dict:
        return {"type": "RK", "order": self.__order__}

    def addLatexToDoc(self, doc, **kwargs):
        doc.append(
            self.name + ": Runge-Kutta integrator - order " + str(self.__order__)
        )


class CVODEIntegrator(Integrator):
    """CVODE integrator See https://sundials.readthedocs.io/en/latest/index.html"""

    def __init__(self, name: str, **kwargs):
        """CVODE integrator See https://sundials.readthedocs.io/en/latest/index.html

        Args:
            name (str): Name of the integrator

        kwargs:

        relTol (float): CVODE solver relative tolerance. Defaults to 1e-5.

        absTol (float): CVODE solver absolute tolerance. Defaults to 1e-10.

        maxGMRESRestarts (int): SPGMR maximum number of restarts. Defaults to 0.

        CVODEBBDPreParams (List[int]): BBD preconditioner parameters in order [mudq,mldq,mukeep,mlkeep]. Defaults to [0,0,0,0].

        useAdamsMoulton (bool): If true will use Adams Moulton method instead of the default BDF. Defaults to False.

        useStabLimitDet (bool): If true will use stability limit detection. Defaults to False.

        maxOrder (int): Maximum integrator order (set to BDF default, AM default is 12). Defaults to 5.

        maxInternalStep (int): Maximum number of internal CVODE steps per ReMKiT1D timestep. Defaults to 500.

        minTimestep (float): Minimum allowed internal timestep. Defaults to 0.0.

        maxTimestep (float): Maximum allowed internal timestep. Defaults to 0.0, resulting in no limit.

        initTimestep (float): Initial internal timestep. Defaults to 0.0, letting CVODE decide.
        """
        self.__relTol__: float = kwargs.get("relTol", 1e-5)
        self.__absTol__: float = kwargs.get("absTol", 1e-10)
        self.__maxGMRESRestarts__: int = kwargs.get("maxGMRESRestarts", 0)
        self.__CVODEBBDPreParams__: Tuple[int, int, int, int] = kwargs.get(
            "CVODEBBDPreParams", (0, 0, 0, 0)
        )
        self.__useAdamsMoulton__: bool = kwargs.get("useAdamsMoulton", False)
        self.__useStabLimitDet__: bool = kwargs.get("useStabLimitDet", False)
        self.__maxOrder__: int = kwargs.get("maxOrder", 5)
        self.__maxInternalStep__: int = kwargs.get("maxInternalStep", 500)
        self.__minTimestep__: float = kwargs.get("minTimestep", 0.0)
        self.__maxTimestep__: float = kwargs.get("maxTimestep", 0.0)
        self.__initTimestep__: float = kwargs.get("initTimestep", 0.0)
        super().__init__(name)

    def dict(self) -> Dict:
        return {
            "type": "CVODE",
            "relTol": self.__relTol__,
            "absTol": self.__absTol__,
            "maxRestarts": self.__maxGMRESRestarts__,
            "CVODEPreBBDParams": list(self.__CVODEBBDPreParams__),
            "CVODEUseAdamsMoulton": self.__useAdamsMoulton__,
            "CVODEUseStabLimDet": self.__useStabLimitDet__,
            "CVODEMaxOrder": self.__maxOrder__,
            "CVODEMaxInternalSteps": self.__maxInternalStep__,
            "CVODEMaxStepSize": self.__maxTimestep__,
            "CVODEMinStepSize": self.__minTimestep__,
            "CVODEInitStepSize": self.__initTimestep__,
        }

    def addLatexToDoc(self, doc, **kwargs):
        doc.append(self.name + ": CVODE Integrator")
