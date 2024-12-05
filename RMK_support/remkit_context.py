import numpy as np
from .grid import Grid
from typing import Union, List, Dict, cast, Tuple, Optional
from typing_extensions import Self
from .variable_container import VariableContainer, Variable, MPIContext
from . import IO_support as io
from . import derivations as dv
from . import integrators as it
from . import model_construction as mc
from . import sk_normalization as skn
import pylatex as tex  # type: ignore
from pylatex.utils import bold  # type: ignore
import warnings
import os
from abc import ABC, abstractmethod


class IOContext:

    def __init__(
        self,
        jsonFilepath: str = "./config.json",
        HDF5Dir: str = "./RMKOutput/",
        **kwargs,
    ):

        self.__jsonFilepath__ = jsonFilepath
        self.__HDF5Dir__ = HDF5Dir

        self.__outputVars__: List[Variable] = []
        self.__inputVars__: List[Variable] = []
        self.__inputHDF5File__: Union[str, None] = kwargs.get("initValFilename", None)

        self.__restartSave__: bool = kwargs.get("restartSave", False)
        self.__restartLoad__: bool = kwargs.get("restartLoad", False)
        self.__restartFreq__: int = kwargs.get("restartFrequency", 1000)
        self.__restartResetTime__: bool = kwargs.get("restartResetTime", False)
        self.__restartInitialOutputIndex__: int = kwargs.get(
            "restartInitialOutputIndex", 0
        )

    @property
    def jsonFilepath(self):
        return self.__jsonFilepath__

    @jsonFilepath.setter
    def jsonFilepath(self, jsonpath: str):
        self.__jsonFilepath__ = jsonpath

    @property
    def HDF5Dir(self):
        return self.__HDF5Dir__

    @HDF5Dir.setter
    def HDF5Dir(self, dir: str):
        self.__HDF5Dir__ = dir

    def setRestartOptions(self, **kwargs) -> None:
        if "save" in kwargs:
            self.__restartSave__ = cast(bool, kwargs.get("save"))

        if "load" in kwargs:
            self.__restartLoad__ = cast(bool, kwargs.get("load"))

        if "frequency" in kwargs:
            self.__restartFreq__ = cast(int, kwargs.get("frequency"))

        if "resetTime" in kwargs:
            self.__restartResetTime__ = cast(bool, kwargs.get("resetTime"))

        if "initialOutputIndex" in kwargs:
            self.__restartInitialOutputIndex__ = cast(
                int, kwargs.get("initialOutputIndex")
            )

    def populateOutputVars(self, variables: VariableContainer):

        self.__outputVars__ = []
        for var in variables.varNames:
            if variables[var].inOutput:
                self.__outputVars__.append(variables[var])

    def setHDF5InputOptions(
        self, inputFile: Union[str, None], inputVars: List[Variable] = []
    ):
        self.__inputHDF5File__ = inputFile
        self.__inputVars__ = inputVars

    def dict(self) -> dict:

        IODict = {
            "HDF5": {
                "outputVars": [var.name for var in self.__outputVars__],
                "filepath": self.HDF5Dir,
                "inputVars": [var.name for var in self.__inputVars__],
            },
            "timeloop": {
                "restart": {
                    "save": self.__restartSave__,
                    "load": self.__restartLoad__,
                    "frequency": self.__restartFreq__,
                    "resetTime": self.__restartResetTime__,
                    "initialOutputIndex": self.__restartInitialOutputIndex__,
                },
                "loadInitValsFromHDF5": self.__inputHDF5File__ is not None,
                "initValFilename": self.__inputHDF5File__,
            },
        }

        return IODict


class Manipulator(ABC):

    def __init__(self, name: str, priority: int = 4) -> None:
        self.__name__ = name
        self.__priority__ = priority

    @property
    def name(self):
        return self.__name__

    @property
    def priority(self):
        return self.__priority__

    @abstractmethod
    def dict(self) -> Dict:
        pass

    @abstractmethod
    def latex(self, **kwargs) -> str:
        pass


class GroupEvaluator(Manipulator):

    def __init__(
        self,
        name: str,
        model: mc.Model,
        termGroup: int,
        resultVar: Variable,
        priority: int = 4,
    ) -> None:
        super().__init__(name, priority)
        self.__model__ = model
        self.__termGroup__ = termGroup
        assert (
            resultVar.isDerived and resultVar.derivation is None
        ), "resultVar in GroupEvaluator must be derived and without an assigned derivation"

        self.__resultVar__ = resultVar

    def dict(self) -> Dict:
        manip = {
            "type": "groupEvaluator",
            "modelTag": self.__model__.name,
            "evaluatedTermGroup": self.__termGroup__,
            "resultVarName": self.__resultVar__.name,
            "priority": self.priority,
        }

        return manip

    def latex(self, **kwargs) -> str:
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        resultVarName = (
            latexRemap[self.__resultVar__.name]
            if self.__resultVar__.name in latexRemap
            else "\\text{" + self.__resultVar__.name.replace("_", "\_") + "}"
        )

        return (
            resultVarName
            + " = \\text{Eval}_{\\text{group}="
            + str(self.__termGroup__)
            + "}\\left("
            + self.__model__.latexName
            + "\\right)"
        )


class TermEvaluator(Manipulator):

    def __init__(
        self,
        name: str,
        modelTermTags: List[Tuple[str, str]],
        resultVar: Variable,
        accumulate=False,
        update=False,
        priority: int = 4,
    ) -> None:
        super().__init__(name, priority)
        self.__resultVar__ = resultVar
        assert (
            resultVar.isDerived and resultVar.derivation is None
        ), "resultVar in GroupEvaluator must be derived and without an assigned derivation"

        self.__modelTermTags__ = modelTermTags
        self.__accumulate__ = accumulate
        self.__update__ = update

    def dict(self) -> Dict:
        models, terms = zip(*self.__modelTermTags__)
        manip = {
            "type": "termEvaluator",
            "evaluatedModelNames": list(models),
            "evaluatedTermNames": list(terms),
            "resultVarName": self.__resultVar__.name,
            "priority": self.priority,
            "update": self.__update__,
            "accumulate": self.__accumulate__,
        }

        return manip

    def latex(self, **kwargs) -> str:
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        resultVarName = (
            latexRemap[self.__resultVar__.name]
            if self.__resultVar__.name in latexRemap
            else "\\text{" + self.__resultVar__.name.replace("_", "\_") + "}"
        )
        accu = "_{accu}" if self.__accumulate__ else ""
        return (
            resultVarName
            + " = \\text{Eval}"
            + accu
            + "\\left(\\text{"
            + ",".join(
                model.replace("_", "\_") + "-" + term.replace("_", "\_")
                for model, term in self.__modelTermTags__
            )
            + "}\\right)"
        )


class MBDataExtractor(Manipulator):

    def __init__(
        self,
        name,
        model: mc.Model,
        mbVar: Variable,
        resultVar: Optional[Variable] = None,
        priority=4,
    ):
        super().__init__(name, priority)
        self.__model__ = model
        assert model.mbData is not None, "MBDataExtractor model does not have mbData"
        assert (
            mbVar.name in model.mbData.varNames
        ), "mbVar in MBDataExtractor not in passed model mbData"

        self.__resultVar__ = resultVar
        self.__mbVar__ = mbVar

    def dict(self):

        manip = {
            "type": "modelboundDataExtractor",
            "modelTag": self.__model__.name,
            "modelboundDataName": self.__mbVar__.name,
            "resultVarName": (
                self.__mbVar__.name
                if self.__resultVar__ is None
                else self.__resultVar__.name
            ),
            "priority": self.priority,
        }

        return manip

    def latex(self: Self, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        resultVarName = (
            latexRemap[self.__mbVar__.name]
            if self.__mbVar__.name in latexRemap
            else "\\text{" + self.__mbVar__.name.replace("_", "\_") + "}"
        )

        return (
            resultVarName
            + " = \\text{MBExtract}\\left("
            + self.__model__.latexName
            + "\\right)"
        )


class ManipulatorCollection:

    def __init__(self: Self):
        self.__manipulators__: List[Manipulator] = []

    @property
    def manipNames(self):
        return [m.name for m in self.__manipulators__]

    @property
    def manipulators(self):
        return self.__manipulators__

    def __getitem__(self, key: str):
        if key not in self.manipNames:
            raise KeyError()
        return self.__manipulators__[self.manipNames.index(key)]

    def __setitem__(self, key: str, manip: Manipulator):
        if key not in self.manipNames:
            self.__manipulators__.append(manip)
        else:
            self.__manipulators__[self.manipNames.index(key)] = manip

    def __delitem__(self, key: str):
        if key not in self.manipNames:
            raise KeyError()
        del self.__manipulators__[self.manipNames.index(key)]

    def add(self, *args: Manipulator):
        for manip in args:
            assert manip.name not in self.manipNames, (
                "Duplicate Manipulator name " + manip.name
            )

            self.__manipulators__.append(manip)

    def dict(self):

        manips = {"tags": self.manipNames}
        for manip in self.manipulators:
            manips[manip.name] = manip.dict()

        return manips

    def addLatexToDoc(self, doc: tex.Document, **kwargs):
        if len(self.manipulators) > 0:
            with doc.create(tex.Section("Manipulators")):
                with doc.create(tex.Itemize()) as itemize:
                    for manip in self.manipulators:
                        itemize.add_item(
                            tex.NoEscape(
                                manip.name.replace("_", "\_")
                                + f": \\newline ${manip.latex(**kwargs)}$"
                            )
                        )


class RMKContext:
    def __init__(self) -> None:

        self.__normDens__: float = 1e19
        self.__normTemp__: float = 10
        self.__normZ__: float = 1

        self.__gridObj__: Union[None, Grid] = None

        self.__textbook__: Union[None, dv.Textbook] = None

        self.__species__ = dv.SpeciesContainer()

        self.__variables__: Union[None, VariableContainer] = None

        self.__mpiContext__ = MPIContext(1)

        self.__IOContext__ = IOContext()

        self.__optionsPETSc__: Dict[str, object] = {
            "active": True,  # Sets whether PETSc should be included (set to False if not using any PETSc functionality)
            "solverOptions": {
                "solverToleranceRel": 0.1e-16,  # Relative tolerance for Krylov solver
                "solverToleranceAbs": 1.0e-20,  # Absolute tolerance
                "solverToleranceDiv": 0.1e8,  # Divergence tolerance
                "maxSolverIters": 10000,  # Maximum allowed Krylov iterations
                "kspSolverType": "bcgs",  # Solver type (see PETSc documentation)
                "hyprePCType": "",  # Hypre PC type
                "PETScCommandLineOpts": "-pc_type bjacobi -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1",  # Command line stype PETSc options
            },
            "objGroups": 1,  # Number of PETSc objects groups to construct (different matrix contexts)
        }

        self.__models__: Union[None, mc.ModelCollection] = None

        self.__manipulators__ = ManipulatorCollection()
        self.__integrationScheme__: Union[it.IntegrationScheme, None] = None

    @property
    def normDensity(self):
        return self.__normDens__

    @normDensity.setter
    def normDensity(self, norm: float):
        self.__normDens__ = norm

    @property
    def normTemperature(self):
        return self.__normTemp__

    @normTemperature.setter
    def normTemperature(self, norm: float):
        self.__normTemp__ = norm

    @property
    def normZ(self):
        return self.__normZ__

    @normZ.setter
    def normZ(self, norm: float):
        self.__normZ__ = norm

    @property
    def norms(self):
        return skn.calculateNorms(self.__normTemp__, self.__normDens__, self.__normZ__)

    @property
    def grid(self):
        return self.__gridObj__

    @grid.setter
    def grid(self, grid: Grid):
        self.__gridObj__ = grid

    @property
    def textbook(self):
        if self.__textbook__ is None:
            assert (
                self.grid is not None
            ), "Cannot auto-initialise context textbook without grid"
            self.__textbook__ = dv.Textbook(self.grid)
        return self.__textbook__

    @textbook.setter
    def textbook(self, tb: dv.Textbook):
        self.__textbook__ = tb

    @property
    def species(self):
        return self.__species__

    @species.setter
    def species(self, sp: dv.SpeciesContainer):
        self.__species__ = sp

    @property
    def variables(self):
        if self.__variables__ is None:
            assert (
                self.grid is not None
            ), "Cannot auto-initialise context variable container without grid"
            self.__variables__ = VariableContainer(self.grid)
        return self.__variables__

    @variables.setter
    def variables(self, vc: VariableContainer):
        self.__variables__ = vc

    @property
    def mpiContext(self):
        return self.__mpiContext__

    @mpiContext.setter
    def mpiContext(self, context: MPIContext):
        self.__mpiContext__ = context

    @property
    def optionsPETSc(self):
        return self.__optionsPETSc__

    @property
    def models(self):
        if self.__models__ is None:
            self.__models__ = mc.ModelCollection()
        return self.__models__

    @models.setter
    def models(self, models: mc.ModelCollection):
        self.__models__ = models

    @property
    def integrationScheme(self):
        return self.__integrationScheme__

    @integrationScheme.setter
    def integrationScheme(self, scheme: it.IntegrationScheme):
        self.__integrationScheme__ = scheme

    @property
    def IOContext(self):
        return self.__IOContext__

    @IOContext.setter
    def IOContext(self, context: IOContext):
        self.__IOContext__ = context

    @property
    def manipulators(self):
        return self.__manipulators__

    @manipulators.setter
    def manipulators(self, manips: ManipulatorCollection):
        self.__manipulators__ = manips

    def setPETScOptions(
        self,
        active=True,
        relTol=0.1e-16,
        absTol=1.0e-20,
        divTol=0.1e8,
        maxIters=10000,
        kspSolverType="bcgs",
        hyprePC="",
        cliOpts="-pc_type bjacobi -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1",
        objGroups=1,
    ) -> None:
        """Set PETSc solver options

        Args:
            active (bool, optional): Set to true if the PETSc object is built - turn off if no implicit terms in any model. Defaults to True.
            relTol (float, optional): Solver relative tolerance. Defaults to 0.1e-16.
            absTol (float, optional): Solver absolute tolerance. Defaults to 1.0e-20.
            divTol (float, optional): Solver divergence tolerance. Defaults to 0.1e+8.
            maxIters (int, optional): Maximum allowed number of KSP iterations. Defaults to 10000.
            kspSolverType (str, optional): Type of KSP solver used. Defaults to "bcgs".
            hyprePC (str, optional): Type of Hypre preconditioner used. Defaults to "euclid".
            cliOpts (str, optional): Optional command line PETSc options. Defaults to "".
            objGroups (int, optional): Number of PETSc object groups to create. This is useful when different integrators need to be associated with different models/terms. Defaults to 1.
        """

        self.__optionsPETSc__["active"] = active
        cast(Dict[str, object], self.__optionsPETSc__["solverOptions"])[
            "solverToleranceRel"
        ] = relTol
        cast(Dict[str, object], self.__optionsPETSc__["solverOptions"])[
            "solverToleranceAbs"
        ] = absTol
        cast(Dict[str, object], self.__optionsPETSc__["solverOptions"])[
            "solverToleranceDiv"
        ] = divTol
        cast(Dict[str, object], self.__optionsPETSc__["solverOptions"])[
            "maxSolverIters"
        ] = maxIters
        cast(Dict[str, object], self.__optionsPETSc__["solverOptions"])[
            "kspSolverType"
        ] = kspSolverType
        cast(Dict[str, object], self.__optionsPETSc__["solverOptions"])[
            "hyprePCType"
        ] = hyprePC
        cast(Dict[str, object], self.__optionsPETSc__["solverOptions"])[
            "PETScCommandLineOpts"
        ] = cliOpts
        self.__optionsPETSc__["objGroups"] = objGroups

    def checkAll(self):

        assert self.__gridObj__ is not None, "Grid not set"
        assert self.__variables__ is not None, "VariableContainer not set"
        assert self.__models__ is not None, "Models not set"
        assert self.__integrationScheme__ is not None, "IntegrationScheme not set"

        if self.__textbook__ is None:
            self.__textbook__ = dv.Textbook(self.grid)
        self.__variables__.registerDerivs(self.__textbook__)
        self.__variables__.checkDerivationArgs()
        self.__models__.checkModels(self.variables)
        self.__models__.registerDerivs(self.__textbook__)

    def dict(self) -> dict:
        """Convert wrapper into config.json compatible dictionary

        Returns:
            dict: Immediately writable config.json dictionary
        """

        self.checkAll()

        configFile = {
            "normalization": self.norms,
            "species": self.species.dict(),
            "MPI": self.mpiContext.dict(self.variables),
            "PETSc": self.optionsPETSc,
            "models": cast(mc.ModelCollection, self.__models__).dict(),
            "manipulators": cast(ManipulatorCollection, self.__manipulators__).dict(),
        }
        implicitGroups, generalGroups = cast(
            mc.ModelCollection, self.__models__
        ).numGroups()

        configFile.update(cast(Grid, self.__gridObj__).dict())
        configFile.update(cast(VariableContainer, self.__variables__).dict())
        self.__IOContext__.populateOutputVars(self.variables)
        configFile.update(self.__IOContext__.dict())
        configFile.update(
            cast(it.IntegrationScheme, self.__integrationScheme__).dict(
                implicitGroups, configFile["MPI"]["commData"]
            )
        )
        configFile["timeloop"].update(self.__IOContext__.dict()["timeloop"])
        configFile["integrator"].update(
            {
                "numImplicitGroups": implicitGroups,
                "numGeneralGroups": generalGroups,
            }
        )
        configFile.update(self.textbook.dict())

        return configFile

    def writeConfigFile(self) -> None:
        """Generate a config file based on the current state of the wrapper"""
        try:
            os.remove(self.IOContext.jsonFilepath)
        except FileNotFoundError:
            pass

        io.writeDictToJSON(self.dict(), filepath=self.IOContext.jsonFilepath)

    def generatePDF(
        self, latexFilename="ReMKiT1D", latexRemap: Dict[str, str] = {}, cleanTex=True
    ):

        self.checkAll()

        doc = tex.Document()

        doc.preamble.append(tex.NoEscape("\\usepackage{amsmath}"))
        doc.preamble.append(tex.NoEscape("\\usepackage{enumitem}"))
        with doc.create(tex.MiniPage(align="c")):
            doc.append(tex.LargeText(bold(latexFilename)))
            doc.append(tex.LineBreak())

        cast(VariableContainer, self.__variables__).addLatexToDoc(
            doc, latexRemap=latexRemap
        )
        cast(dv.Textbook, self.__textbook__).addLatexToDoc(doc)
        self.__species__.addLatexToDoc(doc, latexRemap=latexRemap)
        cast(mc.ModelCollection, self.__models__).addLatexToDoc(
            doc, latexRemap=latexRemap
        )
        cast(ManipulatorCollection, self.__manipulators__).addLatexToDoc(
            doc, latexRemap=latexRemap
        )
        implicitGroups, _ = cast(mc.ModelCollection, self.__models__).numGroups()
        cast(it.IntegrationScheme, self.__integrationScheme__).addLatexToDoc(
            doc, implicitGroups, latexRemap=latexRemap
        )

        doc.generate_pdf(latexFilename.replace(" ", "_"), clean_tex=cleanTex)

    def loadSimulation(
        self, onlySteps: Optional[List[int]] = None
    ) -> VariableContainer:

        filenames = [
            self.IOContext.HDF5Dir + file
            for file in io.getOutputFilenames(self.IOContext.HDF5Dir)
        ]
        if onlySteps is not None:
            filteredFiles = [
                file
                for file in filenames
                if any(
                    str(step) == file.split("/")[-1].split(".")[0].split("_")[-1]
                    for step in onlySteps
                )
            ]
            filenames = filteredFiles
        print("Loading files:")
        for file in filenames:
            print(file)

        return io.loadVarContFromHDF5(
            *(var for var in self.variables.variables if var.inOutput),
            filepaths=filenames,
        )

    def addTermDiagnostics(self, *args: Variable):

        for var in args:
            terms = self.models.getTermsThatEvolveVar(var)
            if not len(terms):
                warnings.warn(
                    "addTermDiagnostics called when variable "
                    + var.name
                    + " has no terms that evolve it"
                )

            for pair in terms:
                model, term = pair
                self.variables.add(
                    Variable(
                        model + "_" + term,
                        self.grid,
                        isDerived=True,
                        isDistribution=var.isDistribution,
                        isOnDualGrid=var.isOnDualGrid,
                        isCommunicated=False,
                    )
                )
                self.manipulators.add(
                    TermEvaluator(
                        model + "_" + term, [pair], self.variables[model + "_" + term]
                    )
                )
