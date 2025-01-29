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
    """Class containing IO-related ReMKiT1D options"""

    def __init__(
        self,
        jsonFilepath: str = "./config.json",
        HDF5Dir: str = "./RMKOutput/",
        **kwargs,
    ):
        """IO option container for ReMKiT1D

        Args:
            jsonFilepath (str, optional): JSON config file path. Defaults to "./config.json".
            HDF5Dir (str, optional): Directory for HDF5 file IO. Defaults to "./RMKOutput/".

        kwargs:

            initValFilename (str): HDF5 file containing initial values

            restartSave (bool): Set to true if restart checkpoints should be saved

            restartLoad (bool): Set to true if the run should restart from the latest checkpoint in the HDF5 directory

            restartFrequency (int): Number of timesteps between restart checkpoints

            restartResetTime (bool): If true will reset the time variable on restart

            restartInitialOutputIndex: The first output file index after restarting. Defaults to 0.
        """

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
        """
        save (bool): Set to true if restart checkpoints should be saved

        load (bool): Set to true if the run should restart from the latest checkpoint in the HDF5 directory

        frequency (int): Number of timesteps between restart checkpoints

        resetTime (bool): If true will reset the time variable on restart

        initialOutputIndex: The first output file index after restarting. Defaults to 0.
        """
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
        """Uses the options on variables in a container to populate which variables should be in the code output

        Args:
            variables (VariableContainer): Variable container in the ReMKiT1D context
        """
        self.__outputVars__ = []
        for var in variables.varNames:
            if variables[var].inOutput:
                self.__outputVars__.append(variables[var])

    def setHDF5InputOptions(
        self, inputFile: Union[str, None], inputVars: Optional[List[Variable]] = None
    ):
        """Set options for using HDF5 input files

        Args:
            inputFile (Union[str, None]): Name of the input file (without h5 extension!)
            inputVars (Optional[List[Variable]], optional): Variables to load from the input file. Defaults to None.
        """
        self.__inputHDF5File__ = inputFile
        self.__inputVars__ = inputVars if inputVars is not None else []

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
    """Abstract manipulator class"""

    def __init__(self, name: str, priority: int = 4) -> None:
        """Abstract manipulator class

        Args:
            name (str): Name of the manipulator
            priority (int, optional): Manipulator priority (0 called at every internal integrator iteration, 1 called at every integrator substep, 2 called between integration steps in a single global step, 3 called after each global integrator step, 4 called only before IO operations). Defaults to 4.
        """
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
    """Manipulator evaluating one term group in a model"""

    def __init__(
        self,
        name: str,
        model: mc.Model,
        termGroup: int,
        resultVar: Variable,
        priority: int = 4,
    ) -> None:
        """Manipulator evaluating one term group in a model

        Args:
            name (str): Name of the manipulator
            model (mc.Model): Model containing the evaluated group
            termGroup (int): Term group index (for general groups add number of implicit groups to the general group index)
            resultVar (Variable): Variable to store the evaluation result in
            priority (int, optional): Manipulator priority (0 called at every internal integrator iteration, 1 called at every integrator substep, 2 called between integration steps in a single global step, 3 called after each global integrator step, 4 called only before IO operations). Defaults to 4.
        """
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
            else "\\text{" + self.__resultVar__.name.replace("_", r"\_") + "}"
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
    """Manipulator evaluating terms by model+term names"""

    def __init__(
        self,
        name: str,
        modelTermTags: List[Tuple[str, str]],
        resultVar: Variable,
        accumulate=False,
        update=False,
        priority: int = 4,
    ) -> None:
        """Manipulator evaluating terms by model+term names

        Args:
            name (str): Name of the manipulator
            modelTermTags (List[Tuple[str, str]]): List of model,term name tuples representing the model+term pairs to be evaluated by this manipulator
            resultVar (Variable): Variable to store the evaluation result in
            accumulate (bool, optional): If true will accumulate the values into the result variable instead of overwriting. Defaults to False.
            update (bool, optional): If true will independently request updates for the evaluated terms/models. Defaults to False.
            priority (int, optional): Manipulator priority (0 called at every internal integrator iteration, 1 called at every integrator substep, 2 called between integration steps in a single global step, 3 called after each global integrator step, 4 called only before IO operations). Defaults to 4.
        """
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
            else "\\text{" + self.__resultVar__.name.replace("_", r"\_") + "}"
        )
        accu = "_{accu}" if self.__accumulate__ else ""
        return (
            resultVarName
            + " = \\text{Eval}"
            + accu
            + "\\left(\\text{"
            + ",".join(
                model.replace("_", r"\_") + "-" + term.replace("_", r"\_")
                for model, term in self.__modelTermTags__
            )
            + "}\\right)"
        )


class MBDataExtractor(Manipulator):
    """Manipulator extracting a variable from the modelbound data within a given model"""

    def __init__(
        self,
        name: str,
        model: mc.Model,
        mbVar: Variable,
        resultVar: Optional[Variable] = None,
        priority=4,
    ):
        """Manipulator extracting a variable from the modelbound data within a given model

        Args:
            name (str): Name of the manipulator
            model (mc.Model): Model containing the modelbound data
            mbVar (Variable): Modelbound data variable to extract
            resultVar (Optional[Variable], optional): Variable to put the extracted mbVar result to. Defaults to None, using the mbVar itself, assuming it is also in the global VariableContainer.
            priority (int, optional): Manipulator priority (0 called at every internal integrator iteration, 1 called at every integrator substep, 2 called between integration steps in a single global step, 3 called after each global integrator step, 4 called only before IO operations). Defaults to 4.
        """
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
            else "\\text{" + self.__mbVar__.name.replace("_", r"\_") + "}"
        )

        return (
            resultVarName
            + " = \\text{MBExtract}\\left("
            + self.__model__.latexName
            + "\\right)"
        )


class ManipulatorCollection:
    """Manipulator container object providing accessors methods"""

    def __init__(self: Self):
        """Manipulator container object providing accessors methods"""
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
        """Add manipulators to the container"""
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
                                manip.name.replace("_", r"\_")
                                + f": \\newline ${manip.latex(**kwargs)}$"
                            )
                        )


class RMKContext:
    """Central object in ReMKiT1D Python package - centralises the construction of ReMKiT1D simulations"""

    def __init__(self) -> None:
        """Central object in ReMKiT1D Python package - centralises the construction of ReMKiT1D simulations"""
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
        """Density normalisation in m^{-3}"""
        return self.__normDens__

    @normDensity.setter
    def normDensity(self, norm: float):
        self.__normDens__ = norm

    @property
    def normTemperature(self):
        """Temperature normalisation in eV"""
        return self.__normTemp__

    @normTemperature.setter
    def normTemperature(self, norm: float):
        self.__normTemp__ = norm

    @property
    def normZ(self):
        """Reference ion charge"""
        return self.__normZ__

    @normZ.setter
    def normZ(self, norm: float):
        self.__normZ__ = norm

    @property
    def norms(self):
        """Normalisation dictionary containing all basic and default derived normalisation values - requires that the textbook component of the context is set is set"""
        return skn.calculateNorms(
            self.__normTemp__,
            self.__normDens__,
            self.__normZ__,
            self.textbook.removeLogLeiDiscontinuity,
        )

    @property
    def grid(self):
        """Grid component of the context containing spatial and velocity space data"""
        return self.__gridObj__

    @grid.setter
    def grid(self, grid: Grid):
        self.__gridObj__ = grid

    @property
    def textbook(self):
        """Textbook component of the context containing the various derivations registered in the context"""
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
        """Species container component of the context - containing Species information used in the ReMKiT1D simulation"""
        return self.__species__

    @species.setter
    def species(self, sp: dv.SpeciesContainer):
        self.__species__ = sp

    @property
    def variables(self):
        """Variable container component of the context - contains all of the globally defined variables in the ReMKiT1D simulation"""
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
        """MPI options component of the contextd"""
        return self.__mpiContext__

    @mpiContext.setter
    def mpiContext(self, context: MPIContext):
        self.__mpiContext__ = context

    @property
    def optionsPETSc(self):
        """PETSc library options used by this context"""
        return self.__optionsPETSc__

    @property
    def models(self):
        """Model container component of the context"""
        if self.__models__ is None:
            self.__models__ = mc.ModelCollection()
        return self.__models__

    @models.setter
    def models(self, models: mc.ModelCollection):
        self.__models__ = models

    @property
    def integrationScheme(self):
        """Time integration scheme used by this context"""
        return self.__integrationScheme__

    @integrationScheme.setter
    def integrationScheme(self, scheme: it.IntegrationScheme):
        self.__integrationScheme__ = scheme

    @property
    def IOContext(self):
        """IO option container component of this simulation context"""
        return self.__IOContext__

    @IOContext.setter
    def IOContext(self, context: IOContext):
        self.__IOContext__ = context

    @property
    def manipulators(self):
        """Manipulator container component of this simulation context"""
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
        """Perform all consistency checks within the components"""
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
                implicitGroups, configFile["MPI"]["commData"], self.models
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
        """Generate a config file based on the current state of the context"""
        try:
            os.remove(self.IOContext.jsonFilepath)
        except FileNotFoundError:
            pass

        io.writeDictToJSON(self.dict(), filepath=self.IOContext.jsonFilepath)

    def generatePDF(
        self, latexFilename="ReMKiT1D", latexRemap: Dict[str, str] = {}, cleanTex=True
    ):
        """Generate a LaTeX pdf summary of the simulation context

        Args:
            latexFilename (str, optional): Name of the pdf file without the extension. Defaults to "ReMKiT1D".
            latexRemap (Dict[str, str], optional): Dictionary remapping variable names (for example "v":"\\vec{v}"). Defaults to {}.
            cleanTex (bool, optional): Remove all generated tex files. Defaults to True.
        """

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
            doc, implicitGroups, latexRemap=latexRemap, models=self.models
        )

        doc.generate_pdf(latexFilename.replace(" ", "_"), clean_tex=cleanTex)

    def loadSimulation(
        self, onlySteps: Optional[List[int]] = None
    ) -> VariableContainer:
        """Load the simulation output from the HDF5 directory set in the IOContext component

        Args:
            onlySteps (Optional[List[int]], optional): Output file indices to load. Defaults to None - loading all detected output files.

        Returns:
            VariableContainer: Variable container containing the loaded data
        """

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
        """Add diagnostic variables for all terms evolving given variables. **NOTE**: Does not detect terms generated by TermGenerators - to include these use GroupEvaluators."""
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
