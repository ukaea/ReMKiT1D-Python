import numpy as np
from .grid import Grid
from typing import Union, List, Dict, cast, Tuple
from .variable_container import VariableContainer
from . import simple_containers as sc
from . import init_templates as it
from . import IO_support as io
import warnings
import os


class RKWrapper:
    """Wrapper allowing for convenience when building ReMKiT1D config.json file"""

    def __init__(self) -> None:
        self.__normalization__ = {
            "density": 1.0e19,
            "eVTemperature": 10.0,
            "referenceIonZ": 1.0,
        }

        self.__gridObj__: Union[None, Grid] = None

        self.__customDerivs__: Dict[str, object] = {"tags": []}

        self.__standardTextook__ = {
            "temperatureDerivSpeciesIDs": [],  # IDs for those species whose temperature derivation rules should be included
            "electronPolytropicCoeff": 1.0,  # Electron polytropic coefficient
            "ionPolytropicCoeff": 1.0,  # Ion polytropic coefficient
            "electronSheathGammaIonSpeciesID": -1,  # ID of ions whose mass is used to calculate the electron sheath heat transmission coefficient (Defaults to -1)
        }

        self.__species__: Dict[str, sc.Species] = {}
        self.__speciesData__: Dict[str, object] = {"names": []}

        self.__varCont__: Union[None, VariableContainer] = None

        mpiCommData: Dict[str, object] = {
            "varsToBroadcast": [],
            "haloExchangeVars": [],
            "scalarVarsToBroadcast": [],
            "scalarBroadcastRoots": [],
        }

        self.__mpiData__: Dict[str, object] = {
            "numProcsX": 1,
            "numProcsH": 1,
            "xHaloWidth": 1,
            "commData": mpiCommData,
        }

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

        self.__optionsHDF5__: Dict[str, object] = {
            "outputVars": [],  # Variables to output
            "filepath": "./RMKOutput/",  # Location of IO HDF5 files
        }

        self.__jsonFilepath__ = "./config.json"

        self.__modelData__: Dict[str, object] = {"tags": []}
        self.__manipulatorData__: Dict[str, object] = {"tags": []}
        self.__integratorData__: Dict[str, object] = {
            "stepTags": [],
            "integratorTags": [],
            "timestepController": {"active": False},
            "initialTimestep": 0.1,
            "numImplicitGroups": 1,
            "numGeneralGroups": 1,
        }
        self.__timeloopData__: Dict[str, object] = {
            "mode": "fixedNumSteps",  # Main timeloop mode.
            # Options: "fixedNumSteps" - advances for a fixed number of (global) integration steps, with no regard to elapsed time
            #          "normalizedTimeTarget" - advances until a set amount of time has passed, expressed in normalized time units
            #          "realTimeTarget" - advances until a set amount of time has passed in seconds
            "numTimesteps": 1,  # Number of timesteps to advance for, ignored unless mode = fixedNumSteps
            "timeValueTarget": 1.0,  # Time value for the two non-fixed step number modes - the mode governs how this number is interpreted
            "outputMode": "fixedNumSteps",  # Output options: "fixedNumSteps" - output once a fixed number of steps has passed
            #            "minimumSaveInterval" - outputs once a set amount of time has passed (in normalized time units)
            "fixedSaveInterval": 1,  # Save frequency if the output mode is fixedNumSteps, ignored otherwise
            "minimumSaveInterval": 0.1,  # Interval corresponding to the output mode of the same name
            "restart": {  # Options governing the saving and loading of restart data.
                "save": False,  # If true, restart data will be saved
                "load": False,  # If true, restart data will be loaded at the start of the loop. Will throw error if loadInitValsFromHDF5 is also true
                "frequency": 1,  # Restart save frequency - saving every n steps
            },
            "loadInitValsFromHDF5": False,  # True if variables should be loaded from a complete HDF5 file based on the input vars list in the HDF5 options.
            "initValFilename": "ReMKiT1DVarInput",  # Name of the input hdf5 file
        }

    @property
    def normalization(self):
        return self.__normalization__

    @normalization.setter
    def normalization(self, norm: dict):
        self.__normalization__.update(norm)

    @property
    def grid(self):
        return self.__gridObj__

    @grid.setter
    def grid(self, grid: Grid):
        self.__gridObj__ = grid

    @property
    def customDerivs(self):
        return self.__customDerivs__

    @property
    def standardTextbook(self):
        return self.__standardTextook__

    @property
    def speciesData(self):
        return self.__speciesData__

    @property
    def varCont(self):
        return self.__varCont__

    @varCont.setter
    def varCont(self, vc: VariableContainer):
        self.__varCont__ = vc

    @property
    def mpiData(self):
        return self.__mpiData__

    @property
    def optionsPETSc(self):
        return self.__optionsPETSc__

    @property
    def jsonFilepath(self):
        return self.__jsonFilepath__

    @jsonFilepath.setter
    def jsonFilepath(self, jsonpath: str):
        self.__jsonFilepath__ = jsonpath

    @property
    def modelData(self):
        return self.__modelData__

    @property
    def manipulatorData(self):
        return self.__manipulatorData__

    @property
    def integratorData(self):
        return self.__integratorData__

    @property
    def timeloopData(self):
        return self.__timeloopData__

    @property
    def hdf5Filepath(self):
        return self.__optionsHDF5__["filepath"]

    def modelTags(self):
        return self.__modelData__["tags"]

    def setNormDensity(self, dens: float):
        self.__normalization__["density"] = dens

    def setNormTemperature(self, temp: float):
        self.__normalization__["eVTemperature"] = temp

    def setNormRefZ(self, refZ: float):
        self.__normalization__["referenceIonZ"] = refZ

    def addVarToComm(
        self, name: str, isDistribution=False, isScalar=False, hostScalarProcess=0
    ) -> None:
        """Add variable to global MPI communication pattern

        Args:
            name (str): Name of the variable
            isDistribution (bool, optional): True if the variable is a distribution. Defaults to False.
            isScalar (bool, optional): True if the variable is a scalar. Defaults to False.
            hostScalarProcess (int, optional): The host processor in charge of calculating and broadcasting the scalar variable. Defaults to 0.
        """

        if isDistribution:
            cast(
                List[str],
                cast(Dict[str, object], self.__mpiData__["commData"])[
                    "varsToBroadcast"
                ],
            ).append(name)
            cast(
                List[str],
                cast(Dict[str, object], self.__mpiData__["commData"])[
                    "haloExchangeVars"
                ],
            ).append(name)
        elif isScalar:
            cast(
                List[str],
                cast(Dict[str, object], self.__mpiData__["commData"])[
                    "scalarVarsToBroadcast"
                ],
            ).append(name)
            cast(
                List[str],
                cast(Dict[str, object], self.__mpiData__["commData"])[
                    "scalarBroadcastRoots"
                ],
            ).append(hostScalarProcess)
        else:
            cast(
                List[str],
                cast(Dict[str, object], self.__mpiData__["commData"])[
                    "haloExchangeVars"
                ],
            ).append(name)
            if cast(int, self.__mpiData__["numProcsH"]) > 1:
                cast(
                    List[str],
                    cast(Dict[str, object], self.__mpiData__["commData"])[
                        "varsToBroadcast"
                    ],
                ).append(name)

    def addVar(
        self,
        name: str,
        data: Union[np.ndarray, None] = None,
        isDerived=False,
        isDistribution=False,
        units="normalized units",
        isStationary=False,
        isScalar=False,
        isOnDualGrid=False,
        priority=0,
        derivationRule: Union[None, dict] = None,
        outputVar=True,
        isCommunicated=False,
        hostScalarProcess=0,
    ) -> None:
        """Add variable to the wrapper variable container

        Args:
            name (str): Variable names
            data (Union[numpy.ndarray,None], optional): Optional numpy array representing variable data. Defaults to None, which initializes data to 0.
            isDerived (bool, optional): True if the variable is treated as derived by ReMKiT1D. Defaults to False.
            isDistribution (bool, optional): True for distribution-like variables. Defaults to False.
            units (str, optional): Variable units. Defaults to 'normalized units'.
            isStationary (bool, optional): True if the variable is stationary (d/dt = 0). Defaults to False.
            isScalar (bool, optional): True if the variable is a scalar. Defaults to False.
            isOnDualGrid (bool, optional): True if the variable is defined on dual grid. Defaults to False.
            priority (int, optional): Variable priority used in things like derivation call in integrators. Defaults to 0 (highest priority).
            derivationRule (Union[None,dict], optional) Optional derivation rule for derived variables. Defaults to None.
            outputVar (bool, optional): True if the variable should be added to the code output. Defaults to True.
            isCommunicated (bool, optional): True if the variable should be communicated. Defaults to False.
            hostScalarProcess (int, optional): Host process in case of a communicated scalar variable. Defaults to 0.
        """

        if self.__varCont__ is None:
            assert (
                self.__gridObj__ is not None
            ), "Attempted to add variable to RKWrapper variable container without first setting grid"
            self.__varCont__ = VariableContainer(self.__gridObj__)

        self.__varCont__.setVariable(
            name,
            data,
            isDerived,
            isDistribution,
            units,
            isStationary,
            isScalar,
            isOnDualGrid,
            priority,
            derivationRule,
        )

        if outputVar:
            cast(List[str], self.__optionsHDF5__["outputVars"]).append(name)

        if isCommunicated:
            self.addVarToComm(name, isDistribution, isScalar, hostScalarProcess)

    def addVarAndDual(
        self,
        varName: str,
        data: Union[np.ndarray, None] = None,
        isDerived=False,
        derivationRule: Union[None, dict] = None,
        isDistribution=False,
        isStationary=False,
        primaryOnDualGrid=False,
        units="normalized units",
        priority=0,
        dualSuffix="_dual",
        outputVar=True,
        isCommunicated=False,
        communicateSecondary=True,
    ) -> None:
        """Add variable and its dual

        Args:
            varName (str): Name of variable on regular grid
            data (Union[numpy.ndarray,None], optional): Optional numpy array representing variable data. Defaults to None, which initializes data to 0.
            isDerived (bool, optional): True if both the primary variable is derived. Defaults to False.
            derivationRule (Union[None,dict], optional): Derivation rule for primary derived variable. Defaults to None.
            isDistribution (bool, optional): True if variable is a distribution. Defaults to False.
            isStationary (bool, optional): True if primary variable is stationary. Defaults to False.
            primaryOnDualGrid (bool, optional): True if the primary variable is on the dual grid. Defaults to False.
            units (str, optional): Units for both primary and secondary. Defaults to 'normalized units'.
            priority (int, optional): Variable priority for both primary and secondary. Defaults to 0 (highest priority).
            dualSuffix (str, optional): Suffix for the variable on the dual grid. Defaults to "_dual".
            outputVar (bool, optional): Set to true if both variable and dual should be added to code output. Defaults to True.
            isCommunicated (bool, optional): Set to true if primary variable should be communicated. Defaults to False.
            communicateSecondary (bool, optional): Set to true if secondary variable should be communicated (only if primary is communicated). Defaults to True.
        """

        if self.__varCont__ is None:
            assert (
                self.__gridObj__ is not None
            ), "Attempted to add variable and dual to RKWrapper variable container without first setting grid"
            self.__varCont__ = VariableContainer(self.__gridObj__)

        it.addVarAndDual(
            self.__varCont__,
            varName,
            data,
            isDerived,
            derivationRule,
            isDistribution,
            isStationary,
            primaryOnDualGrid,
            units,
            priority,
            dualSuffix,
        )

        if outputVar:
            cast(List[str], self.__optionsHDF5__["outputVars"]).append(varName)
            cast(List[str], self.__optionsHDF5__["outputVars"]).append(
                varName + dualSuffix
            )

        primaryVar = varName
        secondaryVar = varName + dualSuffix
        if primaryOnDualGrid:
            primaryVar = varName + dualSuffix
            secondaryVar = varName

        if isCommunicated:
            self.addVarToComm(primaryVar, isDistribution)
            if communicateSecondary:
                self.addVarToComm(secondaryVar, isDistribution)

    def setMPIData(self, numProcsX: int, numProcsH=1, haloWidth=1) -> None:
        """Set general MPI data

        Args:
            numProcsX (int): Number of processors in the x direction
            numProcsH (int): Number of processors in the harmonic direction. Defaults to 1.
            haloWidth (int, optional): Halo width in the x direction in cells. Defaults to 1.
        """
        self.__mpiData__["numProcsX"] = numProcsX
        self.__mpiData__["numProcsH"] = numProcsH
        self.__mpiData__["xHaloWidth"] = haloWidth

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
            objGroups (int, optional): Number of PETSc object groups to create. Defaults to 1.
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

    def setHDF5Path(self, path: str) -> None:
        """Set HDF5 filepath

        Args:
            path (str): Filepath used for HDF5 input and output files
        """
        self.__optionsHDF5__["filepath"] = path

    def setStandardTextbookOptions(
        self,
        tempDerivSpeciesIDs: List[int] = [],
        ePolyCoeff=1.0,
        ionPolyCoeff=1.0,
        electronSheathGammaIonSpeciesID=-1,
    ) -> None:
        """Set options used by the standard textbook object containing common derivations

        Args:
            tempDerivSpeciesIDs (List[int], optional): List of species IDs for which a temperature derivation object is constructed. Defaults to [].
            ePolyCoeff (float, optional): Electron polytropic coefficient for sound speed calculations. Defaults to 1.0.
            ionPolyCoeff (float, optional): Ion polytropic coefficient for sound speed calculations. Defaults to 1.0.
            electronSheathGammaIonSpeciesID (int, optional): Ion species ID used for the default electron sheath heat transmission coefficient calculation. Defaults to -1.
        """

        self.__standardTextook__["temperatureDerivSpeciesIDs"] = tempDerivSpeciesIDs
        self.__standardTextook__["electronPolytropicCoeff"] = ePolyCoeff
        self.__standardTextook__["ionPolytropicCoeff"] = ionPolyCoeff
        self.__standardTextook__[
            "electronSheathGammaIonSpeciesID"
        ] = electronSheathGammaIonSpeciesID

    def addCustomDerivation(self, derivName: str, derivOptions: dict) -> None:
        """Add a custom derivation object

        Args:
            derivName (str): Name of the derivation
            derivOptions (dict): Option dictionary containing properties of the derivation
        """

        assert derivName not in cast(List[str], self.__customDerivs__["tags"]), (
            "Duplicate custom derivation tag " + derivName
        )
        cast(List[str], self.__customDerivs__["tags"]).append(derivName)
        self.__customDerivs__[derivName] = derivOptions

    def addDerivationCollection(self, derivCollection: dict) -> None:
        """Add a collection of custom derivations

        Args:
            derivCollection (dict): Dictionary with derivation tags as keys and derivation options as velues
        """

        for key in derivCollection.keys():
            if not key in cast(List[str], self.__customDerivs__["tags"]):
                cast(List[str], self.__customDerivs__["tags"]).append(key)
            else:
                warnings.warn("Custom derivation " + key + " overwritten in wrapper")

            self.__customDerivs__[key] = derivCollection[key]

    def addSpecies(
        self,
        name: str,
        speciesID: int,
        atomicA: float = 1.0,
        charge: float = 0.0,
        associatedVars: List[str] = [],
    ) -> None:
        """Add a species object to the wrapper

        Args:
            name (str): Name of the species
            speciesID (int): Integer ID of species (0 reserved for electrons)
            atomicA (float, optional): Atomic mass in amus. Defaults to 1.0.
            charge (float, optional): Charge in units of e. Defaults to 0.0.
            associatedVars (List[str], optional): Variables associated with this species. Defaults to [].
        """

        assert atomicA > 0, "Zero mass species are not allowed"

        cast(List[str], self.__speciesData__["names"]).append(name)
        self.__species__[name] = sc.Species(
            name, speciesID, atomicA, charge, associatedVars
        )

        self.__speciesData__[name] = self.__species__[name].dict()

    def getSpecies(self, name: str) -> sc.Species:
        """Return species object with given name

        Args:
            name (str): Name of the species to be returned

        Returns:
            sc.Species: Species object containing species data for the required species
        """
        return self.__species__[name]

    def addModel(self, properties: dict) -> None:
        """Add model to wrapper

        Args:
            properties (dict): Model properies dictionary. Should have a single key pointing to all properties, with the key being the model tag.
        """
        cast(List[str], self.__modelData__["tags"]).append(list(properties.keys())[0])
        self.__modelData__.update(properties)

    def addManipulator(self, tag: str, properties: dict) -> None:
        """Add a manipulator object to wrapper

        Args:
            tag (str): Manipulator tag
            properties (dict): Properties dictionary containing manipulator options
        """
        cast(List[str], self.__manipulatorData__["tags"]).append(tag)
        self.__manipulatorData__[tag] = properties

    def setIntegratorGlobalData(
        self, numImplicitGroups=1, numGeneralGroups=1, initialTimestep=0.1
    ) -> None:
        """Set global data for the time integration routines

        Args:
            numImplicitGroups (int, optional): Maximum number of allowed implicit groups. Defaults to 1.
            numGeneralGroups (int, optional): Maximum number of allowed general groups. Defaults to 1.
            initialTimestep (float, optional): Default/initial timestep value in normalized units. Defaults to 0.1.
        """
        self.__integratorData__["numImplicitGroups"] = numImplicitGroups
        self.__integratorData__["numGeneralGroups"] = numGeneralGroups
        self.__integratorData__["initialTimestep"] = initialTimestep

    def addIntegrator(self, tag: str, properties: dict) -> None:
        """Add an integrator object to the wrapper

        Args:
            tag (str): Tag of the integrator
            properties (dict): Properties dictionary determining the options used by the integrator
        """
        cast(List[str], self.__integratorData__["integratorTags"]).append(tag)
        self.__integratorData__[tag] = properties

    def addIntegrationStep(
        self, tag: str, properties: dict, mpiComm: Union[None, dict] = None
    ) -> None:
        """Add an integration step to the wrapper. Steps are executed within a single timestep in order of their addition to the wrapper.

        Args:
            tag (str): Tag of the step
            properties (dict): Property dictionary of the step
            mpiComm (Union[None,dict], optional): Optional custom communication data used by the step. Defaults to None, resulting in the use of the global communication options.
        """

        cast(List[str], self.__integratorData__["stepTags"]).append(tag)
        self.__integratorData__[tag] = properties
        cast(Dict[str, object], self.__integratorData__[tag])["commData"] = (
            self.__mpiData__["commData"] if mpiComm is None else mpiComm
        )

    def setTimestepController(self, properties: dict) -> None:
        """Set timestep controller options

        Args:
            properties (dict): Dictionary containing timestep controller options
        """
        cast(Dict[str, object], self.__integratorData__["timestepController"])[
            "active"
        ] = True
        cast(Dict[str, object], self.__integratorData__["timestepController"]).update(
            properties
        )

    def setFixedNumTimesteps(self, numTimesteps: int) -> None:
        """Sets timeloop mode to fixed number of timesteps and changes the number of timesteps

        Args:
            numTimesteps (int): Number of timesteps to run the code fore
        """

        self.__timeloopData__["mode"] = "fixedNumSteps"
        self.__timeloopData__["numTimesteps"] = numTimesteps

    def setTimeTargetTimestepping(
        self, timeTarget: float, realTimeTarget=False
    ) -> None:
        """Set timeloop mode to one of the two time target modes and set the time target value

        Args:
            timeTarget (float): Value of the time variable at which to stop the simulations
            realTimeTarget (bool, optional): Set to true if the timeTarget value is given seconds. Defaults to False.
        """

        self.__timeloopData__["mode"] = "normalizedTimeTarget"
        if realTimeTarget:
            self.__timeloopData__["mode"] = "realTimeTarget"
        self.__timeloopData__["timeValueTarget"] = timeTarget

    def setFixedStepOutput(self, fixedStep: int) -> None:
        """Set the timeloop output mode to fixed step and set the output interval

        Args:
            fixedStep (int): Output interval in timesteps
        """

        self.__timeloopData__["outputMode"] = "fixedNumSteps"
        self.__timeloopData__["fixedSaveInterval"] = fixedStep

    def setMinimumIntervalOutput(self, minimumInterval: float) -> None:
        """Set the timeloop output mode to minimum elapsed normalized time and set the minimum interval

        Args:
            minimumInterval (float): Minimum interval required to elapse before a save is triggered. Expects normalized time value.
        """

        self.__timeloopData__["outputMode"] = "minimumSaveInterval"
        self.__timeloopData__["minimumSaveInterval"] = minimumInterval

    def setRestartOptions(self, save=False, load=False, frequency=1) -> None:
        """Set restart options in timeloop object

        Args:
            save (bool, optional): Set to true if the code should save restart data. Defaults to False.
            load (bool, optional): Set to true if the code should initialize from restart data. Defaults to False.
            frequency (int, optional): Frequency at which restart data is saved in timesteps. Defaults to 1.
        """

        cast(Dict[str, object], self.__timeloopData__["restart"])["save"] = save
        cast(Dict[str, object], self.__timeloopData__["restart"])["load"] = load
        cast(Dict[str, object], self.__timeloopData__["restart"])[
            "frequency"
        ] = frequency

    def setHDF5FileInitialData(
        self,
        inputVars: List[str] = [],
        useHDF5Input=True,
        filename="ReMKiT1DVarInput",
    ) -> None:
        """Set HDF5 input options

        Args:
            inputVars (List[str], optional): Names of input variables to take from input file. Defaults to [].
            useHDF5Input (bool, optional): True if initial data in variable container should be ignored and HDF5 data used instead. Defaults to True.
            filename (str, optional): Name of the input file without .h5 extension. Defaults to "ReMKiT1DVarInput".
        """

        self.__optionsHDF5__["inputVars"] = inputVars
        self.__timeloopData__["loadInitValsFromHDF5"] = useHDF5Input
        self.__timeloopData__["initValFilename"] = filename

    def getTermsThatEvolveVar(self, var: str) -> List[Tuple[str, str]]:
        """Return all model,term pairs where a given variable is the evolved variable

        Args:
            var (str): Name of the variable

        Returns:
            List[Tuple[str,str]]: List of model,term pairs
        """

        terms = []
        for model in self.modelTags():
            for term in self.modelData[model]["termTags"]:
                if self.modelData[model][term]["evolvedVar"] == var:
                    terms.append((model, term))

        return terms

    def varList(self) -> List[str]:
        """Return list of all variables in wrapper's variable container

        Returns:
            List[str]: List of variables
        """

        return list(self.varCont.dataset.keys())

    def dict(self) -> dict:
        """Convert wrapper into config.json compatible dictionary

        Returns:
            dict: Immediately writable config.json dictionary
        """

        configFile = {
            "normalization": self.normalization,
            "customDerivations": self.customDerivs,
            "standardTextbook": self.standardTextbook,
            "species": self.speciesData,
            "MPI": self.mpiData,
            "PETSc": self.optionsPETSc,
            "HDF5": self.__optionsHDF5__,
            "models": self.modelData,
            "manipulators": self.manipulatorData,
            "integrator": self.integratorData,
            "timeloop": self.timeloopData,
        }

        configFile.update(cast(Grid, self.__gridObj__).dict())
        configFile.update(cast(VariableContainer, self.__varCont__).dict())

        return configFile

    def varsInOutput(self) -> List[str]:
        """Return list of variables in HDF5 output

        Returns:
            List[str]: list of outputted vars
        """

        return cast(List[str], self.__optionsHDF5__["outputVars"])

    def writeConfigFile(self) -> None:
        try:
            os.remove(self.jsonFilepath)
        except FileNotFoundError:
            pass

        io.writeDictToJSON(self.dict(), filepath=self.jsonFilepath)

    def addTermDiagnosisForVars(self, names: List[str]) -> None:
        """Add all terms that evolve given variables as diagnostic variables

        Args:
            name (List[str]): Names of variables whose evolution terms should be added
        """

        for name in names:
            terms = self.getTermsThatEvolveVar(name)

            for pair in terms:
                model, term = pair
                self.addVar(model + term, isDerived=True)
                self.addManipulator(
                    model + term, sc.termEvaluatorManipulator([pair], model + term)
                )
