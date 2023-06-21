from . import simple_containers as sc
from typing import Union, List, Tuple, cast
import numpy as np
from .grid import Grid
from .rk_wrapper import RKWrapper


def collocatedAdvection(
    modelTag: str,
    advectedVar: str,
    advectionSpeed: str,
    lowerBoundVar: Union[str, None] = None,
    leftOutflow=False,
    rightOutflow=False,
    centralDiff=False,
) -> sc.CustomModel:
    """Create a collocated advection model

    Args:
        modelTag (str): Model tag
        advectedVar (str): Name of advected variable
        advectionSpeed (str): Name of advection speed variable
        lowerBoundVar (Union[str,None], optional): Name of outflow lower bound variable, can be None, which defaults to a fixed
                                                   lower bound of zero. Defaults to None.
        leftOutflow (bool, optional): If true left boundary is treated as outflow, otherwise it is reflective
                                      or periodic depending on the grid. Defaults to False.
        rightOutflow (bool, optional): If true right boundary is treated as outflow, otherwise it is reflective
                                      or periodic depending on the grid. Defaults to False.
        centralDiff (bool, optional): If true uses central difference stencil instead of upwinding. Defaults to False.

    Returns:
        sc.CustomModel: CustomModel ready for insertion into JSON config file
    """

    # Initializing model
    newModel = sc.CustomModel(modelTag=modelTag)

    # Setting normalization constant calculation (here -u_0*t_0/x_0 - should be -1 with default normalization)
    normConst = sc.CustomNormConst(
        multConst=-1.0,
        normNames=["speed", "time", "length"],
        normPowers=[1.0, 1.0, -1.0],
    )

    evolvedVar = advectedVar
    divStencil = sc.upwindedDiv(advectionSpeed)

    if centralDiff:
        divStencil = sc.centralDiffStencilDiv(advectionSpeed)

    divFluxTerm = sc.GeneralMatrixTerm(
        evolvedVar, customNormConst=normConst, stencilData=divStencil
    )

    newModel.addTerm("divFlux", divFluxTerm)

    if leftOutflow:
        # Add left boundary term

        boundaryStencilLeft = sc.boundaryStencilDiv(
            advectionSpeed, lowerBoundVar=lowerBoundVar, isLeft=True
        )

        leftBCTerm = sc.GeneralMatrixTerm(
            evolvedVar, customNormConst=normConst, stencilData=boundaryStencilLeft
        )

        newModel.addTerm("leftBC", leftBCTerm)

    if rightOutflow:
        boundaryStencilRight = sc.boundaryStencilDiv(
            advectionSpeed, lowerBoundVar=lowerBoundVar
        )

        # Add Right boundary term

        rightBCTerm = sc.GeneralMatrixTerm(
            evolvedVar, customNormConst=normConst, stencilData=boundaryStencilRight
        )

        newModel.addTerm("rightBC", rightBCTerm)

    return newModel


def collocatedPressureGrad(
    modelTag: str,
    fluxVar: str,
    densityVar: str,
    temperatureVar: str,
    speciesMass: float,
    addExtrapolatedBCs=True,
) -> sc.CustomModel:
    """Create a collocated pressure gradient model for the momentum equation

    Args:
        modelTag (str): Model tag
        fluxVar (str): Evolved flux variable name
        densityVar (str): Species density name factoring into pressure
        temperatureVar (str): Species temperature name factoring into pressure
        speciesMass (float): Species mass in kg
        addExtrapolatedBCs (bool, optional): If true will add extrapolated values as left and right
                                            boundary condition (set fo False for periodic grids). Defaults to True

    Returns:
        sc.CustomModel: CustomModel object ready for insertion into JSON config file
    """

    # Initializing model
    newModel = sc.CustomModel(modelTag=modelTag)

    elCharge = 1.60218e-19
    # Setting normalization constant calculation
    normConst = sc.CustomNormConst(
        multConst=-elCharge / speciesMass,
        normNames=["eVTemperature", "time", "length", "speed"],
        normPowers=[1.0, 1.0, -1.0, -1.0],
    )

    evolvedVar = fluxVar
    implicitVar = densityVar

    # Central differenced gradient term

    gradStencil = sc.centralDiffStencilGrad()

    # Required variable data for electron pressure
    vData = sc.VarData(reqColVars=[temperatureVar])

    gradTerm = sc.GeneralMatrixTerm(
        evolvedVar,
        implicitVar=implicitVar,
        customNormConst=normConst,
        stencilData=gradStencil,
        varData=vData,
    )

    newModel.addTerm("bulkGrad", gradTerm)

    if addExtrapolatedBCs:
        # Left boundary condition with extrapolation

        boundaryStencilLeft_grad = sc.boundaryStencilGrad(isLeft=True)

        # Right boundary condition with extrapolation

        boundaryStencilRight_grad = sc.boundaryStencilGrad()

        # Add left boundary term

        leftBCTerm = sc.GeneralMatrixTerm(
            evolvedVar,
            implicitVar=implicitVar,
            customNormConst=normConst,
            stencilData=boundaryStencilLeft_grad,
            varData=vData,
        )

        newModel.addTerm("leftBC", leftBCTerm)

        # Add Right boundary term

        rightBCTerm = sc.GeneralMatrixTerm(
            evolvedVar,
            implicitVar=implicitVar,
            customNormConst=normConst,
            stencilData=boundaryStencilRight_grad,
            varData=vData,
        )

        newModel.addTerm("rightBC", rightBCTerm)

    return newModel


def simpleSourceTerm(
    evolvedVar: str, sourceProfile: np.ndarray, timeSignal=sc.TimeSignalData()
) -> sc.GeneralMatrixTerm:
    """Simple implicit source term with given source profile

    Args:
        evolvedVar (str): Name of evolved variable
        sourceProfile (np.ndarray): Spatial source profile
        timeSignal (sc.TimeSignalData): Optional time signal component of source. Defaults to constant signal.

    Returns:
        sc.GeneralMatrixTerm: Term ready for adding to a model
    """

    vData = sc.VarData(reqRowVars=[evolvedVar], reqRowPowers=[-1.0])

    sourceTerm = sc.GeneralMatrixTerm(
        evolvedVar,
        spatialProfile=sourceProfile.tolist(),
        varData=vData,
        stencilData=sc.diagonalStencil(),
        timeSignalData=timeSignal,
    )

    return sourceTerm


def staggeredAdvection(
    modelTag: str,
    advectedVar: str,
    fluxVar: str,
    advectionSpeed: Union[str, None] = None,
    staggeredAdvectionSpeed: Union[str, None] = None,
    lowerBoundVar: Union[str, None] = None,
    leftOutflow=False,
    rightOutflow=False,
    staggeredAdvectedVar=False,
    vData=sc.VarData(),
) -> sc.CustomModel:
    """Create a staggered grid advection model

    Args:
        modelTag (str): Model tag
        fluxVar (str): Name of flux variable - always on dual grid
        advectionSpeed (Union[str,None], optional): Name of advection speed variable - on regular grid - only required if there is outflow. Defaults to None.
        staggeredAdvectionSpeed (Union[str,None], optional): Name of advection speed variable - on dual grid - required if staggeredAdvectedVar=True. Defaults to None.
        lowerBoundVar (Union[str,None], optional): Name of outflow lower bound variable, can be None, which defaults to a fixed lower bound of zero. Defaults to None.
        leftOutflow (bool, optional): If true left boundary is treated as outflow, otherwise it is reflective or periodic depending on the grid. Defaults to False.
        rightOutflow (bool, optional): If true right boundary is treated as outflow, otherwise it is reflective or periodic depending on the grid. Defaults to False.
        staggeredAdvectedVar (bool, optional): If true will construct the flux from the advectedVar and staggered advection speed. Defaults to False.
        vData (sc.VarData, optional): Optional VarData object. Defaults to empty object.
    Returns:
        sc.CustomModel: CustomModel ready for insertion into JSON config file
    """

    if staggeredAdvectedVar:
        assert (
            staggeredAdvectionSpeed is not None
        ), "In staggeredAdvection if advected variable is staggered the staggered advection speed must be specified"
    # Initializing model
    newModel = sc.CustomModel(modelTag=modelTag)

    # Setting normalization constant calculation (here -u_0*t_0/x_0 - should be -1 with default normalization)
    normConst = sc.CustomNormConst(
        multConst=-1.0,
        normNames=["speed", "time", "length"],
        normPowers=[1.0, 1.0, -1.0],
    )

    evolvedVar = advectedVar
    divStencil = sc.staggeredDivStencil()
    implicitVar = fluxVar

    if staggeredAdvectedVar:
        divStencil = sc.centralDiffStencilDiv(cast(str, staggeredAdvectionSpeed))
        implicitVar = evolvedVar

    divFluxTerm = sc.GeneralMatrixTerm(
        evolvedVar,
        implicitVar,
        customNormConst=normConst,
        stencilData=divStencil,
        varData=vData,
    )

    newModel.addTerm("divFlux", divFluxTerm)

    if leftOutflow:
        # Add left boundary term

        assert (
            advectionSpeed is not None
        ), "advectionSpeed must be passed to staggeredAdvection if there is outflow"
        boundaryStencilLeft = sc.boundaryStencilDiv(
            advectionSpeed, lowerBoundVar=lowerBoundVar, isLeft=True
        )

        leftBCTerm = sc.GeneralMatrixTerm(
            evolvedVar,
            customNormConst=normConst,
            stencilData=boundaryStencilLeft,
            varData=vData,
        )

        newModel.addTerm("leftBC", leftBCTerm)

    if rightOutflow:
        assert (
            advectionSpeed is not None
        ), "advectionSpeed must be passed to staggeredAdvection if there is outflow"

        boundaryStencilRight = sc.boundaryStencilDiv(
            advectionSpeed, lowerBoundVar=lowerBoundVar
        )

        # Add Right boundary term

        rightBCTerm = sc.GeneralMatrixTerm(
            evolvedVar,
            customNormConst=normConst,
            stencilData=boundaryStencilRight,
            varData=vData,
        )

        newModel.addTerm("rightBC", rightBCTerm)

    return newModel


def staggeredPressureGrad(
    modelTag: str,
    fluxVar: str,
    densityVar: str,
    temperatureVar: str,
    speciesMass: float,
) -> sc.CustomModel:
    """Create a staggered grid pressure gradient model for the momentum equation, assuming the flux is on the dual grid

    Args:
        modelTag (str): Model tag
        fluxVar (str): Evolved flux variable name
        densityVar (str): Species density name factoring into pressure
        temperatureVar (str): Species temperature name factoring into pressure
        speciesMass (float): Species mass in kg

    Returns:
        sc.CustomModel: CustomModel object ready for insertion into JSON config file
    """

    # Initializing model
    newModel = sc.CustomModel(modelTag=modelTag)

    elCharge = 1.60218e-19
    # Setting normalization constant calculation
    normConst = sc.CustomNormConst(
        multConst=-elCharge / speciesMass,
        normNames=["eVTemperature", "time", "length", "speed"],
        normPowers=[1.0, 1.0, -1.0, -1.0],
    )

    evolvedVar = fluxVar
    implicitVar = densityVar

    gradStencil = sc.staggeredGradStencil()

    # Required variable data for pressure
    vData = sc.VarData(reqColVars=[temperatureVar])

    gradTerm = sc.GeneralMatrixTerm(
        evolvedVar,
        implicitVar=implicitVar,
        customNormConst=normConst,
        stencilData=gradStencil,
        varData=vData,
    )

    newModel.addTerm("bulkGrad", gradTerm)

    return newModel


def ampereMaxwell(
    modelTag: str,
    eFieldName: str,
    speciesFluxes: List[str],
    species: List[sc.Species],
) -> sc.CustomModel:
    """Generate dE/dt = -j/epsilon0 terms by calculating currents based on species fluxes and charges

    Args:
        eFieldName (str): Name of evolved electric field variable
        speciesFluxes (List[str]): Names of species fluxes
        species (list[sc.Species]): Species objects for each species

    Returns:
        sc.CustomModel: CustomModel object ready for insertion into JSON config file
    """

    assert len(speciesFluxes) == len(
        species
    ), "speciesFluxes and species passed to ampereMaxwell model must be of same size"
    # Initializing model
    newModel = sc.CustomModel(modelTag=modelTag)

    elCharge = 1.60218e-19
    epsilon0 = 8.854188e-12  # vacuum permittivity
    evolvedVar = eFieldName

    for i, flux in enumerate(speciesFluxes):
        normConst = sc.CustomNormConst(
            multConst=-species[i].charge * elCharge / epsilon0,
            normNames=["density", "time", "speed", "EField"],
            normPowers=[1.0, 1.0, 1.0, -1.0],
        )

        newTerm = sc.GeneralMatrixTerm(
            evolvedVar,
            implicitVar=flux,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(),
        )

        newModel.addTerm("current" + flux, newTerm)

    return newModel


def lorentzForces(
    modelTag: str,
    eFieldName: str,
    speciesFluxes: List[str],
    speciesDensities: List[str],
    species: List[sc.Species],
) -> sc.CustomModel:
    """Generate Lorentz force terms for each species

    Args:
        eFieldName (str): Name of  electric field variable
        speciesFluxes (List[str]): Names of evolved species fluxes
        speciesDensities (List[str]): Names of species densities
        species (list[sc.Species]): Species objects for each species

    Returns:
        sc.CustomModel: CustomModel object ready for insertion into JSON config file
    """

    assert len(speciesFluxes) == len(
        species
    ), "speciesFluxes and species passed to lorentzForces model must be of same size"
    assert len(speciesDensities) == len(
        species
    ), "speciesDensities and species passed to lorentzForces model must be of same size"

    # Initializing model
    newModel = sc.CustomModel(modelTag=modelTag)

    elCharge = 1.60218e-19
    amu = 1.6605390666e-27  # atomic mass unit
    implicitVar = eFieldName

    for i, flux in enumerate(speciesFluxes):
        speciesMass = amu * species[i].atomicA
        speciesCharge = elCharge * species[i].charge

        normConst = sc.CustomNormConst(
            multConst=speciesCharge / speciesMass,
            normNames=["EField", "time", "speed"],
            normPowers=[1.0, 1.0, -1.0],
        )

        vData = sc.VarData(reqRowVars=[speciesDensities[i]])

        newTerm = sc.GeneralMatrixTerm(
            flux,
            implicitVar=implicitVar,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(),
            varData=vData,
        )

        newModel.addTerm("lorentz" + flux, newTerm)

    return newModel


def lorentzForceWork(
    modelTag: str,
    eFieldName: str,
    speciesFluxes: List[str],
    speciesEnergies: List[str],
    species: List[sc.Species],
) -> sc.CustomModel:
    """Generate Lorentz force work terms for each species

    Args:
        eFieldName (str): Name of  electric field variable
        speciesFluxes (List[str]): Names of evolved species fluxes
        speciesEnergies (List[str]): Names of species energies
        species (list[sc.Species]): Species objects for each species

    Returns:
        sc.CustomModel: CustomModel object ready for insertion into JSON config file
    """

    assert len(speciesFluxes) == len(
        species
    ), "speciesFluxes and species passed to lorentzForces model must be of same size"
    assert len(speciesEnergies) == len(
        species
    ), "speciesEnergies and species passed to lorentzForces model must be of same size"

    # Initializing model
    newModel = sc.CustomModel(modelTag=modelTag)

    implicitVar = eFieldName

    for i, flux in enumerate(speciesFluxes):
        speciesCharge = species[i].charge

        normConst = sc.CustomNormConst(
            multConst=speciesCharge,
            normNames=["EField", "time", "speed", "eVTemperature"],
            normPowers=[1.0, 1.0, 1.0, -1.0],
        )

        vData = sc.VarData(reqColVars=[flux])

        # Electron term
        evolvedVar = speciesEnergies[i]

        electronTerm = sc.GeneralMatrixTerm(
            evolvedVar,
            implicitVar=implicitVar,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(),
            varData=vData,
        )

        newModel.addTerm("lorentzWork" + flux, electronTerm)

    return newModel


def implicitTemperatures(
    modelTag: str,
    speciesFluxes: List[str],
    speciesEnergies: List[str],
    speciesDensities: List[str],
    speciesTemperatures: List[str],
    species: List[sc.Species],
    speciesDensitiesDual: Union[List[str], None] = None,
    evolvedXU2Cells: Union[List[int], None] = None,
    ignoreKineticContribution=False,
) -> sc.CustomModel:
    """Generate implicit temperature derivation terms for each species

    Args:
        speciesFluxes (List[str]): Names of evolved species fluxes
        speciesEnergies (List[str]): Names of species energies
        speciesDensities (List[str]): Names of species densities
        speciesTemperatures (List[str]): Names of species temperature
        species (list[sc.Species]): Species objects for each species
        speciesDensitiesDual (Union[List[str],None], optional): Names of species densities on dual grid (use when fluxes are staggered). Defaults to None.
        evolvedXU2Cells (Union[List[int],None], optional): Optional list of evolved X cells in kinetic energy term. Defaults to None, evolving all cells.
        ignoreKineticContribution (bool, optional): Ignores all kinetic contributions to the temperature. Defaults to False.
    Returns:
        sc.CustomModel: CustomModel object ready for insertion into JSON config file
    """

    assert len(speciesFluxes) == len(
        species
    ), "speciesFluxes and species passed to implicitTemperatures model must be of same size"
    assert len(speciesEnergies) == len(
        species
    ), "speciesEnergies and species passed to implicitTemperatures model must be of same size"
    assert len(speciesDensities) == len(
        species
    ), "speciesDensities and species passed to implicitTemperatures model must be of same size"
    assert len(speciesTemperatures) == len(
        species
    ), "speciesEnergies and species passed to implicitTemperatures model must be of same size"

    # Initializing model
    newModel = sc.CustomModel(modelTag=modelTag)

    elCharge = 1.60218e-19
    amu = 1.6605390666e-27  # atomic mass unit

    normConstI = sc.CustomNormConst(multConst=-1.0)
    normConstW = sc.CustomNormConst(multConst=2 / 3)

    for i, temp in enumerate(speciesTemperatures):
        speciesMass = amu * species[i].atomicA

        # Setting normalization constant calculation
        normConstU2 = sc.CustomNormConst(
            multConst=-speciesMass / (3 * elCharge),
            normNames=["speed", "eVTemperature"],
            normPowers=[2.0, -1.0],
        )

        vDataW = sc.VarData(reqRowVars=[speciesDensities[i]], reqRowPowers=[-1.0])
        colDensity = speciesDensities[i]
        if speciesDensitiesDual is not None:
            colDensity = speciesDensitiesDual[i]
        # Kinetic energy term will be implicit in fluxes so need to be converted to speeds
        vDataU2 = sc.VarData(
            reqColVars=[speciesFluxes[i], colDensity], reqColPowers=[1.0, -2.0]
        )

        # Identity term

        identityTerm = sc.GeneralMatrixTerm(
            temp, customNormConst=normConstI, stencilData=sc.diagonalStencil()
        )

        newModel.addTerm("identityTerm" + temp, identityTerm)

        # 2/3 W/n term

        termW = sc.GeneralMatrixTerm(
            temp,
            implicitVar=speciesEnergies[i],
            customNormConst=normConstW,
            varData=vDataW,
            stencilData=sc.diagonalStencil(),
        )

        newModel.addTerm("wTerm" + temp, termW)

        if not ignoreKineticContribution:
            # kinetic energy term

            if evolvedXU2Cells is not None:
                termU2 = sc.GeneralMatrixTerm(
                    temp,
                    implicitVar=speciesFluxes[i],
                    customNormConst=normConstU2,
                    varData=vDataU2,
                    stencilData=sc.diagonalStencil(evolvedXCells=evolvedXU2Cells),
                )
            else:
                termU2 = sc.GeneralMatrixTerm(
                    temp,
                    implicitVar=speciesFluxes[i],
                    customNormConst=normConstU2,
                    varData=vDataU2,
                    stencilData=sc.diagonalStencil(),
                )

            newModel.addTerm("u2Term" + temp, termU2)

    return newModel


def kinAdvX(
    modelTag: str, distFunName: str, gridObj: Grid, evolvedHarmonics: List[int] = []
) -> sc.CustomModel:
    """Return kinetic advection model in x direction

    Args:
        modelTag (str): Tag for the generated model
        distFunName (str): Name of the advected distribution function
        gridObj (Grid): Grid object used to get harmonic and velocity space information
        evolvedHarmonics (List[int], optional): List of evolved harmonic indices (Fortran 1-indices!). Defaults to [], evolving all harmonics.

    Returns:
        sc.CustomModel: CustomModel object ready to be converted into JSON input data
    """

    evolvedVar = distFunName

    usedHarmonics = list(range(1, gridObj.numH() + 1))
    if len(evolvedHarmonics) > 0:
        usedHarmonics = evolvedHarmonics

    lNums = [gridObj.lGrid[i - 1] for i in range(1, gridObj.numH() + 1)]
    mNums = [gridObj.mGrid[i - 1] for i in range(1, gridObj.numH() + 1)]

    newModel = sc.CustomModel(modelTag=modelTag)
    for harmonic in usedHarmonics:
        if lNums[harmonic - 1] > 0:
            normConst = sc.CustomNormConst(
                multConst=-(lNums[harmonic - 1] - mNums[harmonic - 1])
                / (2.0 * lNums[harmonic - 1] - 1.0),
                normNames=["time", "velGrid", "length"],
                normPowers=[1.0, 1.0, -1.0],
            )
            newTerm = sc.GeneralMatrixTerm(
                evolvedVar,
                velocityProfile=gridObj.vGrid.tolist(),
                customNormConst=normConst,
                stencilData=sc.kineticSpatialDiffStencil(
                    harmonic,
                    gridObj.getH(
                        lNum=lNums[harmonic - 1] - 1,
                        mNum=mNums[harmonic - 1],
                        im=gridObj.imaginaryHarmonic[harmonic - 1],
                    ),
                ),
                fixedMatrix=True,
            )

            newModel.addTerm("adv_minus" + str(harmonic), newTerm)

        if lNums[harmonic - 1] < gridObj.lMax:
            normConst = sc.CustomNormConst(
                multConst=-(lNums[harmonic - 1] + mNums[harmonic - 1] + 1.0)
                / (2.0 * lNums[harmonic - 1] + 3.0),
                normNames=["time", "velGrid", "length"],
                normPowers=[1.0, 1.0, -1.0],
            )
            newTerm = sc.GeneralMatrixTerm(
                evolvedVar,
                velocityProfile=gridObj.vGrid.tolist(),
                customNormConst=normConst,
                stencilData=sc.kineticSpatialDiffStencil(
                    harmonic,
                    gridObj.getH(
                        lNum=lNums[harmonic - 1] + 1,
                        mNum=mNums[harmonic - 1],
                        im=gridObj.imaginaryHarmonic[harmonic - 1],
                    ),
                ),
                fixedMatrix=True,
            )

            newModel.addTerm("adv_plus" + str(harmonic), newTerm)

    return newModel


def ghDerivations(wrapper: RKWrapper) -> dict:
    """Returns derivation collection dictionary for Gl and Hl derivations used for E-field terms

    Args:
        wrapper (RKWrapper): Wrapper to use when building the derivations

    Returns:
        dict: Derivation collection dictionary to be added to the wrapper
    """

    vGrid = wrapper.grid.vGrid
    dv: List[float] = []

    for v in vGrid:
        dv.append(2 * (v - sum(dv)))

    vBoundary = [vGrid[i] + dv[i] / 2 for i in range(len(dv))]
    numH = wrapper.grid.numH()

    derivCollection = {}
    # Add custom derivations and modelbound data
    for harmonic in range(1, numH + 1):
        lNum = wrapper.grid.lGrid[harmonic - 1]
        innerV = [[v ** (-lNum) for v in vBoundary]]
        innerV[0][-1] = 0
        # Add G_l derivations
        vifAtZero = (
            [
                [
                    1.0 / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
                    -vGrid[0] ** 2
                    / vGrid[1] ** 2
                    / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
                ]
            ]
            if lNum == 0
            else [[1.0 / vGrid[1] ** lNum, 0]]
        )
        derivG = sc.ddvDerivation(
            [harmonic],
            innerV=innerV,
            outerV=[[v**lNum for v in vGrid.tolist()]],
            vifAtZero=vifAtZero,
        )
        derivCollection["G_h=" + str(harmonic)] = derivG

        innerV = [[v ** (lNum + 1) for v in vBoundary]]
        innerV[0][-1] = 0
        # Add H_l derivations
        derivH = sc.ddvDerivation(
            [harmonic],
            innerV=innerV,
            outerV=[[v ** (-lNum - 1) for v in vGrid.tolist()]],
        )
        derivCollection["H_h=" + str(harmonic)] = derivH

    return derivCollection


def advectionEx(
    modelTag: str,
    distFunName: str,
    eFieldName: str,
    wrapper: RKWrapper,
    dualDistFun: Union[str, None] = None,
) -> sc.CustomModel:
    """Returns electric field advection terms in x direction. Adds necessary custom derivations to the wrapper.

    Args:
        modelTag (str): Tag for the generated model
        distFunName (str): Name of the evolved distribution function
        eFieldName (str): Name of the implicit E-field variable
        wrapper (RKWrapper): Wrapper object to add custom derivations to
        dualDistFun (Union[str,None], optional): Optional distribution function to be used as the required variable in derivations (useful for staggered grids). Defaults to None.
    """

    numH = wrapper.grid.numH()
    derivReqFun = distFunName if dualDistFun is None else dualDistFun
    mbData = sc.VarlikeModelboundData()
    # Add custom derivations and modelbound data

    wrapper.addDerivationCollection(ghDerivations(wrapper))
    for harmonic in range(1, numH + 1):
        mbData.addVariable(
            "G_h=" + str(harmonic),
            derivationRule=sc.derivationRule("G_h=" + str(harmonic), [derivReqFun]),
            isSingleHarmonic=True,
        )
        mbData.addVariable(
            "H_h=" + str(harmonic),
            derivationRule=sc.derivationRule("H_h=" + str(harmonic), [derivReqFun]),
            isSingleHarmonic=True,
        )

    newModel = sc.CustomModel(modelTag=modelTag)

    newModel.setModelboundData(mbData.dict())

    elCharge = 1.60218e-19
    elMass = 9.10938e-31

    lNums = [wrapper.grid.lGrid[i] for i in range(numH)]
    mNums = [wrapper.grid.mGrid[i] for i in range(numH)]

    chargeMassRatio = elCharge / elMass
    for harmonic in range(1, numH + 1):
        # Add G terms
        if lNums[harmonic - 1] > 0:
            normConst = sc.CustomNormConst(
                multConst=chargeMassRatio
                * (lNums[harmonic - 1] - mNums[harmonic - 1])
                / (2.0 * lNums[harmonic - 1] - 1.0),
                normNames=["EField", "time", "velGrid"],
                normPowers=[1.0, 1.0, -1.0],
            )
            vData = sc.VarData(
                reqMBRowVars=[
                    "G_h="
                    + str(
                        wrapper.grid.getH(
                            lNum=lNums[harmonic - 1] - 1,
                            mNum=mNums[harmonic - 1],
                            im=wrapper.grid.imaginaryHarmonic[harmonic - 1],
                        )
                    )
                ]
            )

            newTerm = sc.GeneralMatrixTerm(
                distFunName,
                eFieldName,
                customNormConst=normConst,
                varData=vData,
                stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
            )

            newModel.addTerm("eAdv_G" + str(harmonic), newTerm)

        # Add H terms
        if lNums[harmonic - 1] < wrapper.grid.lMax:
            normConst = sc.CustomNormConst(
                multConst=chargeMassRatio
                * (lNums[harmonic - 1] + mNums[harmonic - 1] + 1.0)
                / (2.0 * lNums[harmonic - 1] + 3.0),
                normNames=["EField", "time", "velGrid"],
                normPowers=[1.0, 1.0, -1.0],
            )
            vData = sc.VarData(
                reqMBRowVars=[
                    "H_h="
                    + str(
                        wrapper.grid.getH(
                            lNum=lNums[harmonic - 1] + 1,
                            mNum=mNums[harmonic - 1],
                            im=wrapper.grid.imaginaryHarmonic[harmonic - 1],
                        )
                    )
                ]
            )

            newTerm = sc.GeneralMatrixTerm(
                distFunName,
                eFieldName,
                customNormConst=normConst,
                varData=vData,
                stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
            )

            newModel.addTerm("eAdv_H" + str(harmonic), newTerm)

    return newModel


def addExAdvectionModel(
    modelTag: str,
    distFunName: str,
    eFieldName: str,
    wrapper: RKWrapper,
    dualDistFun: Union[str, None] = None,
) -> None:
    """Adds electric field advection terms in x direction. Adds necessary custom derivations to the wrapper.

    Args:
        modelTag (str): Tag for the generated model
        distFunName (str): Name of the evolved distribution function
        eFieldName (str): Name of the implicit E-field variable
        wrapper (RKWrapper): Wrapper object to add the model and add custom derivations to
        dualDistFun (Union[str,None], optional): Optional distribution function to be used as the required variable in derivations (useful for staggered grids). Defaults to None.
    """
    newModel = advectionEx(modelTag, distFunName, eFieldName, wrapper, dualDistFun)

    wrapper.addModel(newModel.dict())


def eeCollIsotropic(
    modelTag: str, distFunName: str, elTempVar: str, elDensVar: str, wrapper: RKWrapper
) -> sc.CustomModel:
    """Return e-e collision model for l=0 harmonic

    Args:
        modelTag (str): Tag of the model to be added
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable
        elDensVar (str): Name of the electron density variable
        wrapper (RKWrapper): Wrapper to add custom derivations to
    """

    if "f0Extraction" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation("f0Extraction", sc.harmonicExtractorDerivation(1))
    mbData = sc.VarlikeModelboundData()
    mbData.addVariable(
        "f0", sc.derivationRule("f0Extraction", [distFunName]), isSingleHarmonic=True
    )
    mbData.addVariable(
        "dragCCL",
        sc.derivationRule("cclDragCoeff", ["f0"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "diffCCL",
        sc.derivationRule("cclDiffusionCoeff", ["f0", "weightCCL"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "weightCCL",
        sc.derivationRule("cclWeight", ["dragCCL", "diffCCL"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable("logLee", sc.derivationRule("logLee", [elTempVar, elDensVar]))

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)
    normConst = sc.CustomNormConst(
        multConst=gamma0norm,
        normNames=["density", "time", "velGrid"],
        normPowers=[1.0, 1.0, -3.0],
    )
    varData = sc.VarData(reqMBRowVars=["logLee"])
    vOuter = [1.0 / v**2 for v in wrapper.grid.vGrid]

    newModel = sc.CustomModel(modelTag=modelTag)
    newModel.setModelboundData(mbData.dict())

    # Add drag term

    dragTerm = sc.GeneralMatrixTerm(
        distFunName,
        customNormConst=normConst,
        velocityProfile=vOuter,
        varData=varData,
        stencilData=sc.ddvStencil(
            1, 1, modelboundC="dragCCL", modelboundInterp="weightCCL"
        ),
    )

    newModel.addTerm("dragTerm", dragTerm)

    # Add diffusion term

    diffTerm = sc.GeneralMatrixTerm(
        distFunName,
        customNormConst=normConst,
        velocityProfile=vOuter,
        varData=varData,
        stencilData=sc.velDiffusionStencil(1, 1, modelboundA="diffCCL"),
    )

    newModel.addTerm("diffTerm", diffTerm)

    return newModel


def addEECollIsotropic(
    modelTag: str, distFunName: str, elTempVar: str, elDensVar: str, wrapper: RKWrapper
) -> None:
    """Add e-e collision terms for l=0 harmonic

    Args:
        modelTag (str): Tag of the model to be added
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable
        elDensVar (str): Name of the electron density variable
        wrapper (RKWrapper): Wrapper to add the term to
    """

    newModel = eeCollIsotropic(modelTag, distFunName, elTempVar, elDensVar, wrapper)

    wrapper.addModel(newModel.dict())


def eiCollIsotropic(
    modelTag: str,
    distFunName: str,
    elTempVar: str,
    elDensVar: str,
    ionTempVar: str,
    ionDensVar: str,
    ionSpeciesName: str,
    ionEnVar: Union[str, None],
    wrapper: RKWrapper,
) -> sc.CustomModel:
    """Return e-i collision model for l=0 harmonic

    Args:
        modelTag (str): Tag of the model to be added
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable
        elDensVar (str): Name of the electron density variable
        ionTempVar (str): Name of the ion temperature variable
        ionDensVar (str): Name of the ion density variable
        ionSpeciesName (str): Name of ion species colliding with the electrons
        ionEnVar (Union[str,None]): Name of the ion energy density variable. If present, the energy moment of the collision operator
                                    will be added this variable. Defaults to None.
        wrapper (RKWrapper): Wrapper to add custom derivations to
    """

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity
    amu = 1.6605390666e-27  # atomic mass unit
    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)

    ionSpecies = wrapper.getSpecies(ionSpeciesName)
    gamma0norm = (
        gamma0norm * ionSpecies.charge**2 * elMass / (ionSpecies.atomicA * amu)
    )

    mbData = sc.VarlikeModelboundData()
    mbData.addVariable(
        "logLei", sc.derivationRule("logLei" + ionSpeciesName, [elTempVar, elDensVar])
    )

    normConst = sc.CustomNormConst(
        multConst=gamma0norm,
        normNames=["density", "time", "velGrid"],
        normPowers=[1.0, 1.0, -3.0],
    )
    varDataDrag = sc.VarData(reqMBRowVars=["logLei"], reqRowVars=[ionDensVar])
    varDataDiff = sc.VarData(
        reqMBRowVars=["logLei"], reqRowVars=[ionDensVar, ionTempVar]
    )

    vGrid = wrapper.grid.vGrid
    dv = wrapper.grid.vWidths
    vBoundary = [vGrid[i] + dv[i] / 2 for i in range(len(dv))]
    vOuter = [1.0 / v**2 for v in wrapper.grid.vGrid]
    innerV = [1.0 / (2 * v) for v in vBoundary]

    newModel = sc.CustomModel(modelTag=modelTag)
    newModel.setModelboundData(mbData.dict())

    # Add drag term
    dragTerm = sc.GeneralMatrixTerm(
        distFunName,
        customNormConst=normConst,
        velocityProfile=vOuter,
        varData=varDataDrag,
        stencilData=sc.ddvStencil(1, 1),
    )  # No cfAtZero to make sure particles are conserved

    newModel.addTerm("dragTerm", dragTerm)

    # Add diffusion term

    diffTerm = sc.GeneralMatrixTerm(
        distFunName,
        customNormConst=normConst,
        velocityProfile=vOuter,
        varData=varDataDiff,
        stencilData=sc.velDiffusionStencil(1, 1, fixedA=innerV),
    )

    newModel.addTerm("diffTerm", diffTerm)

    if ionEnVar is not None:
        normIon = sc.CustomNormConst(multConst=-1.0)

        diffTermIon = sc.GeneralMatrixTerm(
            ionEnVar,
            distFunName,
            customNormConst=normIon,
            stencilData=sc.termMomentStencil(1, 2, "diffTerm"),
        )

        newModel.addTerm("diffTermIon", diffTermIon)

        dragTermIon = sc.GeneralMatrixTerm(
            ionEnVar,
            distFunName,
            customNormConst=normIon,
            stencilData=sc.termMomentStencil(1, 2, "dragTerm"),
        )

        newModel.addTerm("dragTermIon", dragTermIon)

    return newModel


def addEICollIsotropic(
    modelTag: str,
    distFunName: str,
    elTempVar: str,
    elDensVar: str,
    ionTempVar: str,
    ionDensVar: str,
    ionSpeciesName: str,
    wrapper: RKWrapper,
    ionEnVar: Union[str, None] = None,
) -> None:
    """Add e-i collision terms for l=0 harmonic, as well as the corresponding ion energy terms if ionEnVar is present

    Args:
        modelTag (str): Tag of the model to be added
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable
        elDensVar (str): Name of the electron density variable
        ionTempVar (str): Name of the ion temperature variable
        ionDensVar (str): Name of the ion density variable
        ionSpeciesName (str): Name of ion species colliding with the electrons
        wrapper (RKWrapper): Wrapper to add the terms to
        ionEnVar (Union[str,None]): Name of the ion energy density variable. If present, the energy moment of the collision operator
                                    will be added this variable. Defaults to None.
    """

    newModel = eiCollIsotropic(
        modelTag,
        distFunName,
        elTempVar,
        elDensVar,
        ionTempVar,
        ionDensVar,
        ionSpeciesName,
        ionEnVar,
        wrapper,
    )

    wrapper.addModel(newModel.dict())


def stationaryIonEIColl(
    modelTag: str,
    distFunName: str,
    ionDensVar: str,
    electronDensVar: str,
    electronTempVar: str,
    ionSpeciesName: str,
    evolvedHarmonics: List[int],
    wrapper: RKWrapper,
) -> sc.CustomModel:
    """Return stationary ion electron-ion collision operator model

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str): Name of electron distribution function variable
        ionDensVar (str): Name of ion density variable
        electronDensVar (str): Name of electron density variable
        electronTempVar (str): Name of electron temperature variable
        ionSpeciesName (str): Name of ion species colliding with the electrons
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating even and odd harmonics on staggered grid)
        wrapper (RKWrapper): Wrapper to use
    """

    mbData = sc.VarlikeModelboundData()
    mbData.addVariable(
        "logLei",
        sc.derivationRule(
            "logLei" + ionSpeciesName, [electronTempVar, electronDensVar]
        ),
    )

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity
    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)

    ionSpecies = wrapper.getSpecies(ionSpeciesName)
    gamma0norm = gamma0norm * ionSpecies.charge**2

    varData = sc.VarData(reqRowVars=[ionDensVar], reqMBRowVars=["logLei"])

    newModel = sc.CustomModel(modelTag=modelTag)
    newModel.setModelboundData(mbData.dict())

    normConst = sc.CustomNormConst(
        multConst=-gamma0norm,
        normNames=["density", "time", "velGrid"],
        normPowers=[1.0, 1.0, -3.0],
    )

    lNums = wrapper.grid.lGrid
    hProfile = []
    for l in lNums:
        hProfile.append(l * (l + 1.0) / 2.0)

    vProfile = [1.0 / v**3 for v in wrapper.grid.vGrid]

    collTerm = sc.GeneralMatrixTerm(
        distFunName,
        harmonicProfile=hProfile,
        velocityProfile=vProfile,
        customNormConst=normConst,
        varData=varData,
        stencilData=sc.diagonalStencil(evolvedHarmonics=evolvedHarmonics),
    )

    newModel.addTerm("eiCollStationaryIons", collTerm)

    return newModel


def flowingIonEIColl(
    modelTag: str,
    distFunName: str,
    ionDensVar: str,
    ionFlowSpeedVar: str,
    electronDensVar: str,
    electronTempVar: str,
    ionSpeciesName: str,
    wrapper: RKWrapper,
    evolvedHarmonics: List[int],
    dualDistFun: Union[str, None] = None,
    ionFluxVar: Union[str, None] = None,
) -> sc.CustomModel:
    # NOTE: Needs a lot of work on optimization
    assert (
        1 not in evolvedHarmonics
    ), "flowingIonEIColl cannot be used to evolve harmonic with index 1"

    vGrid = wrapper.grid.vGrid
    lNums = wrapper.grid.lGrid
    if "f0Extraction" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation("f0Extraction", sc.harmonicExtractorDerivation(1))

    if "CII0" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation("CII0", sc.coldIonIJIntegralDerivation(0))
    if "CII2" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation("CII2", sc.coldIonIJIntegralDerivation(2))
    if "CIJ-1" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation("CIJ-1", sc.coldIonIJIntegralDerivation(-1, True))
    if "sumTerm" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation(
            "sumTerm",
            sc.polyDerivation(
                0.0, polyPowers=[1.0, 1.0, 1.0], polyCoeffs=[-1.0, 2.0, 3.0]
            ),
        )
        wrapper.addCustomDerivation(
            "df0/dv",
            sc.ddvDerivation(
                [1],
                vifAtZero=[
                    [
                        1.0 / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
                        -vGrid[0] ** 2
                        / vGrid[1] ** 2
                        / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
                    ]
                ],
            ),
        )
        wrapper.addCustomDerivation("d2f0/dv2", sc.d2dv2Derivation([1]))
    if "sumTerm2" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation(
            "sumTerm2",
            sc.polyDerivation(0.0, polyPowers=[1.0, 1.0], polyCoeffs=[-1.0, 2.0]),
        )

    derivReqFun = distFunName if dualDistFun is None else dualDistFun

    mbData = sc.VarlikeModelboundData()
    mbData.addVariable(
        "CII0", sc.derivationRule("CII0", [ionFlowSpeedVar]), isDistribution=True
    )
    mbData.addVariable(
        "CII2", sc.derivationRule("CII2", [ionFlowSpeedVar]), isDistribution=True
    )
    mbData.addVariable(
        "CIJ-1", sc.derivationRule("CIJ-1", [ionFlowSpeedVar]), isDistribution=True
    )
    mbData.addVariable(
        "CII0sh",
        sc.derivationRule("f0Extraction", ["CII0"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "CII2sh",
        sc.derivationRule("f0Extraction", ["CII2"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "CIJ-1sh",
        sc.derivationRule("f0Extraction", ["CIJ-1"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "IJSum",
        sc.derivationRule("sumTerm", ["CII2sh", "CIJ-1sh", "CII0sh"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "IJSum2",
        sc.derivationRule("sumTerm2", ["CII2sh", "CIJ-1sh"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "logLei",
        sc.derivationRule(
            "logLei" + ionSpeciesName, [electronTempVar, electronDensVar]
        ),
    )
    mbData.addVariable(
        "df0", sc.derivationRule("df0/dv", [derivReqFun]), isSingleHarmonic=True
    )
    mbData.addVariable(
        "ddf0", sc.derivationRule("d2f0/dv2", [derivReqFun]), isSingleHarmonic=True
    )

    for h in evolvedHarmonics:
        l = lNums[h - 1]
        if "CII" + str(l) not in wrapper.customDerivs["tags"]:
            wrapper.addCustomDerivation(
                "CII" + str(l), sc.coldIonIJIntegralDerivation(l)
            )
            mbData.addVariable(
                "CII" + str(l),
                sc.derivationRule("CII" + str(l), [ionFlowSpeedVar]),
                isDistribution=True,
            )
        if "CII" + str(l + 2) not in wrapper.customDerivs["tags"]:
            wrapper.addCustomDerivation(
                "CII" + str(l + 2), sc.coldIonIJIntegralDerivation(l + 2)
            )
            mbData.addVariable(
                "CII" + str(l + 2),
                sc.derivationRule("CII" + str(l + 2), [ionFlowSpeedVar]),
                isDistribution=True,
            )
        if "CIJ" + str(-l - 1) not in wrapper.customDerivs["tags"]:
            wrapper.addCustomDerivation(
                "CIJ" + str(-l - 1), sc.coldIonIJIntegralDerivation(-l - 1, True)
            )
            mbData.addVariable(
                "CIJ" + str(-l - 1),
                sc.derivationRule("CIJ" + str(-l - 1), [ionFlowSpeedVar]),
                isDistribution=True,
            )
        if "CIJ" + str(1 - l) not in wrapper.customDerivs["tags"]:
            wrapper.addCustomDerivation(
                "CIJ" + str(1 - l), sc.coldIonIJIntegralDerivation(1 - l, True)
            )
            mbData.addVariable(
                "CIJ" + str(1 - l),
                sc.derivationRule("CIJ" + str(1 - l), [ionFlowSpeedVar]),
                isDistribution=True,
            )

    newModel = sc.CustomModel(modelTag=modelTag)
    newModel.setModelboundData(mbData.dict())

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity
    amu = 1.6605390666e-27  # atomic mass unit
    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)
    ionSpecies = wrapper.getSpecies(ionSpeciesName)
    gamma0norm = gamma0norm * ionSpecies.charge**2
    elIonMassRatio = elMass / (ionSpecies.atomicA * amu)

    normIonFriction = sc.CustomNormConst(
        multConst=-elIonMassRatio / 3,
        normNames=["velGrid", "speed"],
        normPowers=[
            1.0,
            -1.0,
        ],
    )

    for harmonic in evolvedHarmonics:
        l = lNums[harmonic - 1]

        # velocity diffusion terms with fl

        vProfile = [1 / v for v in vGrid]
        normConst = sc.CustomNormConst(
            multConst=gamma0norm / 3,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )
        varDataI = sc.VarData(reqMBRowVars=["logLei", "CII2sh"])
        varDataJ = sc.VarData(reqMBRowVars=["logLei", "CIJ-1sh"])

        adfdvAtZero = [0, 0] if l > 1 else [1.0 / vGrid[1], 0]
        termI = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            customNormConst=normConst,
            varData=varDataI,
            stencilData=sc.velDiffusionStencil(
                harmonic, harmonic, adfAtZero=cast(List[float], adfdvAtZero)
            ),
        )
        newModel.addTerm("diffTermI" + str(harmonic), termI)

        termJ = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            customNormConst=normConst,
            varData=varDataJ,
            stencilData=sc.velDiffusionStencil(
                harmonic, harmonic, adfAtZero=cast(List[float], adfdvAtZero)
            ),
        )
        newModel.addTerm("diffTermJ" + str(harmonic), termJ)

        # velocity deriv terms with fl

        vProfile = [1 / v**2 for v in vGrid]
        normConst = sc.CustomNormConst(
            multConst=gamma0norm / 3,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )
        varData = sc.VarData(reqMBRowVars=["logLei", "IJSum2"])

        termdfldv = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            customNormConst=normConst,
            varData=varData,
            stencilData=sc.ddvStencil(harmonic, harmonic),
        )
        newModel.addTerm("dfdv" + str(harmonic), termdfldv)

        # -l(l+1)/2 terms

        vProfile = [1 / v**3 for v in vGrid]
        normConst = sc.CustomNormConst(
            multConst=-(l * (l + 1.0) / 2.0) * gamma0norm / 3,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )
        varData = sc.VarData(reqMBRowVars=["logLei", "IJSum"])

        termll = sc.GeneralMatrixTerm(
            distFunName,
            customNormConst=normConst,
            varData=varData,
            velocityProfile=vProfile,
            stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
        )
        newModel.addTerm("termLL" + str(harmonic), termll)

        # I/J(f_l) terms

        # d2f0/dv2 terms
        vProfile = [1 / v for v in vGrid]
        # C1*I_{l+2} term
        varData = sc.VarData(reqMBRowVars=["logLei", "ddf0", "CII" + str(l + 2)])
        C = (l + 1) * (l + 2) / ((2 * l + 1) * (2 * l + 3))  # C1
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm / 2,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )

        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            implicitVar=ionDensVar,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
        )

        newModel.addTerm("C1Il+2_h=" + str(harmonic), newTerm)

        # C1*J_{-l-1} term
        varData = sc.VarData(reqMBRowVars=["logLei", "ddf0", "CIJ" + str(-1 - l)])
        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            implicitVar=ionDensVar,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
            skipPattern=True,
        )

        newModel.addTerm("C1J-l-1_h=" + str(harmonic), newTerm)

        # C2*I_l term
        varData = sc.VarData(reqMBRowVars=["logLei", "ddf0", "CII" + str(l)])
        C = -(l - 1) * l / ((2 * l + 1) * (2 * l - 1))  # C2
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm / 2,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )

        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            implicitVar=ionDensVar,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
            skipPattern=True,
        )

        newModel.addTerm("C2Il_h=" + str(harmonic), newTerm)

        # C2*J1-l term
        varData = sc.VarData(reqMBRowVars=["logLei", "ddf0", "CIJ" + str(1 - l)])
        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            implicitVar=ionDensVar,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
            skipPattern=True,
        )

        newModel.addTerm("C2J1-l_h=" + str(harmonic), newTerm)

        # df0/dv terms

        vProfile = [1 / v**2 for v in vGrid]

        # C3*I_{l+2} term
        varData = sc.VarData(reqMBRowVars=["logLei", "df0", "CII" + str(l + 2)])
        C = -((l + 1) * l / 2 + l + 1) / ((2 * l + 1) * (2 * l + 3))  # C3
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )

        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            implicitVar=ionDensVar,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
            skipPattern=True,
        )

        newModel.addTerm("C3Il+2_h=" + str(harmonic), newTerm)

        # C4*J_{-l-1} term
        varData = sc.VarData(reqMBRowVars=["logLei", "df0", "CIJ" + str(-1 - l)])
        C = (-(l + 1) * l / 2 + l + 2) / ((2 * l + 1) * (2 * l + 3)) + l / (
            2 * l + 1
        )  # C4 + l/(2l+1)
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )
        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            implicitVar=ionDensVar,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
            skipPattern=True,
        )

        newModel.addTerm("C4J-l-1_h=" + str(harmonic), newTerm)

        # C5*I_l term
        varData = sc.VarData(reqMBRowVars=["logLei", "df0", "CII" + str(l)])
        C = ((l + 1) * l / 2 + l - 1) / ((2 * l + 1) * (2 * l - 1)) - (l + 1) / (
            2 * l + 1
        )  # C5 -(l+1)/(2l+1)
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )

        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            implicitVar=ionDensVar,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
            skipPattern=True,
        )

        newModel.addTerm("C5Il_h=" + str(harmonic), newTerm)

        # C6*J1-l term
        varData = sc.VarData(reqMBRowVars=["logLei", "df0", "CIJ" + str(1 - l)])
        C = -((l + 1) * l / 2 - l) / ((2 * l + 1) * (2 * l - 1))  # C6
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )
        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            implicitVar=ionDensVar,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(evolvedHarmonics=[harmonic]),
            skipPattern=True,
        )

        newModel.addTerm("C6J1-l_h=" + str(harmonic), newTerm)

    # Add ion friction terms if ionFluxVar is passed
    if 2 in evolvedHarmonics and ionFluxVar is not None:
        diffTermIIon = sc.GeneralMatrixTerm(
            ionFluxVar,
            distFunName,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "diffTermI2"),
        )

        newModel.addTerm("diffTermIIon", diffTermIIon)

        diffTermJIon = sc.GeneralMatrixTerm(
            ionFluxVar,
            distFunName,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "diffTermJ2"),
            skipPattern=True,
        )

        newModel.addTerm("diffTermJIon", diffTermJIon)

        dfdvTermIon = sc.GeneralMatrixTerm(
            ionFluxVar,
            distFunName,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "dfdv2"),
            skipPattern=True,
        )

        newModel.addTerm("dfdvTermIon", dfdvTermIon)

        llTermIon = sc.GeneralMatrixTerm(
            ionFluxVar,
            distFunName,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "termLL2"),
            skipPattern=True,
        )

        newModel.addTerm("llTermIon", llTermIon)

        ddf0TermIon1 = sc.GeneralMatrixTerm(
            ionFluxVar,
            ionDensVar,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "C1Il+2_h=2"),
        )

        newModel.addTerm("ddf0TermIon1", ddf0TermIon1)

        ddf0TermIon2 = sc.GeneralMatrixTerm(
            ionFluxVar,
            ionDensVar,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "C1J-l-1_h=2"),
            skipPattern=True,
        )

        newModel.addTerm("ddf0TermIon2", ddf0TermIon2)

        ddf0TermIon3 = sc.GeneralMatrixTerm(
            ionFluxVar,
            ionDensVar,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "C2Il_h=2"),
            skipPattern=True,
        )

        newModel.addTerm("ddf0TermIon3", ddf0TermIon3)

        ddf0TermIon4 = sc.GeneralMatrixTerm(
            ionFluxVar,
            ionDensVar,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "C2J1-l_h=2"),
            skipPattern=True,
        )

        newModel.addTerm("ddf0TermIon4", ddf0TermIon4)

        df0TermIon1 = sc.GeneralMatrixTerm(
            ionFluxVar,
            ionDensVar,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "C3Il+2_h=2"),
        )

        newModel.addTerm("df0TermIon1", df0TermIon1)

        df0TermIon2 = sc.GeneralMatrixTerm(
            ionFluxVar,
            ionDensVar,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "C4J-l-1_h=2"),
            skipPattern=True,
        )

        newModel.addTerm("df0TermIon2", df0TermIon2)

        df0TermIon3 = sc.GeneralMatrixTerm(
            ionFluxVar,
            ionDensVar,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "C5Il_h=2"),
            skipPattern=True,
        )

        newModel.addTerm("df0TermIon3", df0TermIon3)

        df0TermIon4 = sc.GeneralMatrixTerm(
            ionFluxVar,
            ionDensVar,
            customNormConst=normIonFriction,
            stencilData=sc.termMomentStencil(2, 1, "C6J1-l_h=2"),
            skipPattern=True,
        )

        newModel.addTerm("df0TermIon4", df0TermIon4)

    return newModel


def addStationaryIonEIColl(
    modelTag: str,
    distFunName: str,
    ionDensVar: str,
    electronDensVar: str,
    electronTempVar: str,
    ionSpeciesName: str,
    evolvedHarmonics: List[int],
    wrapper: RKWrapper,
) -> None:
    """Add stationary ion electron-ion collision operator model to wrapper

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str): Name of electron distribution function variable
        ionDensVar (str): Name of ion density variable
        electronDensVar (str): Name of electron density variable
        electronTempVar (str): Name of electron temperature variable
        ionSpeciesName (str): Name of ion species colliding with the electrons
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating even and odd harmonics on staggered grid)
        wrapper (RKWrapper): Wrapper to add the model to
    """

    newModel = stationaryIonEIColl(
        modelTag,
        distFunName,
        ionDensVar,
        electronDensVar,
        electronTempVar,
        ionSpeciesName,
        evolvedHarmonics,
        wrapper,
    )

    wrapper.addModel(newModel.dict())


def addFlowingIonEIColl(
    modelTag: str,
    distFunName: str,
    ionDensVar: str,
    ionFlowSpeedVar: str,
    electronDensVar: str,
    electronTempVar: str,
    ionSpeciesName: str,
    wrapper: RKWrapper,
    evolvedHarmonics: List[int],
    dualDistFun: Union[str, None] = None,
    ionFluxVar: Union[str, None] = None,
) -> None:
    """Add flowing cold ion electron-ion collision model to wrapper

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str):  Name of the electron distribution function variable
        ionDensVar (str): Name of ion density variable
        ionFlowSpeedVar (str): Name of ion flow speed variable
        electronDensVar (str): Name of electron density variable
        electronTempVar (str): Name of electron temperature variable
        ionSpeciesName (str): Name of the ion species
        wrapper (RKWrapper): Wrapper to add model to
        evolvedHarmonics (List[int]): List of evolved harmonics
        dualDistFun (Union[str,None], optional): Interpolated distribution function to be used instead the default for derivation of modelbound variables. Defaults to None.
        ionFluxVar (Union[str,None], optional): Ion flux variable - when present generates ion friction terms corresponding to the electron-ion collisions. Defaults to None.
    """
    newModel = flowingIonEIColl(
        modelTag,
        distFunName,
        ionDensVar,
        ionFlowSpeedVar,
        electronDensVar,
        electronTempVar,
        ionSpeciesName,
        wrapper,
        evolvedHarmonics,
        dualDistFun,
        ionFluxVar,
    )

    wrapper.addModel(newModel.dict())


def eeCollHigherL(
    modelTag: str,
    distFunName: str,
    elTempVar: str,
    elDensVar: str,
    wrapper: RKWrapper,
    evolvedHarmonics: List[int],
    dualDistFun: Union[str, None] = None,
) -> sc.CustomModel:
    """Return e-e collision model for l>0

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable
        elDensVar (str): Name of the electron density variable
        wrapper (RKWrapper): Wrapper to add derivations to
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating odd and even harmonics on staggered grid)
        dualDistFun (Union[str,None], optional): Interpolated distribution function to be used instead the default for derivation of modelbound variables. Defaults to None.
    """

    assert (
        1 not in evolvedHarmonics
    ), "addEECollHigherL cannot be used to evolve harmonic with index 1"

    vGrid = wrapper.grid.vGrid
    lNums = wrapper.grid.lGrid
    derivReqFun = distFunName if dualDistFun is None else dualDistFun
    if "f0Extraction" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation("f0Extraction", sc.harmonicExtractorDerivation(1))
    if "I0" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation("I0", sc.ijIntegralDerivation(0))
    if "I2" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation("I2", sc.ijIntegralDerivation(2))
    if "J-1" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation("J-1", sc.ijIntegralDerivation(-1, True))
    if "sumTerm" not in wrapper.customDerivs["tags"]:
        wrapper.addCustomDerivation(
            "sumTerm",
            sc.polyDerivation(
                0.0, polyPowers=[1.0, 1.0, 1.0], polyCoeffs=[-1.0, 2.0, 3.0]
            ),
        )
        wrapper.addCustomDerivation(
            "df0/dv",
            sc.ddvDerivation(
                [1],
                vifAtZero=[
                    [
                        1.0 / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
                        -vGrid[0] ** 2
                        / vGrid[1] ** 2
                        / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
                    ]
                ],
            ),
        )
        wrapper.addCustomDerivation("d2f0/dv2", sc.d2dv2Derivation([1]))

    mbData = sc.VarlikeModelboundData()
    mbData.addVariable(
        "f0", sc.derivationRule("f0Extraction", [derivReqFun]), isSingleHarmonic=True
    )
    mbData.addVariable(
        "I0",
        sc.derivationRule("I0", ["f0"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "I2",
        sc.derivationRule("I2", ["f0"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "J-1",
        sc.derivationRule("J-1", ["f0"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable(
        "IJSum",
        sc.derivationRule("sumTerm", ["I2", "J-1", "I0"]),
        isSingleHarmonic=True,
        isDerivedFromOtherData=True,
    )
    mbData.addVariable("logLee", sc.derivationRule("logLee", [elTempVar, elDensVar]))
    mbData.addVariable(
        "df0", sc.derivationRule("df0/dv", [derivReqFun]), isSingleHarmonic=True
    )
    mbData.addVariable(
        "ddf0", sc.derivationRule("d2f0/dv2", [derivReqFun]), isSingleHarmonic=True
    )

    newModel = sc.CustomModel(modelTag=modelTag)
    newModel.setModelboundData(mbData.dict())

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)

    # 8*pi*f0*fl term
    normConst = sc.CustomNormConst(
        multConst=8 * np.pi * gamma0norm,
        normNames=["density", "time", "velGrid"],
        normPowers=[1.0, 1.0, -3.0],
    )
    varData = sc.VarData(reqMBRowVars=["logLee", "f0"])
    termf0fl = sc.GeneralMatrixTerm(
        distFunName,
        customNormConst=normConst,
        varData=varData,
        stencilData=sc.diagonalStencil(evolvedHarmonics=evolvedHarmonics),
    )
    newModel.addTerm("8pi*f0*fl", termf0fl)

    # velocity diffusion terms with fl

    vProfile = [1 / v for v in vGrid]
    normConst = sc.CustomNormConst(
        multConst=gamma0norm / 3,
        normNames=["density", "time", "velGrid"],
        normPowers=[1.0, 1.0, -3.0],
    )
    varDataI = sc.VarData(reqMBRowVars=["logLee", "I2"])
    varDataJ = sc.VarData(reqMBRowVars=["logLee", "J-1"])
    for harmonic in evolvedHarmonics:
        adfdvAtZero = [0, 0] if lNums[harmonic - 1] > 1 else [1.0 / vGrid[1], 0]
        termI = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            customNormConst=normConst,
            varData=varDataI,
            stencilData=sc.velDiffusionStencil(
                harmonic, harmonic, adfAtZero=cast(List[float], adfdvAtZero)
            ),
        )
        newModel.addTerm("diffTermI" + str(harmonic), termI)

        termJ = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            customNormConst=normConst,
            varData=varDataJ,
            stencilData=sc.velDiffusionStencil(
                harmonic, harmonic, adfAtZero=cast(List[float], adfdvAtZero)
            ),
        )
        newModel.addTerm("diffTermJ" + str(harmonic), termJ)

    # velocity deriv terms with fl

    vProfile = [1 / v**2 for v in vGrid]
    normConst = sc.CustomNormConst(
        multConst=gamma0norm / 3,
        normNames=["density", "time", "velGrid"],
        normPowers=[1.0, 1.0, -3.0],
    )
    varData = sc.VarData(reqMBRowVars=["logLee", "IJSum"])

    for harmonic in evolvedHarmonics:
        termdfldv = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            customNormConst=normConst,
            varData=varData,
            stencilData=sc.ddvStencil(harmonic, harmonic),
        )
        newModel.addTerm("dfdv" + str(harmonic), termdfldv)

    # -l(l+1)/2 terms
    hProfile = []
    for l in lNums:
        hProfile.append(l * (l + 1.0) / 2.0)

    vProfile = [1 / v**3 for v in vGrid]
    normConst = sc.CustomNormConst(
        multConst=-gamma0norm / 3,
        normNames=["density", "time", "velGrid"],
        normPowers=[1.0, 1.0, -3.0],
    )
    varData = sc.VarData(reqMBRowVars=["logLee", "IJSum"])

    termll = sc.GeneralMatrixTerm(
        distFunName,
        customNormConst=normConst,
        varData=varData,
        velocityProfile=vProfile,
        harmonicProfile=hProfile,
        stencilData=sc.diagonalStencil(evolvedHarmonics=evolvedHarmonics),
    )
    newModel.addTerm("termLL", termll)

    # I/J(f_l) terms

    for harmonic in evolvedHarmonics:
        l = lNums[harmonic - 1]

        # d2f0/dv2 terms
        vProfile = [1 / v for v in vGrid]
        varData = sc.VarData(reqMBRowVars=["logLee", "ddf0"])
        # C1*I_{l+2} term
        C = (l + 1) * (l + 2) / ((2 * l + 1) * (2 * l + 3))  # C1
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm / 2,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )

        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.ijIntegralStencil(harmonic, harmonic, l + 2),
        )

        newModel.addTerm("C1Il+2_h=" + str(harmonic), newTerm)

        # C1*J_{-l-1} term
        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.ijIntegralStencil(
                harmonic, harmonic, -l - 1, isJIntegral=True
            ),
        )

        newModel.addTerm("C1J-l-1_h=" + str(harmonic), newTerm)

        # C2*I_l term
        C = -(l - 1) * l / ((2 * l + 1) * (2 * l - 1))  # C2
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm / 2,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )

        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.ijIntegralStencil(harmonic, harmonic, l),
            skipPattern=True,
        )

        newModel.addTerm("C2Il_h=" + str(harmonic), newTerm)

        # C2*J1-l term
        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.ijIntegralStencil(
                harmonic, harmonic, 1 - l, isJIntegral=True
            ),
            skipPattern=True,
        )

        newModel.addTerm("C2J1-l_h=" + str(harmonic), newTerm)

        # df0/dv terms

        vProfile = [1 / v**2 for v in vGrid]
        varData = sc.VarData(reqMBRowVars=["logLee", "df0"])

        # C3*I_{l+2} term
        C = -((l + 1) * l / 2 + l + 1) / ((2 * l + 1) * (2 * l + 3))  # C3
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )

        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.ijIntegralStencil(harmonic, harmonic, l + 2),
            skipPattern=True,
        )

        newModel.addTerm("C3Il+2_h=" + str(harmonic), newTerm)

        # C4*J_{-l-1} term
        C = (-(l + 1) * l / 2 + l + 2) / ((2 * l + 1) * (2 * l + 3))  # C4
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )
        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.ijIntegralStencil(
                harmonic, harmonic, -l - 1, isJIntegral=True
            ),
            skipPattern=True,
        )

        newModel.addTerm("C4J-l-1_h=" + str(harmonic), newTerm)

        # C5*I_l term
        C = ((l + 1) * l / 2 + l - 1) / ((2 * l + 1) * (2 * l - 1))  # C5
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )

        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.ijIntegralStencil(harmonic, harmonic, l),
            skipPattern=True,
        )

        newModel.addTerm("C5Il_h=" + str(harmonic), newTerm)

        # C6*J1-l term
        C = -((l + 1) * l / 2 - l) / ((2 * l + 1) * (2 * l - 1))  # C6
        normConst = sc.CustomNormConst(
            multConst=C * gamma0norm,
            normNames=["density", "time", "velGrid"],
            normPowers=[1.0, 1.0, -3.0],
        )
        newTerm = sc.GeneralMatrixTerm(
            distFunName,
            velocityProfile=vProfile,
            varData=varData,
            customNormConst=normConst,
            stencilData=sc.ijIntegralStencil(
                harmonic, harmonic, 1 - l, isJIntegral=True
            ),
            skipPattern=True,
        )

        newModel.addTerm("C6J1-l_h=" + str(harmonic), newTerm)

    return newModel


def addEECollHigherL(
    modelTag: str,
    distFunName: str,
    elTempVar: str,
    elDensVar: str,
    wrapper: RKWrapper,
    evolvedHarmonics: List[int],
    dualDistFun: Union[str, None] = None,
) -> None:
    """Add e-e collision model for l>0

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable
        elDensVar (str): Name of the electron density variable
        wrapper (RKWrapper): Wrapper to add the model to
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating odd and even harmonics on staggered grid)
        dualDistFun (Union[str,None], optional): Interpolated distribution function to be used instead the default for derivation of mdoelbound variables. Defaults to None.
    """

    newModel = eeCollHigherL(
        modelTag,
        distFunName,
        elTempVar,
        elDensVar,
        wrapper,
        evolvedHarmonics,
        dualDistFun,
    )
    wrapper.addModel(newModel.dict())


def ampereMaxwellKineticElTerm(
    distFunName: str, eFieldName: str
) -> sc.GeneralMatrixTerm:
    """Return kinetic electron contribution term to Ampere-Maxwell equation for E-field

    Args:
        distFunName (str): Name of distribution function variable
        eFieldName (str): Name of evolved E-field variable

    Returns:
        sc.GeneralMatrixTerm: Term object ready to be added into a model
    """

    elCharge = 1.60218e-19
    epsilon0 = 8.854188e-12  # vacuum permittivity

    normConst = sc.CustomNormConst(
        multConst=elCharge / (3 * epsilon0),
        normNames=["density", "time", "velGrid", "EField"],
        normPowers=[1.0, 1.0, 1.0, -1.0],
    )

    newTerm = sc.GeneralMatrixTerm(
        eFieldName,
        implicitVar=distFunName,
        customNormConst=normConst,
        stencilData=sc.momentStencil(1, 2),
    )

    return newTerm


def diffusiveHeatingTerm(
    distFunName: str,
    densityName: str,
    heatingProfile: List[float],
    wrapper: RKWrapper,
    timeSignal=sc.TimeSignalData(),
) -> sc.GeneralMatrixTerm:
    """Return diffusive kinetic electron heating with a given spatial heating profile

    Args:
        distFunName (str): Name of distribution function variable
        densityName (str): Name of electron density variable
        heatingProfile (List[float]): Heating profile in normalized units of density*temperature/time
        wrapper (RKWrapper): Wrapper used to retrieve velocity grid info
        timeSignal (sc.TimeSignalData): Optional time signal component of heating. Defaults to constant signal.

    Returns:
        sc.GeneralMatrixTerm: Term object ready to be added into a model
    """

    elCharge = 1.60218e-19
    elMass = 9.10938e-31

    vGrid = wrapper.grid.vGrid
    dv = wrapper.grid.vWidths
    vBoundary = [vGrid[i] + dv[i] / 2 for i in range(len(dv))]

    vProfile = [1 / v**2 for v in vGrid]
    varData = sc.VarData(reqRowVars=[densityName], reqRowPowers=[-1.0])

    normConst = sc.CustomNormConst(
        multConst=elCharge / (3 * elMass),
        normNames=["eVTemperature", "velGrid"],
        normPowers=[1.0, -2.0],
    )

    newTerm = sc.GeneralMatrixTerm(
        distFunName,
        customNormConst=normConst,
        spatialProfile=heatingProfile,
        velocityProfile=vProfile,
        varData=varData,
        stencilData=sc.velDiffusionStencil(1, 1, [v**2 for v in vBoundary]),
        timeSignalData=timeSignal,
    )

    return newTerm


def lbcModel(
    modelTag: str,
    distFunName: str,
    wrapper: RKWrapper,
    distExtRule: dict,
    ionCurrentVar: str,
    totalCurrentVar: str = "none",
    bisTol: float = 1.0e-12,
    leftBoundary=False,
    evolvedHarmonics: List[int] = [],
) -> sc.CustomModel:
    """Return logical boundary condition model

    Args:
        modelTag (str): Model tag of added model
        distFunName (str): Name of electron distribution function
        wrapper (RKWrapper): Wrapper to use
        distExtRule (dict): Distribution extrapolation rule
        ionCurrentVar (str): Name of ion current variable at boundary
        totalCurrentVar (str, optional): Name of total current variable at boundary. Defaults to "none", resulting in 0 total current.
        bisTol (float, optional): Bisection tolerance for calculating cut-off. Defaults to 1.0e-12.
        leftBoundary (bool, optional): True if model describes left boundary. Defaults to False.
        evolvedHarmonics (List[int], optional): List of evolved harmonic. Defaults to [], evolving all harmonics. Should be changed to only even l harmonics if staggered grid is used.
    """

    assert wrapper.grid.mMax == 0, "lbcModel assumes m=0 and cannot be used when m>0"

    mbData = sc.modelboundLBCData(
        ionCurrentVar, distExtRule, totalCurrentVar, bisTol, leftBoundary
    )

    newModel = sc.CustomModel(modelTag=modelTag)
    newModel.setModelboundData(mbData)

    usedHarmonics = evolvedHarmonics
    if len(usedHarmonics) == 0:
        usedHarmonics = list(range(1, wrapper.grid.numH() + 1))

    lGrid = wrapper.grid.lGrid

    oddL = [l % 2 == 1 for l in lGrid]
    oddLHarmonics = [i + 1 for i, x in enumerate(oddL) if x]
    evenLHarmonics = [i + 1 for i, x in enumerate(oddL) if not x]

    for h in usedHarmonics:
        l = lGrid[h - 1]

        # l-1 harmonic

        if l > 0:
            if leftBoundary:
                normConst = sc.CustomNormConst(multConst=l / ((2 * l - 1)))

                newTerm = sc.GeneralMatrixTerm(
                    distFunName,
                    customNormConst=normConst,
                    stencilData=sc.scalingLBCStencil(
                        h,
                        wrapper.grid.getH(l - 1, 0),
                        distExtRule,
                        leftBoundary=leftBoundary,
                    ),
                )

                newModel.addTerm("lbcMinus" + str(h), newTerm)
            else:
                normConst = sc.CustomNormConst(multConst=-l / ((2 * l - 1)))

                newTerm = sc.GeneralMatrixTerm(
                    distFunName,
                    customNormConst=normConst,
                    stencilData=sc.scalingLBCStencil(
                        h,
                        wrapper.grid.getH(l - 1, 0),
                        distExtRule,
                        leftBoundary=leftBoundary,
                        decompHarmonics=cast(List, oddLHarmonics),
                    ),
                )

                newModel.addTerm("lbcMinus_odd" + str(h), newTerm)

                newTerm = sc.GeneralMatrixTerm(
                    distFunName,
                    customNormConst=normConst,
                    stencilData=sc.scalingLBCStencil(
                        h,
                        wrapper.grid.getH(l - 1, 0),
                        distExtRule,
                        leftBoundary=leftBoundary,
                        decompHarmonics=cast(List[float], evenLHarmonics),
                    ),
                )

                newModel.addTerm("lbcMinus_even" + str(h), newTerm)

        # l+1 harmonic
        if l < max(lGrid):
            if leftBoundary:
                normConst = sc.CustomNormConst(multConst=(l + 1) / ((2 * l + 3)))

                newTerm = sc.GeneralMatrixTerm(
                    distFunName,
                    customNormConst=normConst,
                    stencilData=sc.scalingLBCStencil(
                        h,
                        wrapper.grid.getH(l + 1, 0),
                        distExtRule,
                        leftBoundary=leftBoundary,
                    ),
                )

                newModel.addTerm("lbcPlus" + str(h), newTerm)
            else:
                normConst = sc.CustomNormConst(multConst=-(l + 1) / ((2 * l + 3)))

                newTerm = sc.GeneralMatrixTerm(
                    distFunName,
                    customNormConst=normConst,
                    stencilData=sc.scalingLBCStencil(
                        h,
                        wrapper.grid.getH(l + 1, 0),
                        distExtRule,
                        leftBoundary=leftBoundary,
                        decompHarmonics=cast(List[float], oddLHarmonics),
                    ),
                )

                newModel.addTerm("lbcPlus_odd" + str(h), newTerm)

                newTerm = sc.GeneralMatrixTerm(
                    distFunName,
                    customNormConst=normConst,
                    stencilData=sc.scalingLBCStencil(
                        h,
                        wrapper.grid.getH(l + 1, 0),
                        distExtRule,
                        leftBoundary=leftBoundary,
                        decompHarmonics=cast(List, evenLHarmonics),
                    ),
                )

                newModel.addTerm("lbcPlus_even" + str(h), newTerm)

    return newModel


def addLBCModel(
    modelTag: str,
    distFunName: str,
    wrapper: RKWrapper,
    distExtRule: dict,
    ionCurrentVar: str,
    totalCurrentVar: str = "none",
    bisTol: float = 1.0e-12,
    leftBoundary=False,
    evolvedHarmonics: List[int] = [],
) -> None:
    """Add logical boundary condition model to wrapper

    Args:
        modelTag (str): Model tag of added model
        distFunName (str): Name of electron distribution function
        wrapper (RKWrapper): Wrapper to add the model to
        distExtRule (dict): Distribution extrapolation rule
        ionCurrentVar (str): Name of ion current variable at boundary
        totalCurrentVar (str, optional): Name of total current variable at boundary. Defaults to "none", resulting in 0 total current.
        bisTol (float, optional): Bisection tolerance for calculating cut-off. Defaults to 1.0e-12.
        leftBoundary (bool, optional): True if model describes left boundary. Defaults to False.
        evolvedHarmonics (List[int], optional): List of evolved harmonic. Defaults to [], evolving all harmonics. Should be changed to only even l harmonics if staggered grid is used.
    """

    newModel = lbcModel(
        modelTag,
        distFunName,
        wrapper,
        distExtRule,
        ionCurrentVar,
        totalCurrentVar,
        bisTol,
        leftBoundary,
        evolvedHarmonics,
    )

    wrapper.addModel(newModel.dict())


def dvEnergyTerm(
    distFunName: str, varData: sc.VarData, wrapper: RKWrapper, multConst: float = -1.0
) -> sc.GeneralMatrixTerm:
    """Return velocity space drag-like heating/cooling term

    Args:
        distFunName (str): Name of distribution function variable
        varData (sc.VarData): Required variable data used to customize the rate (should result in something normalized to temperature/time,
                              assuming velocity is normalized to thermal velocity)
        wrapper (RKWrapper): Wrapper used to retrieve velocity grid info
        multConst (float, optional): Multiplicative constant for the normalization. Defaults to -1.0, which assumes that a positive rate is heating
    Returns:
        sc.GeneralMatrixTerm: Term object ready to be added into a model
    """

    vGrid = wrapper.grid.vGrid
    dv = wrapper.grid.vWidths
    interp = np.zeros(len(dv)).tolist()

    vProfile = [1 / (v**2) for v in vGrid]

    drag = vGrid**2 * dv
    vSum = [
        0 for i in range(len(drag))
    ]  # vGrid**2 if exact energy source is required, 0 if exactly no particle source is required (either way the error is negligible)
    vSum[:-1] = vGrid[1:] ** 2 - vGrid[:-1] ** 2
    drag = drag / vSum
    normConst = sc.CustomNormConst(multConst=multConst)

    newTerm = sc.GeneralMatrixTerm(
        distFunName,
        customNormConst=normConst,
        velocityProfile=vProfile,
        varData=varData,
        stencilData=sc.ddvStencil(1, 1, drag.tolist(), interp),
    )

    return newTerm
