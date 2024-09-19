from . import simple_containers as sc
from typing import Union, List, Tuple, cast
import numpy as np
from .grid import Grid
from .rk_wrapper import RKWrapper
from . import calculation_tree_support as ct
from .sk_normalization import calculateNorms


def collocatedAdvection(
    modelTag: str,
    advectedVar: str,
    advectionSpeed: str,
    lowerBoundVar: Union[str, None] = None,
    leftOutflow=False,
    rightOutflow=False,
    centralDiff=False,
) -> sc.CustomModel:
    """Create a collocated advection model (dn/dt = - div(n*u)) using matrix terms. Defaults to an upwinding stencil, but can be set to use central differencing, where the variables are interpolated to the cell edges.

    Args:
        modelTag (str): Model tag
        advectedVar (str): Name of advected variable (n in above equation), must be implicit
        advectionSpeed (str): Name of advection speed variable (u in above equation)
        lowerBoundVar (Union[str,None], optional): Name of outflow lower bound variable, can be None, which defaults to a fixed lower bound of zero. Defaults to None.
        leftOutflow (bool, optional): If true left boundary is treated as outflow, otherwise it is reflective or periodic depending on the grid. Defaults to False.
        rightOutflow (bool, optional): If true right boundary is treated as outflow, otherwise it is reflective or periodic depending on the grid. Defaults to False.
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
    """Create a collocated pressure gradient model for the momentum equation using matrix terms.

    Args:
        modelTag (str): Model tag
        fluxVar (str): Evolved flux variable name - must be implicit
        densityVar (str): Species density name factoring into pressure - this will be the implicit variable in the matrix term
        temperatureVar (str): Species temperature name factoring into pressure
        speciesMass (float): Species mass in kg
        addExtrapolatedBCs (bool, optional): If true will add extrapolated values as left and right boundary condition (set to False for periodic grids). Defaults to True

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
    """Simple implicit source term with given source profile using matrix terms

    Args:
        evolvedVar (str): Name of evolved variable - must be implicit
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
    """Create a staggered grid advection model (dn/dt = -div(G)) using matrix terms. Can handle advection of both cell centre and cell edge (dual) variables. When advecting cell centre variables, G is expected to live on cell edges, and when advecting cell edge variables G can be calculated as n*u, where u is the advection speed on cell edges, and the divergence is no longer staggered, but centered.

    Args:
        modelTag (str): Model tag
        advectedVar (str): Name of advected (evolved) variable.
        fluxVar (str): Name of flux variable - always on dual grid. Not used when staggeredAdvectedVar is True.
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
    """Create a staggered grid pressure gradient model (m*dG/dt = - grad(p)) for the momentum equation, assuming the flux is on the dual grid and using matrix terms. p is assumed to be n*k*T and normalization used is the default SOL-KiT-like normalization.

    Args:
        modelTag (str): Model tag
        fluxVar (str): Evolved flux variable name (G in the above equation)
        densityVar (str): Species density name factoring into pressure (n in the above equation)
        temperatureVar (str): Species temperature name factoring into pressure (T in the above equation)
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
    """Generate dE/dt = -j/epsilon0 matrix terms by calculating currents based on species fluxes and charges. Assumes default normalization.

    Args:
        modelTag (str): Model tag
        eFieldName (str): Name of evolved electric field variable
        speciesFluxes (List[str]): Names of species fluxes - implicit variables
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
    """Generate Lorentz force matrix terms in the momentum equation for each species b: m_b*dG_b/dt = n_b*Z_b*e*E. Assumes default normalization.

    Args:
        modelTag (str): Model tag
        eFieldName (str): Name of  electric field variable (E in above equation) - this will be the implicit variable
        speciesFluxes (List[str]): Names of evolved species fluxes (G_b in above equation)- the evolved implicit variable
        speciesDensities (List[str]): Names of species densities (n_b in above equation) - should live on the same grid as the electric field and fluxes
        species (list[sc.Species]): Species objects for each species (supply m_b and Z_b in above equation)

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
    """Generate Lorentz force work matrix terms in the energy density evolution equation for each species: dW_b/dt = Z_b * e * G_b * E. Assumes default normalization.

    Args:
        modelTag (str): Model tag
        eFieldName (str): Name of  electric field variable (E in above equation) - this will be the implicit variable
        speciesFluxes (List[str]): Names of evolved species fluxes (G_b in above equation) - these should live on the same grid as the implicit electric field
        speciesEnergies (List[str]): Names of species energies (W_b in above equation)- the evolved variables
        species (list[sc.Species]): Species objects for each species (th)

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

        evolvedVar = speciesEnergies[i]

        workTerm = sc.GeneralMatrixTerm(
            evolvedVar,
            implicitVar=implicitVar,
            customNormConst=normConst,
            stencilData=sc.diagonalStencil(),
            varData=vData,
        )

        newModel.addTerm("lorentzWork" + flux, workTerm)

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
    degreesOfFreedom: int = 3,
) -> sc.CustomModel:
    """Generate implicit temperature derivation matrix terms for each species: d*n_b*k*T_b/2 + m_b*n_b*u_b**2/2 = W_b, where d is the number of degrees of freedom. Temperatures here are assumed to be stationary and implicit variables. The kinetic energy contribution uses interpolation, so should be used with care in regions of poorly resolved flow gradients. Assumes default normalization.

    Args:
        modelTag (str): Model tag
        speciesFluxes (List[str]): Names of species fluxes (should be implicit)
        speciesEnergies (List[str]): Names of species energies (should be implicit)
        speciesDensities (List[str]): Names of species densities (n_b in the above equation, should be on the same grid as the energies)
        speciesTemperatures (List[str]): Names of species temperature - these will be the evolved variables and should be stationary
        species (list[sc.Species]): Species objects for each species (used to get species masses)
        speciesDensitiesDual (Union[List[str],None], optional): Names of species densities on dual grid (use when fluxes are staggered). Defaults to None.
        evolvedXU2Cells (Union[List[int],None], optional): Optional list of evolved X cells in kinetic energy term. Can be used to remove the kinetic energy contribution in cells where it might be unphysical. Defaults to None, evolving all cells.
        ignoreKineticContribution (bool, optional): Ignores all kinetic contributions to the temperature. Defaults to False.
        degreesOfFreedom (int): Number of translational degrees of freedom going into temperature definition (d in above equation). Defaults to 3
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
    normConstW = sc.CustomNormConst(multConst=2 / degreesOfFreedom)

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
    """Return kinetic advection model for electrons in x direction using matrix terms. This is the harmonic form of the vdf/dx term, coupling adjacent harmonics. Assumes default normalization.

    Args:
        modelTag (str): Tag for the generated model
        distFunName (str): Name of the advected distribution function - both the evolved and implicit variable
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
    """Returns electric field advection matrix terms for electrons in x direction: the harmonic decomposition of E/m*df/dv. Adds necessary custom derivations to the wrapper. The electric field is taken to be implicit, so it will be interpolated implicitly onto cell centres for even l harmonics.

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
    """Adds electric field advection matrix terms for electrons in x direction: the harmonic decomposition of E/m*df/dv. Adds necessary custom derivations to the wrapper. The electric field is taken to be implicit, so it will be interpolated implicitly onto cell centres for even l harmonics.

    Args:
        modelTag (str): Tag for the generated model
        distFunName (str): Name of the evolved distribution function
        eFieldName (str): Name of the implicit E-field variable
        wrapper (RKWrapper): Wrapper object to add the model and add custom derivations to
        dualDistFun (Union[str,None], optional): Optional distribution function to be used as the required variable in derivations (useful for staggered grids). Defaults to None.
    """
    newModel = advectionEx(modelTag, distFunName, eFieldName, wrapper, dualDistFun)

    wrapper.addModel(newModel)


def eeCollIsotropic(
    modelTag: str, distFunName: str, elTempVar: str, elDensVar: str, wrapper: RKWrapper
) -> sc.CustomModel:
    """Return e-e collision model for l=0 harmonic using matrix terms. Responsible for adding the derivations needed for its modelbound data. Assumes default normalization.

    Args:
        modelTag (str): Tag of the model to be added
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable - should live in cell centres
        elDensVar (str): Name of the electron density variable - should live in cell centres
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
    """Add e-e collision matrix terms for l=0 harmonic. Responsible for adding derivations needed for the model's data. Assumes default normalization.

    Args:
        modelTag (str): Tag of the model to be added
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable - should live in cell centres
        elDensVar (str): Name of the electron density variable - should live in cell centres
        wrapper (RKWrapper): Wrapper to add the term to
    """

    newModel = eeCollIsotropic(modelTag, distFunName, elTempVar, elDensVar, wrapper)

    wrapper.addModel(newModel)


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
    """Return e-i collision model for l=0 harmonic using matrix terms. Responsible for adding derivations used in its modelbound data. Assumes default normalization.

    Args:
        modelTag (str): Model tag
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable - should live in cell centres
        elDensVar (str): Name of the electron density variable - should live in cell centres
        ionTempVar (str): Name of the ion temperature variable - should live in cell centres
        ionDensVar (str): Name of the ion density variable - should live in cell centres
        ionSpeciesName (str): Name of ion species colliding with the electrons - needed for the Coulomb logarithm
        ionEnVar (Union[str,None]): Name of the ion energy density variable (must be implicit). If present, the energy moment of the collision operator will be added to the equation evolving this variable. Defaults to None.
        wrapper (RKWrapper): Wrapper to add custom derivations to
    """

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity
    amu = 1.6605390666e-27  # atomic mass unit
    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)

    ionSpecies = wrapper.getSpecies(ionSpeciesName)
    gamma0norm = gamma0norm * ionSpecies.charge**2 * elMass / (ionSpecies.atomicA * amu)

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
    """Add e-i collision matrix terms for l=0 harmonic, as well as the corresponding ion energy matrix terms if ionEnVar is present. Responsible for adding derivations used in its modelbound data. Assumes default normalization.

    Args:
        modelTag (str): Model tag
        distFunName (str): Name of the electron distribution function variable
        elTempVar (str): Name of the electron temperature variable - should live in cell centres
        elDensVar (str): Name of the electron density variable - should live in cell centres
        ionTempVar (str): Name of the ion temperature variable - should live in cell centres
        ionDensVar (str): Name of the ion density variable - should live in cell centres
        ionSpeciesName (str): Name of ion species colliding with the electrons - needed for the Coulomb logarithm
        wrapper (RKWrapper): Wrapper to add the terms to
        ionEnVar (Union[str,None]): Name of the ion energy density variable (must be implicit). If present, the energy moment of the collision operator will be added to the equation evolving this variable. Defaults to None.
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

    wrapper.addModel(newModel)


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
    """Return stationary ion electron-ion collision operator model using matrix terms. Assumes default normalization.

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str): Name of electron distribution function variable - this is the evolved variable
        ionDensVar (str): Name of ion density variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        electronDensVar (str): Name of electron density variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        electronTempVar (str): Name of electron temperature variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        ionSpeciesName (str): Name of ion species colliding with the electrons
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating even and odd harmonics on staggered grid) - NOTE: Use Fortran indexing
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
    """Return a flowing cold ion electron-ion collision model using matrix terms. Assumes default normalization.

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str):  Name of the electron distribution function variable - this is the evolved variable
        ionDensVar (str): Name of the implicit ion density variable
        ionFlowSpeedVar (str): Name of ion flow speed variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        electronDensVar (str): Name of electron density variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        electronTempVar (str): Name of electron temperature variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        ionSpeciesName (str): Name of the ion species
        wrapper (RKWrapper): Wrapper to add custom derivations to
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating even and odd harmonics on staggered grid) - NOTE: Use Fortran indexing
        dualDistFun (Union[str,None], optional): Interpolated distribution function to be used instead the default for derivation of modelbound variables - use when using staggered grids. Defaults to None.
        ionFluxVar (Union[str,None], optional): Ion flux variable (should be implicit) - when present generates ion friction terms corresponding to the electron-ion collision operators. Defaults to None.
    """
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
    """Add stationary ion electron-ion collision operator model to wrapper using matrix terms. Assumes default normalization.

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str): Name of electron distribution function variable - this is the evolved variable
        ionDensVar (str): Name of ion density variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        electronDensVar (str): Name of electron density variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        electronTempVar (str): Name of electron temperature variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        ionSpeciesName (str): Name of ion species colliding with the electrons
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating even and odd harmonics on staggered grid) - NOTE: Use Fortran indexing
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

    wrapper.addModel(newModel)


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
    """Add a flowing cold ion electron-ion collision model using matrix terms. Assumes default normalization.

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str):  Name of the electron distribution function variable - this is the evolved variable
        ionDensVar (str): Name of ion density variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        ionFlowSpeedVar (str): Name of ion flow speed variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        electronDensVar (str): Name of electron density variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        electronTempVar (str): Name of electron temperature variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        ionSpeciesName (str): Name of the ion species
        wrapper (RKWrapper): Wrapper to add model to
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating even and odd harmonics on staggered grid) - NOTE: Use Fortran indexing
        dualDistFun (Union[str,None], optional): Interpolated distribution function to be used instead the default for derivation of modelbound variables - use when using staggered grids. Defaults to None.
        ionFluxVar (Union[str,None], optional): Ion flux variable (should be implicit) - when present generates ion friction terms corresponding to the electron-ion collision operators. Defaults to None.
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

    wrapper.addModel(newModel)


def eeCollHigherL(
    modelTag: str,
    distFunName: str,
    elTempVar: str,
    elDensVar: str,
    wrapper: RKWrapper,
    evolvedHarmonics: List[int],
    dualDistFun: Union[str, None] = None,
) -> sc.CustomModel:
    """Return e-e collision model for l>0 using matrix terms. Assumes default normalization.

    Args:
        modelTag (str): Tag of created model
        distFunName (str): Name of the electron distribution function variable - this is the evolved variable
        elTempVar (str): Name of the electron temperature variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        elDensVar (str): Name of the electron density variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        wrapper (RKWrapper): Wrapper to add derivations to
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating even and odd harmonics on staggered grid) - NOTE: Use Fortran indexing
        dualDistFun (Union[str,None], optional): Interpolated distribution function to be used instead the default for derivation of modelbound variables - use when using staggered grids. Defaults to None.
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
    """Add e-e collision model for l>0 using matrix terms. Assumes default normalization.

    Args:
        modelTag (str): Tag of model to be added
        distFunName (str): Name of the electron distribution function variable - this is the evolved variable
        elTempVar (str): Name of the electron temperature variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        elDensVar (str): Name of the electron density variable - should live on the same grid as the evolved harmonics - regular if even l dual if odd
        wrapper (RKWrapper): Wrapper to add derivations to
        evolvedHarmonics (List[int]): List of evolved harmonics (useful when separating even and odd harmonics on staggered grid) - NOTE: Use Fortran indexing
        dualDistFun (Union[str,None], optional): Interpolated distribution function to be used instead the default for derivation of modelbound variables - use when using staggered grids. Defaults to None.
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
    wrapper.addModel(newModel)


def ampereMaxwellKineticElTerm(
    distFunName: str, eFieldName: str
) -> sc.GeneralMatrixTerm:
    """Return kinetic electron contribution matrix term to Ampere-Maxwell equation for E-field. This uses a moment stencil, taking the appropriate moment of the l=1 harmonic. Assumes default normalization.

    Args:
        distFunName (str): Name of distribution function variable - this is the implicit variable
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
    """Return diffusive kinetic electron heating matrix term with a given spatial heating profile. The term is proportional to 1/v**2 * d/dv(v**2 * df_0/dv) Assumes default normalization.

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
    """Return logical boundary condition model using matrix terms. Assumes default normalization. Extrapolates the distribution function to the boundary using a supplied extrapolation rule, and allows for a non-zero current through the sheath.

    Args:
        modelTag (str): Model tag of added model
        distFunName (str): Name of electron distribution function - this is the evolved variable
        wrapper (RKWrapper): Wrapper to use
        distExtRule (dict): Distribution extrapolation derivation rule - see for example distScalingExtrapolationDerivation in simple_containers.py
        ionCurrentVar (str): Name of ion current variable at boundary - scalar variable
        totalCurrentVar (str, optional): Name of total current variable at boundary - scalar variable. Defaults to "none", resulting in 0 total current.
        bisTol (float, optional): Bisection tolerance for calculating the cut-off velocity in the electron distribution function. Defaults to 1.0e-12.
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
    """Add logical boundary condition model built with matrix terms to wrapper. Assumes default normalization. Extrapolates the distribution function to the boundary using a supplied extrapolation rule, and allows for a non-zero current through the sheath.

    Args:
        modelTag (str): Model tag of added model
        distFunName (str): Name of electron distribution function - this is the evolved variable
        wrapper (RKWrapper): Wrapper to add the model to
        distExtRule (dict): Distribution extrapolation derivation rule - see for example distScalingExtrapolationDerivation in simple_containers.py
        ionCurrentVar (str): Name of ion current variable at boundary - scalar variable
        totalCurrentVar (str, optional): Name of total current variable at boundary - scalar variable. Defaults to "none", resulting in 0 total current.
        bisTol (float, optional): Bisection tolerance for calculating the cut-off velocity in the electron distribution function. Defaults to 1.0e-12.
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

    wrapper.addModel(newModel)


def dvEnergyTerm(
    distFunName: str,
    varData: sc.VarData,
    wrapper: RKWrapper,
    multConst: float = -1.0,
    k: int = 0,
    implicitGroups=[1],
) -> sc.GeneralMatrixTerm:
    """Return velocity space drag-like heating/cooling matrix term: proportional to 1/v**2 * d/dv(Df) where D is a velocity space vector proportional to v**k * dv, with k set by the user, controlling which velocity cells get the bulk of the energy source. Assumes default normalization.

    Args:
        distFunName (str): Name of distribution function variable - both the evolved and implicit variable
        varData (sc.VarData): Required variable data used to customize the rate (should result in something normalized to temperature/time, assuming velocity is normalized to thermal velocity)
        wrapper (RKWrapper): Wrapper used to retrieve velocity grid info
        multConst (float, optional): Multiplicative constant for the normalization. Defaults to -1.0, which assumes that a positive rate is heating
        k (int, optional): Optional power for the drag coefficient (effectively multiplies by v**k). If not 0 varData should include the electron density divided by the k-th moment of f_0 (in the moment derivation sense). Defaults to 0.
        implicitGroups (list, optional): Implicit term groups of parent model to which this term belongs to. Defaults to [1].
    Returns:
        sc.GeneralMatrixTerm: Term object ready to be added into a model
    """

    vGrid = wrapper.grid.vGrid
    dv = wrapper.grid.vWidths
    interp = np.zeros(len(dv)).tolist()

    vProfile = [1 / (v**2) for v in vGrid]

    drag = dv * np.ones(len(vGrid))
    vSum = vGrid**k * np.zeros(
        len(drag)
    )  # ones if exact energy source is required, 0 if exactly no particle source is required (either way the error is negligible)
    vSum[:-1] = vGrid[:-1] ** (2 + k) / (vGrid[1:] ** 2 - vGrid[:-1] ** 2)
    drag = drag * vSum
    normConst = sc.CustomNormConst(multConst=multConst)

    newTerm = sc.GeneralMatrixTerm(
        distFunName,
        implicitGroups=implicitGroups,
        customNormConst=normConst,
        velocityProfile=vProfile,
        varData=varData,
        stencilData=sc.ddvStencil(1, 1, drag.tolist(), interp),
    )

    return newTerm


def standardBaseFluid(
    speciesName: str,
    densName: str,
    fluxName: str,
    flowSpeedName: str,
    energyName: str,
    tempName: str,
    electricFieldName: str,
    speciesMass: float,
    speciesCharge: float,
    heatfluxVar: Union[str, None] = None,
    viscVar: Union[str, None] = None,
    viscosityLimitMultName: Union[str, None] = None,
) -> sc.CustomModel:
    """Generates a standard base fluid model for a species with given variable names (on regular grid, assumes staggered grid is used).

    The model will include the continuity equation, momentum equation, and energy equation, with the default reflective boundary conditions (should be specified in a separate model).

    The implicit temperature calculation is also added.

    No heat flux, viscosity, or sources are added.

    If the heatflux variable is present it will be assumed to be stationary and will have its identity term and the corresponding bulk divergence terms in the energy equation added.

    If the viscosity variable is present it will be assumed to be stationary and will have its identity term and the corresponding bulk divergence terms in the momentum and energy equation added.

    Args:
        speciesName (str): Name of the species. Used to name the model
        densName (str): Name of the density variable associated to the species
        fluxName (str): Name of the flux variable associated to the species
        flowSpeedName (str): Name of the flow speed variable associated to the species
        energyName (str): Name of the energy variable associated to the species
        tempName (str): Name of the temperature variable associated with the species
        electricFieldName (str): Name of the implicit electric field variable
        speciesMass (float): Species mass in kg
        speciesCharge (float): Species charge in e
        heatfluxVar (Union[str,None], optional): Name of the stationary heat flux variable associated with the species. Defaults to None, not building any related terms.
        viscVar (Union[str,None], optional): Name of the stationary viscosity variable (unlimited) associated with the species. Defaults to None, not building any related terms.
        viscosityLimitMultName (Union[str,None], optional): Viscosity limitation multiplier (applied directly in relevant terms). Defaults to None, removing the limit.

    Returns:
        sc.CustomModel: Model object holding the base fluid terms for a given species
    """

    newModel = sc.CustomModel(modelTag="fluidBase_" + speciesName + "_")

    elMass = 9.10938e-31
    massRatio = elMass / speciesMass

    # Continuity equation flux divergence
    contDivFluxTerm = sc.GeneralMatrixTerm(
        densName,
        fluxName + "_dual",
        customNormConst=-1,
        stencilData=sc.staggeredDivStencil(),
    )

    newModel.addTerm("cont_divFlux", contDivFluxTerm)

    # Momentum equation pressure gradient

    vDataPressure = sc.VarData(reqColVars=[tempName])

    pressureFradTerm = sc.GeneralMatrixTerm(
        fluxName + "_dual",
        implicitVar=densName,
        customNormConst=-massRatio / 2,
        stencilData=sc.staggeredGradStencil(),
        varData=vDataPressure,
    )

    newModel.addTerm("momentum_gradPressure", pressureFradTerm)

    # Momentum advection

    momentumDivFluxTerm = sc.GeneralMatrixTerm(
        fluxName + "_dual",
        fluxName + "_dual",
        customNormConst=-1,
        stencilData=sc.centralDiffStencilDiv(cast(str, flowSpeedName + "_dual")),
    )

    newModel.addTerm("momentum_divFlux", momentumDivFluxTerm)

    # Lorentz force
    if abs(speciesCharge) > 1e-6:  # Species effectively neutral
        vDataLorentzForce = sc.VarData(reqRowVars=[densName + "_dual"])

        lorentzForceTerm = sc.GeneralMatrixTerm(
            fluxName + "_dual",
            implicitVar=electricFieldName + "_dual",
            customNormConst=speciesCharge * massRatio,
            stencilData=sc.diagonalStencil(),
            varData=vDataLorentzForce,
        )

        newModel.addTerm("momentum_lorentzForce", lorentzForceTerm)

    # Implicit temperature calculation

    # Identity term

    identityTermT = sc.GeneralMatrixTerm(
        tempName, customNormConst=-1, stencilData=sc.diagonalStencil()
    )

    newModel.addTerm("temperature_identity", identityTermT)

    # 2/3 W/n term

    vDataW = sc.VarData(reqRowVars=[densName], reqRowPowers=[-1.0])

    termW = sc.GeneralMatrixTerm(
        tempName,
        implicitVar=energyName,
        customNormConst=2 / 3,
        varData=vDataW,
        stencilData=sc.diagonalStencil(),
    )

    newModel.addTerm("temperature_w", termW)

    # kinetic energy term

    vDataU2 = sc.VarData(
        reqColVars=[fluxName + "_dual", densName + "_dual"], reqColPowers=[1.0, -2.0]
    )

    termU2 = sc.GeneralMatrixTerm(
        tempName,
        implicitVar=fluxName + "_dual",
        customNormConst=-2 / (3 * massRatio),
        varData=vDataU2,
        stencilData=sc.diagonalStencil(),
    )

    newModel.addTerm("temperature_U2", termU2)

    # Energy advection

    vDataWAdv = sc.VarData(
        reqColVars=[energyName + "_dual", densName + "_dual"], reqColPowers=[1.0, -1.0]
    )

    wAdvFluxDiv = sc.GeneralMatrixTerm(
        energyName,
        fluxName + "_dual",
        customNormConst=-1,
        stencilData=sc.staggeredDivStencil(),
        varData=vDataWAdv,
    )

    newModel.addTerm("energy_wAdv", wAdvFluxDiv)

    # Pressure advection

    vDataPAdv = sc.VarData(reqColVars=[tempName + "_dual"])

    pAdvFluxDiv = sc.GeneralMatrixTerm(
        energyName,
        fluxName + "_dual",
        customNormConst=-1,
        stencilData=sc.staggeredDivStencil(),
        varData=vDataPAdv,
    )

    newModel.addTerm("energy_pAdv", pAdvFluxDiv)

    # Lorentz Force work
    if abs(speciesCharge) > 1e-6:  # Species effectively neutral
        vDataLorentzWork = sc.VarData(reqColVars=[fluxName + "_dual"])

        workTerm = sc.GeneralMatrixTerm(
            energyName,
            implicitVar=electricFieldName + "_dual",
            customNormConst=2 * speciesCharge,
            stencilData=sc.diagonalStencil(),
            varData=vDataLorentzWork,
        )

        newModel.addTerm("energy_lorentzWork", workTerm)

    # Heatflux terms

    if heatfluxVar is not None:
        # Identity term

        identityTermQ = sc.GeneralMatrixTerm(
            cast(str, heatfluxVar) + "_dual",
            customNormConst=-1,
            stencilData=sc.diagonalStencil(),
        )

        newModel.addTerm("heatflux_identity", identityTermQ)

        # Heatflux divergence in energy equation

        divq = sc.GeneralMatrixTerm(
            energyName,
            implicitVar=cast(str, heatfluxVar) + "_dual",
            customNormConst=-1,
            stencilData=sc.staggeredDivStencil(),
        )

        newModel.addTerm("energy_divq", divq)

    # Viscosity
    if viscVar is not None:
        # Identity term

        identityTermPI = sc.GeneralMatrixTerm(
            cast(str, viscVar), customNormConst=-1, stencilData=sc.diagonalStencil()
        )

        newModel.addTerm("viscosity_identity", identityTermPI)

        # Viscosity divergence in momentum equation

        vData = sc.VarData()
        if viscosityLimitMultName is not None:
            vData = sc.VarData(reqColVars=[viscosityLimitMultName])

        divpi = sc.GeneralMatrixTerm(
            fluxName + "_dual",
            implicitVar=viscVar,
            customNormConst=-massRatio / 2,
            stencilData=sc.staggeredDivStencil(),
            varData=vData,
        )

        newModel.addTerm("momentum_divpi", divpi)

        # Viscosity advection/heating in energy equation

        vData = sc.VarData(
            reqColVars=[viscVar + "_dual", densName + "_dual"], reqColPowers=[1.0, -1.0]
        )
        if viscosityLimitMultName is not None:
            vData = sc.VarData(
                reqColVars=[
                    viscosityLimitMultName + "_dual",
                    viscVar + "_dual",
                    densName + "_dual",
                ],
                reqColPowers=[1.0, 1.0, -1.0],
            )

        divpiu = sc.GeneralMatrixTerm(
            energyName,
            implicitVar=fluxName + "_dual",
            customNormConst=-1,
            stencilData=sc.staggeredDivStencil(),
            varData=vData,
        )

        newModel.addTerm("energy_divpiu", divpiu)

    return newModel


def bohmBoundaryModel(
    speciesName: str,
    densityName: str,
    fluxName: str,
    flowSpeedName: str,
    energyName: str,
    temperatureName: str,
    sonicSpeed: str,
    speciesMass: float,
    sheathGamma: str,
    boundaryFlowSpeed: Union[str, None] = None,
    viscosityName: Union[str, None] = None,
    viscosityLimitMultName: Union[str, None] = None,
    leftBoundary=False,
) -> sc.CustomModel:
    """Adds Bohm outflow boundary conditions on the continuity, momentum, and energy equations for a given species.

    Args:
        speciesName (str): Name of the species. Used to name the model
        densName (str): Name of the density variable associated to the species
        fluxName (str): Name of the flux variable associated to the species
        flowSpeedName (str): Name of the flow speed variable associated to the species
        energyName (str): Name of the energy variable associated to the species
        tempName (str): Name of the temperature variable associated with the species
        sonicSpeed (str): Name of the sonic speed variable used to get the Bohm speed at the boundary
        speciesMass (float): Species mass in kg
        sheathGamma (str): Sheath gamma (scalar) at the boundary
        boundaryFlowSpeed (Union[str,None], optional): Flow speed at the boundary (scalar) used to calculate the kinetic energy outflow. Defaults to None, excluding this term.
        viscosityName (Union[str,None], optional): Viscosity used to calculate the boundary component of div(u*pi) heating. Defaults to None, excluding this term.
        viscosityLimitMultName (Union[str,None], optional): Viscosity limitation multiplier used to calculate the boundary component of div(u*pi) heating. Defaults to None, removing the limit.
        leftBoundary (bool, optional): If true will treat the boundary conditions as if they were at the left boundary of the domain. Defaults to False.

    Returns:
        sc.CustomModel: Model object containing boundary conditions on the fluid equations of given species
    """

    elMass = 9.10938e-31
    massRatio = elMass / speciesMass

    newModel = sc.CustomModel(modelTag="bohmBoundary_" + speciesName + "_")

    # Continuity BC
    bcCont = sc.GeneralMatrixTerm(
        densityName,
        customNormConst=-1.0,
        stencilData=sc.boundaryStencilDiv(
            flowSpeedName, lowerBoundVar=sonicSpeed, isLeft=leftBoundary
        ),
    )

    newModel.addTerm("continuity_Bohm", bcCont)

    # Momentum BC
    bcMom = sc.GeneralMatrixTerm(
        fluxName + "_dual",
        customNormConst=-1.0,
        stencilData=sc.boundaryStencilDiv(
            flowSpeedName, lowerBoundVar=sonicSpeed, isLeft=leftBoundary
        ),
    )

    newModel.addTerm("momentum_Bohm", bcMom)

    # Energy BC

    vDataBC = sc.VarData(reqRowVars=[sheathGamma], reqColVars=[temperatureName])

    energyBCGamma = sc.GeneralMatrixTerm(
        energyName,
        implicitVar=densityName,
        customNormConst=-1.0,
        varData=vDataBC,
        stencilData=sc.boundaryStencilDiv(
            flowSpeedName, sonicSpeed, isLeft=leftBoundary
        ),
    )

    newModel.addTerm("energy_BCGamma", energyBCGamma)

    # Kinetic energy BC

    if boundaryFlowSpeed is not None:
        vDataBCKin = sc.VarData(
            reqRowVars=[cast(str, boundaryFlowSpeed)], reqRowPowers=[2.0]
        )

        energyBCU = sc.GeneralMatrixTerm(
            energyName,
            implicitVar=densityName,
            customNormConst=-1 / massRatio,
            varData=vDataBCKin,
            stencilData=sc.boundaryStencilDiv(
                flowSpeedName, sonicSpeed, isLeft=leftBoundary
            ),
        )
        newModel.addTerm("energy_BCKin", energyBCU)

    # Viscous heating BC

    if viscosityName is not None:
        vData = sc.VarData()
        if viscosityLimitMultName is not None:
            vData = sc.VarData(reqColVars=[viscosityLimitMultName])

        energyBCVisc = sc.GeneralMatrixTerm(
            energyName,
            implicitVar=viscosityName,
            customNormConst=-1.0,
            stencilData=sc.boundaryStencilDiv(
                flowSpeedName, sonicSpeed, isLeft=leftBoundary
            ),
            varData=vData,
        )
        newModel.addTerm("energy_BCVisc", energyBCVisc)

    return newModel


def addNodeMatrixTermModel(
    wrapper: RKWrapper,
    modelTag: str,
    evolvedVar: str,
    termDefs: List[Tuple[ct.Node, str]],
    stencilData: Union[List[dict], None] = None,
):
    """Adds model with additive matrix terms of the form rowVar * implicitVar, where rowVar is a modelbound variable derived from a treeDerivation given a node. Optionally gives each matrix terms a different stencil.

    Args:
        wrapper (RKWrapper): Wrapper to add model to
        modelTag (str): Model tag for model to be added
        evolvedVar (str): Evolved variable for all matrix terms
        termDefs (List[Tuple[ct.Node,str]]): Term definitions. A list of (Node,implicitVarName) tuples, such that each matrix term is given by the variable calulated using the corresponding tuple's first component (the Node) and with the implicit variable name given by the second component
        stencilData (Union[List[dict, optional): Optional list of stencil data for each matrix term. Defaults to None.
    """

    newModel = sc.CustomModel(modelTag)

    mbData = sc.VarlikeModelboundData()

    if stencilData is not None:
        assert len(stencilData) == len(
            termDefs
        ), "If provided, stencilData in addNodeMatrixTermModel must conform to length of termDefs"

    for i, term in enumerate(termDefs):
        wrapper.addCustomDerivation(
            "nodeModelDeriv_" + modelTag + "_" + str(i), ct.treeDerivation(term[0])
        )
        mbData.addVariable(
            "nodeVar_" + str(i),
            sc.derivationRule(
                "nodeModelDeriv_" + modelTag + "_" + str(i), ct.getLeafVars(term[0])
            ),
        )

        usedStencilData = sc.diagonalStencil()
        if stencilData is not None:
            usedStencilData = stencilData[i]

        newTerm = sc.GeneralMatrixTerm(
            evolvedVar,
            term[1],
            varData=sc.VarData(reqMBRowVars=["nodeVar_" + str(i)]),
            stencilData=usedStencilData,
        )

        newModel.addTerm("nodeTerm_" + str(i), newTerm)

    newModel.setModelboundData(mbData.dict())
    wrapper.addModel(newModel)
