from . import stencils
from typing import Union, List, Tuple, cast, Optional, Dict
import numpy as np
from .grid import Grid, Profile
from .variable_container import Variable, MultiplicativeArgument
from .derivations import Species
from . import model_construction as mc
from . import derivations


def simpleSourceTerm(
    dummyImplicitVar: Variable, sourceProfile: Profile, timeSignal=mc.TimeSignalData()
) -> mc.MatrixTerm:
    """Simple "implicit" source term with given source profile. Uses a dummy implicit variable to cast the term into an implicit form by taking the source profile and multiplying and dividing by the implicit variable.

    Args:
        dummyImplicitVar (Variable): Dummy implicit variable
        sourceProfile (Profile): Spatial source profile
        timeSignal (TimeSignalData): Optional time signal component of source. Defaults to constant signal.

    Returns:
        MatrixTerm: Term ready for adding to a model
    """

    assert (
        sourceProfile.dim == "X"
    ), "simpleSourceTerm requires a spatial source profile"

    return timeSignal * (
        sourceProfile
        * (dummyImplicitVar ** (-1) * mc.DiagonalStencil()(dummyImplicitVar))
    ).rename("source_term")


def advection(
    advectedVar: Variable,
    flux: Union[Variable, MultiplicativeArgument],
    outflow: Tuple[bool, bool] = (False, False),
    advectionSpeed: Optional[Variable] = None,
    outflowLowerBound: Optional[Variable] = None,
) -> mc.Model:
    """Advection model with outflow, handling both simple and composite flux cases. If both the advected variable and the flux live on the same grid, the flux is interpolated on the corresponding cell boundaries and central differencing is used, otherwise staggered forward/backwards differencing is used.

    Args:
        advectedVar (Variable): The advected variable - must be implicit
        flux (Union[Variable,MultiplicativeArgument]): The advective flux, if it is composed of multiple variables, the last one is implicit
        outflow (Tuple[bool,bool], optional): Tuple representing whether there is outflow on the left/right side of the system (outflow[0] is left, [1] is right). Defaults to (False,False).
        advectionSpeed (Optional[Variable], optional): Advection speed used to reconstruct the outflow flux on the boundary as advectionSpeed*advectedVar - must live on regular grid. Defaults to None.
        outflowLowerBound (Optional[Variable], optional): Outflow lower bound variable. Defaults to None - setting the lower bound to 0, preventing inflow.

    Returns:
        Model: Advection model ready for adding to a ReMKiT1D contexts
    """

    advModel = mc.Model("advection_" + advectedVar.name)

    div: mc.Stencil = stencils.StaggeredDivStencil()

    fluxOnDualGrid = (
        flux.isOnDualGrid if isinstance(flux, Variable) else flux.firstArg.isOnDualGrid
    )
    if isinstance(flux, MultiplicativeArgument):
        assert (
            flux.scalar == 1
        ), "flux cannot have non-trivial scalar multiplier in advection"
        assert all(
            arg.isOnDualGrid == flux.firstArg.isOnDualGrid
            for _, arg in flux.args.items()
        ), "If flux in advection is a MultiplicativeArgument all components must live on the same grid"

    if fluxOnDualGrid == advectedVar.isOnDualGrid:
        div = (
            stencils.CentralDiffDivStencil()
        )  # Here the flux jacobian is not separately interpolated

    advModel.ddt[advectedVar] += -div(flux).rename("bulk_div")

    if any(outflow):
        assert (
            advectionSpeed is not None
        ), "advectionSpeed on the regular grid must be provided to advection if there is any outflow"
        assert (
            not advectionSpeed.isOnDualGrid
        ), "advectionSpeed in advection model must be on regular grid"
        if outflow[0]:
            divBC = stencils.BCDivStencil(
                advectionSpeed, outflowLowerBound, isLeft=True
            )
            advModel.ddt[advectedVar] += -divBC(advectedVar).rename("div_BC_left")
        if outflow[1]:
            divBC = stencils.BCDivStencil(advectionSpeed, outflowLowerBound)
            advModel.ddt[advectedVar] += -divBC(advectedVar).rename("div_BC_right")

    return advModel


def pressureGrad(
    fluxVar: Variable,
    pressure: Union[Variable, MultiplicativeArgument],
    normConst: float,
) -> mc.Model:
    """Pressure gradient contribution to the momentum equation. Can handle both collocated and staggered fluxes, but the pressure must live on the regular grid. If collocated uses central differencing with interpolation. If the grid is not periodic linearly extrapolates pressure onto boundaries if the flux is on the regular grid.

    Args:
        fluxVar (Variable): Implicit flux variable
        pressure (Union[Variable,MultiplicativeArgument]): Pressure variable or composite varible (the last variable in the MultiplicativeArgument is taken to be the implicit variable)
        normConst (float): Normalisation constant for the pressure gradient term

    Returns:
        Model: Pressure gradient model ready for adding to a ReMKiT1D context
    """

    pressureGradModel = mc.Model("pressure_grad_" + fluxVar.name)

    grad: mc.Stencil = stencils.StaggeredGradStencil()

    fluxOnDualGrid = fluxVar.isOnDualGrid
    if isinstance(pressure, Variable):
        assert (
            not pressure.isOnDualGrid
        ), "pressure in pressureGrad must be on regular grid"
    if isinstance(pressure, MultiplicativeArgument):
        assert (
            pressure.scalar == 1
        ), "pressure cannot have non-trivial scalar multiplier in pressureGrad"
        assert all(
            not arg.isOnDualGrid for _, arg in pressure.args.items()
        ), "If pressure in pressureGrad is a MultiplicativeArgument all components must live on the regular grid"

    if not fluxOnDualGrid:
        grad = stencils.CentralDiffGradStencil()

    pressureGradModel.ddt[fluxVar] += -normConst * grad(pressure).rename("bulk_grad")

    if not fluxOnDualGrid:
        if not fluxVar.grid.isPeriodic:
            pressureGradModel.ddt[fluxVar] += -normConst * stencils.BCGradStencil(True)(
                pressure
            ).rename("grad_BC_left") - normConst * stencils.BCGradStencil()(
                pressure
            ).rename(
                "grad_BC_right"
            )

    return pressureGradModel


def ampereMaxwell(
    eField: Variable,
    speciesFluxes: List[Variable],
    species: List[Species],
    norms: Dict[str, float],
) -> mc.Model:
    """Generate dE/dt = -j/epsilon0 matrix terms by calculating currents based on species fluxes and charges.

    Args:
        eField (Variable): Electric field variable
        speciesFluxes (List[Variable]): List of fluxes associated with each of the passed species
        species (List[Species]): List of species, must conform to list of fluxes
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)

    Returns:
        Model: Ampere-Maxwell model ready for addition to ReMKiT1D context
    """
    assert len(speciesFluxes) == len(
        species
    ), "speciesFluxes and species passed to ampereMaxwell model must be of same size"

    elCharge = 1.60218e-19
    epsilon0 = 8.854188e-12
    normConst = (
        elCharge
        / epsilon0
        * norms["density"]
        * norms["time"]
        * norms["speed"]
        / norms["EField"]
    )

    amModel = mc.Model("AM_model")
    for i, flux in enumerate(speciesFluxes):
        amModel.ddt[eField] += (
            -species[i].charge
            * normConst
            * mc.DiagonalStencil()(flux).rename("current_" + flux.name)
        )

    return amModel


def lorentzForces(
    eField: Variable,
    speciesFluxes: List[Variable],
    speciesDensities: List[Variable],
    species: List[Species],
    norms: Dict[str, float],
) -> mc.Model:
    """Generate Lorentz force matrix terms in the momentum equation for each species b: m_b*dG_b/dt = n_b*Z_b*e*E.
    Args:
        eField (Variable): Electric field variable - implicit
        speciesFluxes (List[Variable]): Evolved fluxes - implicit - should live on the same grid as eField
        speciesDensities (List[Variable]): Species densities - should live on the same grid as eField
        species (List[Species]): Species corresponding to densities and fluxes
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)

    Returns:
        Model: Lorentz Force model ready to be added to a ReMKiT1D context
    """
    assert len(speciesFluxes) == len(
        species
    ), "speciesFluxes and species passed to lorentzForces model must be of same size"
    assert len(speciesDensities) == len(
        species
    ), "speciesDensities and species passed to lorentzForces model must be of same size"

    elCharge = 1.60218e-19
    amu = 1.6605390666e-27  # atomic mass unit

    lForceModel = mc.Model("lorentz_forces")
    for i, flux in enumerate(speciesFluxes):

        speciesMass = amu * species[i].atomicA
        speciesCharge = elCharge * species[i].charge
        normConst = (
            speciesCharge
            / speciesMass
            * norms["EField"]
            * norms["time"]
            / norms["speed"]
        )

        lForceModel.ddt[flux] += (
            normConst
            * speciesDensities[i]
            * mc.DiagonalStencil()(eField).rename("lorentz_" + flux.name)
        )

    return lForceModel


def lorentzForceWork(
    eField: Variable,
    speciesFluxes: List[Variable],
    speciesEnergies: List[Variable],
    species: List[Species],
    norms: Dict[str, float],
) -> mc.Model:
    """Generate Lorentz force work matrix terms in the energy density evolution equation for each species: dW_b/dt = Z_b * e * G_b * E. Assumes default energy normalization (e*T_0[eV]).

    Args:
        eField (Variable): Electric field variable - implicit
        speciesFluxes (List[Variable]): Species fluxes - should live on the same grid as the eField
        speciesEnergies (List[Variable]): Species energy densities - evolved variables
        species (List[Species]): List of species corresponding to the evolved energy densities
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)

    Returns:
        Model: Lorentz Force Work model ready to be added to a ReMKiT1D context
    """
    assert len(speciesFluxes) == len(
        species
    ), "speciesFluxes and species passed to lorentzForces model must be of same size"
    assert len(speciesEnergies) == len(
        species
    ), "speciesEnergies and species passed to lorentzForces model must be of same size"

    lFWModel = mc.Model("lorentz_force_work")
    for i, flux in enumerate(speciesFluxes):
        speciesCharge = species[i].charge
        normConst = (
            speciesCharge
            * norms["EField"]
            * norms["time"]
            * norms["speed"]
            / norms["eVTemperature"]
        )

        lFWModel.ddt[speciesEnergies[i]] += normConst * mc.DiagonalStencil()(
            flux * eField
        ).rename("lorentz_work_" + speciesEnergies[i].name)

    return lFWModel


def implicitTemperatures(
    speciesEnergies: List[Variable],
    speciesDensities: List[Variable],
    speciesTemperatures: List[Variable],
    species: List[Species],
    norms: Dict[str, float],
    speciesFluxes: Optional[List[Variable]] = None,
    **kwargs
) -> mc.Model:
    """Generate implicit temperature derivation matrix terms for each species: d*n_b*k*T_b/2 + m_b*n_b*u_b**2/2 = W_b, where d is the number of degrees of freedom. Temperatures here are assumed to be stationary and implicit variables. The kinetic energy contribution uses interpolation, so should be used with care in regions of poorly resolved flow gradients. Assumes default normalization.

    Args:
        speciesEnergies (List[Variable]): Species energy densities - implicit
        speciesDensities (List[Variable]): Species densities
        speciesTemperatures (List[Variable]): Species densities - implicit and stationary
        species (List[Species]): Species corresponding to each temperature
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)
        speciesFluxes (Optional[List[Variable]], optional): Species fluxes - implicit - if not present kinetic contributions are ignored. Defaults to None

    kwargs:
        degreesOfFreedom (int): Number of degrees of freedom. Defaults to 3.
        evolvedXU2Cells (List[int]): Optional spatial cell indices where the kinetic contribution is added. Defaults to None, including it everywhere if fluxes are passed.

    Returns:
        Model: Implicit temperature derivation model ready to be added into a ReMKiT1D context
    """
    if speciesFluxes is not None:
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

    tempModel = mc.Model("implicit_temperatures")
    elCharge = 1.60218e-19
    amu = 1.6605390666e-27  # atomic mass unit
    normW = 2 / kwargs.get("degreesOfFreedom", 3)
    for i, temp in enumerate(speciesTemperatures):
        assert (
            temp.isStationary
        ), "Temperatures in implicitTemperatures are expected to be stationary"
        speciesMass = amu * species[i].atomicA
        normU2 = (
            -speciesMass / (3 * elCharge) * norms["speed"] ** 2 / norms["eVTemperature"]
        )
        diag = mc.DiagonalStencil()

        tempModel.ddt[temp] += -diag(temp).rename(
            "identity_" + temp.name
        ) + normW * speciesDensities[i] ** (-1) * diag(speciesEnergies[i]).rename(
            "W_term_" + temp.name
        )

        if speciesFluxes is not None:
            colDensity: Variable = (
                cast(Variable, speciesDensities[i].dual)
                if speciesFluxes[i].isOnDualGrid
                else speciesDensities[i]
            )
            evolvedXU2Cells = kwargs.get("evolvedXU2Cells", None)
            if evolvedXU2Cells is not None:
                diag = mc.DiagonalStencil(evolvedXU2Cells)

            tempModel.ddt[temp] += normU2 * diag(
                speciesFluxes[i] ** 2 / colDensity**2
            ).rename("U2_term_" + temp.name)

    return tempModel


def kinAdvX(
    distribution: Variable, grid: Grid, evolvedHarmonics: Optional[List[int]] = None
) -> mc.Model:
    """Return kinetic advection model for electrons in x direction using matrix terms. This is the harmonic form of the vdf/dx term, coupling adjacent harmonics. Assumes default normalization.

    Args:
        distribution (Variable): Electron distribution variable
        grid (Grid): Used grid object
        evolvedHarmonics (Optional[List[int]], optional): Optional subselection of evolved harmonics. Defaults to None, evolving all harmonics.

    Returns:
        Model: Distribution x advection model ready for addition into a ReMKiT1D context
    """
    assert (
        distribution.isDistribution
    ), "kinAdvX distribution must be a distribution variable"
    usedHarmonics = list(range(1, grid.numH + 1))

    if evolvedHarmonics is not None:
        usedHarmonics = evolvedHarmonics

    advModel = mc.Model("kinetic_X_advection")
    lNums = [grid.lGrid[i - 1] for i in range(1, grid.numH + 1)]
    mNums = [grid.mGrid[i - 1] for i in range(1, grid.numH + 1)]
    v = grid.profile(grid.vGrid, dim="V", latexName="v")

    for harmonic in usedHarmonics:
        if lNums[harmonic - 1] > 0:
            normConst = -(lNums[harmonic - 1] - mNums[harmonic - 1]) / (
                2.0 * lNums[harmonic - 1] - 1.0
            )

            stencil = stencils.DistGradStencil(
                harmonic,
                grid.getH(
                    lNum=lNums[harmonic - 1] - 1,
                    mNum=mNums[harmonic - 1],
                    im=grid.imaginaryHarmonic[harmonic - 1],
                ),
            )
            advModel.ddt[distribution] += (
                normConst
                * (v * stencil(distribution))
                .rename("adv_minus_" + str(harmonic))
                .withFixedMatrix()
            )

        if lNums[harmonic - 1] < grid.lMax:
            normConst = -(lNums[harmonic - 1] + mNums[harmonic - 1] + 1.0) / (
                2.0 * lNums[harmonic - 1] + 3.0
            )

            stencil = stencils.DistGradStencil(
                harmonic,
                grid.getH(
                    lNum=lNums[harmonic - 1] + 1,
                    mNum=mNums[harmonic - 1],
                    im=grid.imaginaryHarmonic[harmonic - 1],
                ),
            )

            advModel.ddt[distribution] += (
                normConst
                * (v * stencil(distribution))
                .rename("adv_plus_" + str(harmonic))
                .withFixedMatrix()
            )

    return advModel


def advectionEx(
    distribution: Variable, eField: Variable, grid: Grid, norms: Dict[str, float]
) -> mc.Model:
    """Returns electric field advection matrix terms for electrons in x direction: the harmonic decomposition of E/m*df/dv. Defines necessary custom derivations. The electric field is taken to be implicit, so it will be interpolated implicitly onto cell centres for even l harmonics.

    Args:
        distribution (Variable): Implicit electron distribution function
        eField (Variable): Implicit electric field
        grid (Grid): Used grid object
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)

    Returns:
        Model: Electric field velocity space advection model ready for adding to ReMKiT1D context
    """
    numH = grid.numH
    derivReqFun = distribution.dual if distribution.isOnDualGrid else distribution
    mbData = mc.VarlikeModelboundData()
    vGrid = grid.vGrid
    dv: List[float] = []

    for v in vGrid:
        dv.append(2 * (v - sum(dv)))

    vBoundary = [vGrid[i] + dv[i] / 2 for i in range(len(dv))]
    G = {}
    H = {}
    for harmonic in range(1, numH + 1):
        lNum = grid.lGrid[harmonic - 1]
        # Add G_l variables
        innerV = np.array([v ** (-lNum) for v in vBoundary])
        innerV[-1] = 0
        iV = grid.profile(innerV, "V", latexName="v^{-" + str(lNum) + "}")
        oV = grid.profile(
            np.array([v**lNum for v in vGrid.tolist()]),
            "V",
            latexName="v^{" + str(lNum) + "}",
        )
        vifAtZero = (
            (
                1.0 / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
                -vGrid[0] ** 2 / vGrid[1] ** 2 / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
            )
            if lNum == 0
            else (1.0 / vGrid[1] ** lNum, 0)
        )

        derivG = derivations.DDVDerivation(
            "G" + str(harmonic), grid, harmonic, iV, oV, vifAtZero
        )
        G[harmonic] = derivG(derivReqFun)
        mbData.addVar(G[harmonic])

        # Add H_l variables

        innerV = np.array([v ** (lNum + 1) for v in vBoundary])
        innerV[-1] = 0
        iV = grid.profile(innerV, "V", latexName="v^{" + str(lNum + 1) + "}")
        oV = grid.profile(
            np.array([v ** (-lNum - 1) for v in vGrid.tolist()]),
            "V",
            latexName="v^{" + str(-lNum - 1) + "}",
        )

        derivH = derivations.DDVDerivation("H" + str(harmonic), grid, harmonic, iV, oV)
        H[harmonic] = derivH(derivReqFun)
        mbData.addVar(H[harmonic])

    advExModel = mc.Model("advection_Ex")
    advExModel.setModelboundData(mbData)
    elCharge = 1.60218e-19
    elMass = 9.10938e-31

    lNums = [grid.lGrid[i] for i in range(numH)]
    mNums = [grid.mGrid[i] for i in range(numH)]

    chargeMassRatio = elCharge / elMass
    for harmonic in range(1, numH + 1):
        # Add G terms
        if lNums[harmonic - 1] > 0:
            multConst = (
                chargeMassRatio
                * (lNums[harmonic - 1] - mNums[harmonic - 1])
                / (2.0 * lNums[harmonic - 1] - 1.0)
                * norms["EField"]
                * norms["time"]
                / norms["velGrid"]
            )
            GHarmonic = grid.getH(
                lNum=lNums[harmonic - 1] - 1,
                mNum=mNums[harmonic - 1],
                im=grid.imaginaryHarmonic[harmonic - 1],
            )
            advExModel.ddt[distribution] += (
                multConst
                * G[GHarmonic]
                @ mc.DiagonalStencil(evolvedHarmonics=[harmonic])(eField).rename(
                    "eAdv_G_" + str(harmonic)
                )
            )

        # Add H terms
        if lNums[harmonic - 1] < grid.lMax:
            multConst = (
                chargeMassRatio
                * (lNums[harmonic - 1] + mNums[harmonic - 1] + 1.0)
                / (2.0 * lNums[harmonic - 1] + 3.0)
                * norms["EField"]
                * norms["time"]
                / norms["velGrid"]
            )

            HHarmonic = grid.getH(
                lNum=lNums[harmonic - 1] + 1,
                mNum=mNums[harmonic - 1],
                im=grid.imaginaryHarmonic[harmonic - 1],
            )
            advExModel.ddt[distribution] += (
                multConst
                * H[HHarmonic]
                @ mc.DiagonalStencil(evolvedHarmonics=[harmonic])(eField).rename(
                    "eAdv_H_" + str(harmonic)
                )
            )

    return advExModel


def eeCollIsotropic(
    distribution: Variable,
    elTemperature: Variable,
    elDensity: Variable,
    norms: Dict[str, float],
    grid: Grid,
    textbook: derivations.Textbook,
) -> mc.Model:
    """Return e-e collision model for l=0 harmonic using matrix terms. Responsible for adding the derivations needed for its modelbound data. Assumes default normalization.

    Args:
        distribution (Variable): Electron distribution variable - implicit
        elTemperature (Variable): Electron temperature variable
        elDensity (Variable): Electron density variable
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)
        grid (Grid): Used grid object
        textbook (derivations.Textbook): Used textbook object - needed to retrieve built-in derivations

    Returns:
        Model: Isotropic collision model for electrons ready for addition to ReMKiT1D context
    """
    mbData = mc.VarlikeModelboundData()

    mbData.addVar(derivations.HarmonicExtractorDerivation("f0", grid, 1)(distribution))

    mbData.addVar(
        Variable(
            "dragCCL",
            grid,
            derivation=textbook["cclDragCoeff"],
            derivationArgs=["f0"],
            isSingleHarmonic=True,
        )
    )

    mbData.addVar(
        Variable(
            "diffCCL",
            grid,
            derivation=textbook["cclDiffusionCoeff"],
            derivationArgs=["f0", "weightCCL"],
            isSingleHarmonic=True,
        )
    )

    mbData.addVar(
        Variable(
            "weightCCL",
            grid,
            derivation=textbook["cclWeight"],
            derivationArgs=["dragCCL", "diffCCL"],
            isSingleHarmonic=True,
        )
    )

    mbData.addVar(textbook["logLee"](elTemperature, elDensity))

    eeModel = mc.Model("isotropic_ee")
    eeModel.setModelboundData(mbData)

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    normConst = (
        elCharge**4
        / (4 * np.pi * elMass**2 * epsilon0**2)
        * norms["density"]
        * norms["time"]
        / norms["velGrid"] ** 3
    )

    vOuter = grid.profile(
        np.array([1.0 / v**2 for v in grid.vGrid]), dim="V", latexName="v^{-2}"
    )

    # Drag term
    eeModel.ddt[distribution] += (
        normConst
        * mbData["logLee"]
        @ (
            vOuter
            * stencils.DDVStencil(
                1, 1, CCoeff=mbData["dragCCL"], interpCoeffs=mbData["weightCCL"]
            )(distribution)
        ).rename("drag_term")
    )

    # Diffusion term
    eeModel.ddt[distribution] += (
        normConst
        * mbData["logLee"]
        @ (
            vOuter
            * stencils.D2DV2Stencil(1, 1, diffCoeff=mbData["diffCCL"])(distribution)
        ).rename("diff_term")
    )

    return eeModel


def eiCollIsotropic(
    grid: Grid,
    textbook: derivations.Textbook,
    norms: Dict[str, float],
    distribution: Variable,
    elTemperature: Variable,
    elDensity: Variable,
    ionTemperature: Variable,
    ionDensity: Variable,
    ionSpecies: Species,
    ionEnVar: Optional[Variable] = None,
) -> mc.Model:
    """Return e-i collision model for l=0 harmonic using matrix terms. Responsible for adding derivations used in its modelbound data. Assumes default normalization.

    Args:
        grid (Grid): Used grid object
        textbook (Textbook): Textbook object used to retrieve built-in derivations
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)
        grid (Grid): Used grid object
        distribution (Variable): Electron distribution function - implicit
        elTemperature (Variable): Electron temperature variable
        elDensity (Variable): Electron density variable
        ionTemperature (Variable): Ion temperature variable
        ionDensity (Variable): Ion density variable
        ionSpecies (Species): Species object corresponding to the ions
        ionEnVar (Optional[Variable], optional): Ion energy density to pass on the energy moment of the collision operator to - must be implicit. Defaults to None, excluding ion terms.

    Returns:
        Model: Ion-electron isotropic collision operator model ready for addition to a ReMKiT1D context
    """
    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity
    amu = 1.6605390666e-27  # atomic mass unit
    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)

    gamma0norm = gamma0norm * ionSpecies.charge**2 * elMass / (ionSpecies.atomicA * amu)

    mbData = mc.VarlikeModelboundData()
    mbData.addVar(
        Variable(
            "logLei",
            grid,
            derivation=textbook["logLei" + ionSpecies.name],
            derivationArgs=[elTemperature.name, elDensity.name],
        )
    )

    normConst = gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    vGrid = grid.vGrid
    dv = grid.vWidths
    vBoundary = [vGrid[i] + dv[i] / 2 for i in range(len(dv))]
    vOuter = grid.profile(
        np.array([1.0 / v**2 for v in grid.vGrid]), "V", latexName="v^{-2}"
    )
    innerV = grid.profile(
        np.array([1.0 / (2 * v) for v in vBoundary]), "V", latexName="\\frac{1}{2v}"
    )

    eiModel = mc.Model("isotropic_ei")
    eiModel.setModelboundData(mbData)
    # drag term
    eiModel.ddt[distribution] += (
        normConst
        * ionDensity
        * (
            mbData["logLei"] @ (vOuter * stencils.DDVStencil(1, 1)(distribution))
        ).rename("drag_term")
    )

    # diffusion term
    eiModel.ddt[distribution] += (
        normConst
        * ionDensity
        * ionTemperature
        * (
            mbData["logLei"]
            @ (vOuter * stencils.D2DV2Stencil(1, 1, diffCoeff=innerV)(distribution))
        ).rename("diff_term")
    )

    if ionEnVar is not None:

        eiModel.ddt[ionEnVar] += -stencils.TermMomentStencil(1, 2, "diff_term")(
            distribution
        ).rename("diff_term_en") - stencils.TermMomentStencil(1, 2, "drag_term")(
            distribution
        ).rename(
            "drag_term_en"
        )

    return eiModel


def stationaryIonEIColl(
    grid: Grid,
    textbook: derivations.Textbook,
    norms: Dict[str, float],
    distribution: Variable,
    ionDensity: Variable,
    elDensity: Variable,
    elTemperature: Variable,
    ionSpecies: Species,
    evolvedHarmonics: Optional[List[int]] = None,
) -> mc.Model:
    """Return stationary ion electron-ion collision operator model using matrix terms. Assumes default normalization. NOTE: Must be called separately for odd and even harmonics if staggered, and the fluid quantities used must reflect that (dual for odd harmonics).

    Args:
        grid (Grid): Used grid object
        textbook (derivations.Textbook): Textbook object used to retrieve built-in derivations
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)
        distribution (Variable): Electron distribution function - implicit
        ionDensity (Variable): Ion density
        elDensity (Variable): Electron density
        elTemperature (Variable): Electron temperature
        ionSpecies (Species): Ion species associated with the collision
        evolvedHarmonics (Optional[List[int]], optional): List of evolved harmonics. Defaults to None- evolving all harmonics.

    Returns:
        Model: Electron-ion collision operator model for l>0 assuming stationary ions, ready for addition into a ReMKiT1D context
    """
    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)
    gamma0norm = gamma0norm * ionSpecies.charge**2
    normConst = -gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    mbData = mc.VarlikeModelboundData()
    mbData.addVar(
        Variable(
            "logLei",
            grid,
            derivation=textbook["logLei" + ionSpecies.name],
            derivationArgs=[elTemperature.name, elDensity.name],
        )
    )

    lNums = grid.lGrid
    hProfile = []
    for l in lNums:
        hProfile.append(l * (l + 1.0) / 2.0)

    h = grid.profile(np.array(hProfile), "H", "\\frac{l(l+1)}{2}")
    v = grid.profile(np.array([1.0 / v**3 for v in grid.vGrid]), "V", "\\frac{1}{v^3}")

    eiModel = mc.Model("ei_stationary_anisotropic")
    eiModel.setModelboundData(mbData)

    eiModel.ddt[distribution] += (
        normConst
        * ionDensity
        * (
            mbData["logLei"]
            @ (
                h
                * (
                    v
                    * mc.DiagonalStencil(evolvedHarmonics=evolvedHarmonics)(
                        distribution
                    )
                )
            )
        ).rename("ei_colls_stationary")
    )

    return eiModel


def flowingIonEIColl(
    grid: Grid,
    textbook: derivations.Textbook,
    norms: Dict[str, float],
    distribution: Variable,
    ionDensity: Variable,
    ionFlowSpeed: Variable,
    elDensity: Variable,
    elTemperature: Variable,
    ionSpecies: Species,
    evolvedHarmonics: List[int],
    ionFlux: Optional[Variable] = None,
) -> mc.Model:
    """Return a flowing cold ion electron-ion collision model using matrix terms. Assumes default normalization. NOTE: Must be called separately for odd and even harmonics if staggered, and the fluid quantities used must reflect that (dual for odd harmonics), except for the ion density, which must be implicit (its dual will be inferred)

    Args:
        grid (Grid): Used grid object
        textbook (Textbook): Textbook object used to retrieve built-in derivations
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)
        distribution (Variable): Electron distribution function - implicit
        ionDensity (Variable): Ion density - should live on the regular grid, i.e. be implicit
        ionFlowSpeed (Variable): Ion flow speed - on the same grid as the evolved harmonics
        elDensity (Variable): Electron density - on the same grid as the evolved harmonics
        elTemperature (Variable): Electron temperature - on the same grid as the evolved harmonics
        ionSpecies (Variable): Ion species corresponding to the cold ions
        evolvedHarmonics (List[int]): List of evolved harmonics, must all correspond to either odd or even l numbers
        ionFlux (Optional[Variable], optional): Ion flux variable (dual if harmonics staggered) used to pass the electron-ion friction terms to the ions (requires l=1 to be among the evolved harmonics). Defaults to None.

    Returns:
        Model: Electron-ion collision operator model assuming flowing cold ions
    """
    # NOTE: Needs a lot of work on optimization
    assert (
        1 not in evolvedHarmonics
    ), "flowingIonEIColl cannot be used to evolve harmonic with index 1"

    vGrid = grid.vGrid
    lNums = grid.lGrid

    derivReqFun = distribution.dual if ionFlowSpeed.isOnDualGrid else distribution

    mbData = mc.VarlikeModelboundData()

    mbData.addVar(
        Variable(
            "CII0",
            grid,
            isDistribution=True,
            derivation=derivations.coldIonIDeriv("CII0", 0),
            derivationArgs=[ionFlowSpeed.name],
        ).onDualGrid(ionFlowSpeed.isOnDualGrid)
    )
    mbData.addVar(
        Variable(
            "CII2",
            grid,
            isDistribution=True,
            derivation=derivations.coldIonIDeriv("CII2", 2),
            derivationArgs=[ionFlowSpeed.name],
        ).onDualGrid(ionFlowSpeed.isOnDualGrid)
    )
    mbData.addVar(
        Variable(
            "CIJ-1",
            grid,
            isDistribution=True,
            derivation=derivations.coldIonJDeriv("CIJ-1", -1),
            derivationArgs=[ionFlowSpeed.name],
        ).onDualGrid(ionFlowSpeed.isOnDualGrid)
    )
    mbData.addVar(
        derivations.HarmonicExtractorDerivation("f0", grid, 1)(mbData["CII0"])
        .onDualGrid(ionFlowSpeed.isOnDualGrid)
        .rename("CII0sh")
    )
    mbData.addVar(
        derivations.HarmonicExtractorDerivation("f0", grid, 1)(mbData["CII2"])
        .onDualGrid(ionFlowSpeed.isOnDualGrid)
        .rename("CII2sh")
    )
    mbData.addVar(
        derivations.HarmonicExtractorDerivation("f0", grid, 1)(mbData["CIJ-1"])
        .onDualGrid(ionFlowSpeed.isOnDualGrid)
        .rename("CIJ-1sh")
    )

    polyDeriv1 = derivations.PolynomialDerivation(
        "sumTerm",
        0.0,
        polyPowers=np.array([1.0, 1.0, 1.0]),
        polyCoeffs=np.array([-1.0, 2.0, 3.0]),
    )
    polyDeriv2 = derivations.PolynomialDerivation(
        "sumTerm2",
        0.0,
        polyPowers=np.array([1.0, 1.0]),
        polyCoeffs=np.array([-1.0, 2.0]),
    )
    mbData.addVar(
        polyDeriv1(mbData["CII2sh"], mbData["CIJ-1sh"], mbData["CII0sh"])
        .onDualGrid(ionFlowSpeed.isOnDualGrid)
        .rename("IJSum")
    )
    mbData.addVar(
        polyDeriv2(mbData["CII2sh"], mbData["CIJ-1sh"])
        .onDualGrid(ionFlowSpeed.isOnDualGrid)
        .rename("IJSum2")
    )

    df0 = derivations.DDVDerivation(
        "df0/dv",
        grid,
        1,
        vifAtZero=(
            1.0 / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
            -vGrid[0] ** 2 / vGrid[1] ** 2 / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
        ),
    )
    ddf0 = derivations.D2DV2Derivation("d2f0/dv2", grid, 1)

    mbData.addVar(df0(derivReqFun).onDualGrid(ionFlowSpeed.isOnDualGrid).rename("df0"))
    mbData.addVar(
        ddf0(derivReqFun).onDualGrid(ionFlowSpeed.isOnDualGrid).rename("ddf0")
    )
    mbData.addVar(
        Variable(
            "logLei",
            grid,
            derivation=textbook["logLei" + ionSpecies.name],
            derivationArgs=[elTemperature.name, elDensity.name],
        ).onDualGrid(ionFlowSpeed.isOnDualGrid)
    )

    for h in evolvedHarmonics:
        l = lNums[h - 1]
        if "CII" + str(l) not in mbData.varNames:
            mbData.addVar(
                Variable(
                    "CII" + str(l),
                    grid,
                    isDistribution=True,
                    derivation=derivations.coldIonIDeriv("CII" + str(l), l),
                    derivationArgs=[ionFlowSpeed.name],
                ).onDualGrid(ionFlowSpeed.isOnDualGrid)
            )

        if "CII" + str(l + 2) not in mbData.varNames:
            mbData.addVar(
                Variable(
                    "CII" + str(l + 2),
                    grid,
                    isDistribution=True,
                    derivation=derivations.coldIonIDeriv("CII" + str(l + 2), l + 2),
                    derivationArgs=[ionFlowSpeed.name],
                ).onDualGrid(ionFlowSpeed.isOnDualGrid)
            )

        if "CIJ" + str(-l - 1) not in mbData.varNames:
            mbData.addVar(
                Variable(
                    "CIJ" + str(-l - 1),
                    grid,
                    isDistribution=True,
                    derivation=derivations.coldIonJDeriv("CIJ" + str(-l - 1), -l - 1),
                    derivationArgs=[ionFlowSpeed.name],
                ).onDualGrid(ionFlowSpeed.isOnDualGrid)
            )

        if "CIJ" + str(1 - l) not in mbData.varNames:
            mbData.addVar(
                Variable(
                    "CIJ" + str(1 - l),
                    grid,
                    isDistribution=True,
                    derivation=derivations.coldIonJDeriv("CIJ" + str(1 - l), 1 - l),
                    derivationArgs=[ionFlowSpeed.name],
                ).onDualGrid(ionFlowSpeed.isOnDualGrid)
            )

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity
    amu = 1.6605390666e-27  # atomic mass unit
    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)
    gamma0norm = gamma0norm * ionSpecies.charge**2
    elIonMassRatio = elMass / (ionSpecies.atomicA * amu)

    normIonFriction = -elIonMassRatio / 3

    eiColl = mc.Model("cold_ion_ei_colls")
    eiColl.setModelboundData(mbData)

    for harmonic in evolvedHarmonics:
        l = lNums[harmonic - 1]

        usedDensity: Variable = (
            cast(Variable, ionDensity.dual) if ionFlowSpeed.isOnDualGrid else ionDensity
        )
        # velocity diffusion terms with fl

        v1 = grid.profile(np.array([1 / v for v in vGrid]), "V", "\\frac{1}{v}")

        normConst = (
            gamma0norm / 3 * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )

        velDiff = stencils.D2DV2Stencil(
            harmonic, harmonic, adfAtZero=(0, 0) if l > 1 else (1 / vGrid[1], 0)
        )

        eiColl.ddt[distribution] += (
            normConst
            * (mbData["logLei"] * mbData["CII2sh"])
            @ (usedDensity * (v1 * velDiff(distribution))).rename(
                "diffTermI" + str(harmonic)
            )
        )

        eiColl.ddt[distribution] += (
            normConst
            * (mbData["logLei"] * mbData["CIJ-1sh"])
            @ (usedDensity * (v1 * velDiff(distribution))).rename(
                "diffTermJ" + str(harmonic)
            )
        )

        # velocity deriv terms with fl

        v2 = grid.profile(np.array([1 / v**2 for v in vGrid]), "V", "\\frac{1}{v^2}")

        ddv = stencils.DDVStencil(harmonic, harmonic)

        eiColl.ddt[distribution] += (
            normConst
            * (mbData["logLei"] * mbData["IJSum2"])
            @ (usedDensity * (v2 * ddv(distribution))).rename("dfdv" + str(harmonic))
        )

        # -l(l+1)/2 terms

        v3 = grid.profile(np.array([1 / v**3 for v in vGrid]), "V", "\\frac{1}{v^3}")

        diag = mc.DiagonalStencil(evolvedHarmonics=[harmonic])

        eiColl.ddt[distribution] += (
            -(l * (l + 1.0) / 2.0)
            * normConst
            * (mbData["logLei"] * mbData["IJSum"])
            @ (usedDensity * (v3 * diag(distribution))).rename("termLL" + str(harmonic))
        )

        # I/J(f_l) terms

        # d2f0/dv2 terms
        # C1*I_{l+2} term
        MBVars = mbData["logLei"] * mbData["ddf0"] * mbData["CII" + str(l + 2)]
        C = (l + 1) * (l + 2) / ((2 * l + 1) * (2 * l + 3))  # C1
        normConst = (
            C
            * gamma0norm
            / 2
            * norms["density"]
            * norms["time"]
            / norms["velGrid"] ** 3
        )

        eiColl.ddt[distribution] += normConst * (
            MBVars @ (v1 * diag(ionDensity)).rename("C1Il+2_h=" + str(harmonic))
        )

        # C1*J_{-l-1} term
        MBVars = mbData["logLei"] * mbData["ddf0"] * mbData["CIJ" + str(-1 - l)]

        eiColl.ddt[distribution] += normConst * (
            MBVars
            @ (v1 * diag(ionDensity))
            .rename("C1J-l-1_h=" + str(harmonic))
            .withSkippingPattern()
        )

        # C2*I_l term
        C = -(l - 1) * l / ((2 * l + 1) * (2 * l - 1))  # C2
        normConst = (
            C
            * gamma0norm
            / 2
            * norms["density"]
            * norms["time"]
            / norms["velGrid"] ** 3
        )
        MBVars = mbData["logLei"] * mbData["ddf0"] * mbData["CII" + str(l)]

        eiColl.ddt[distribution] += normConst * (
            MBVars
            @ (v1 * diag(ionDensity))
            .rename("C2Il_h=" + str(harmonic))
            .withSkippingPattern()
        )

        # C2*J1-l term
        MBVars = mbData["logLei"] * mbData["ddf0"] * mbData["CIJ" + str(1 - l)]

        eiColl.ddt[distribution] += (
            normConst
            * (MBVars @ (v1 * diag(ionDensity)))
            .rename("C2J1-l_h=" + str(harmonic))
            .withSkippingPattern()
        )

        # df0/dv terms

        # C3*I_{l+2} term
        MBVars = mbData["logLei"] * mbData["df0"] * mbData["CII" + str(l + 2)]
        C = -((l + 1) * l / 2 + l + 1) / ((2 * l + 1) * (2 * l + 3))  # C3
        normConst = (
            C * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )

        eiColl.ddt[distribution] += (
            normConst
            * MBVars
            @ (v2 * diag(ionDensity))
            .rename("C3Il+2_h=" + str(harmonic))
            .withSkippingPattern()
        )

        # C4*J_{-l-1} term
        MBVars = mbData["logLei"] * mbData["df0"] * mbData["CIJ" + str(-1 - l)]
        C = (-(l + 1) * l / 2 + l + 2) / ((2 * l + 1) * (2 * l + 3)) + l / (
            2 * l + 1
        )  # C4 + l/(2l+1)
        normConst = (
            C * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )

        eiColl.ddt[distribution] += (
            normConst
            * MBVars
            @ (v2 * diag(ionDensity))
            .rename("C4J-l-1_h=" + str(harmonic))
            .withSkippingPattern()
        )

        # C5*I_l term
        MBVars = mbData["logLei"] * mbData["df0"] * mbData["CII" + str(l)]

        C = ((l + 1) * l / 2 + l - 1) / ((2 * l + 1) * (2 * l - 1)) - (l + 1) / (
            2 * l + 1
        )  # C5 -(l+1)/(2l+1)
        normConst = (
            C * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )
        eiColl.ddt[distribution] += (
            normConst
            * MBVars
            @ (v2 * diag(ionDensity))
            .rename("C5Il_h=" + str(harmonic))
            .withSkippingPattern()
        )

        # # C6*J1-l term
        MBVars = mbData["logLei"] * mbData["df0"] * mbData["CIJ" + str(1 - l)]

        C = -((l + 1) * l / 2 - l) / ((2 * l + 1) * (2 * l - 1))  # C6
        normConst = (
            C * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )
        eiColl.ddt[distribution] += (
            normConst
            * MBVars
            @ (v2 * diag(ionDensity))
            .rename("C6J1-l_h=" + str(harmonic))
            .withSkippingPattern()
        )

    # Add ion friction terms if ionFlux is passed
    if 2 in evolvedHarmonics and ionFlux is not None:

        for term in ["diffTermI2", "diffTermJ2", "dfdv2", "termLL2"]:

            eiColl.ddt[ionFlux] += normIonFriction * stencils.TermMomentStencil(
                2, 1, term
            )(distribution).rename(term + "Ion")

        for term in [
            "C1Il+2_h=2",
            "C1J-l-1_h=2",
            "C2Il_h=2",
            "C2J1-l_h=2",
            "C3Il+2_h=2",
            "C4J-l-1_h=2",
            "C5Il_h=2",
            "C6J1-l_h=2",
        ]:

            eiColl.ddt[ionFlux] += normIonFriction * stencils.TermMomentStencil(
                2, 1, term
            )(ionDensity).rename(term + "Ion")

    return eiColl


def eeCollHigherL(
    grid: Grid,
    textbook: derivations.Textbook,
    norms: Dict[str, float],
    distribution: Variable,
    elTemperature: Variable,
    elDensity: Variable,
    evolvedHarmonics: List[int],
) -> mc.Model:
    """Return e-e collision model for l>0 using matrix terms. Assumes default normalization.
    NOTE: Must be called separately for odd and even harmonics if staggered, and the fluid quantities used must reflect that (dual for odd harmonics).

    Args:
        grid (Grid): Used grid object
        textbook (Textbook): Textbook object used to retrieve built-in derivations
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)
        distribution (Variable): Electron distribution function - implicit
        elTemperature (Variable): Electron temperature - should live on the same grid as the evolved harmonics
        elDensity (Variable): Electron density - should live on the same grid as the evolved harmonics
        evolvedHarmonics (List[int]): List of evolved harmonics, must all correspond to either odd or even l numbers

    Returns:
        Model: Electron-electron collision operator model for l>0, ready for addition into a ReMKiT1D context
    """
    assert (
        1 not in evolvedHarmonics
    ), "addEECollHigherL cannot be used to evolve harmonic with index 1"

    vGrid = grid.vGrid
    lNums = grid.lGrid

    mbData = mc.VarlikeModelboundData()
    derivDistribution = distribution.dual if elDensity.isOnDualGrid else distribution
    mbData.addVar(
        derivations.HarmonicExtractorDerivation("f0", grid, 1)(derivDistribution)
    )

    mbData.addVar(
        Variable(
            "I0",
            grid,
            isSingleHarmonic=True,
            derivation=derivations.shkarofskyIIntegralDeriv("I0", 0),
            derivationArgs=["f0"],
        ).onDualGrid(elDensity.onDualGrid)
    )
    mbData.addVar(
        Variable(
            "I2",
            grid,
            isSingleHarmonic=True,
            derivation=derivations.shkarofskyIIntegralDeriv("I2", 2),
            derivationArgs=["f0"],
        ).onDualGrid(elDensity.onDualGrid)
    )
    mbData.addVar(
        Variable(
            "J-1",
            grid,
            isSingleHarmonic=True,
            derivation=derivations.shkarofskyJIntegralDeriv("J-1", -1),
            derivationArgs=["f0"],
        ).onDualGrid(elDensity.onDualGrid)
    )

    polyDeriv1 = derivations.PolynomialDerivation(
        "sumTerm",
        0.0,
        polyPowers=np.array([1.0, 1.0, 1.0]),
        polyCoeffs=np.array([-1.0, 2.0, 3.0]),
    )
    mbData.addVar(
        polyDeriv1(mbData["I2"], mbData["J-1"], mbData["I0"])
        .onDualGrid(elDensity.isOnDualGrid)
        .rename("IJSum")
    )
    mbData.addVar(textbook["logLee"](elTemperature, elDensity))

    df0 = derivations.DDVDerivation(
        "df0/dv",
        grid,
        1,
        vifAtZero=(
            1.0 / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
            -vGrid[0] ** 2 / vGrid[1] ** 2 / (1.0 - vGrid[0] ** 2 / vGrid[1] ** 2),
        ),
    )
    ddf0 = derivations.D2DV2Derivation("d2f0/dv2", grid, 1)

    mbData.addVar(
        df0(derivDistribution).onDualGrid(elDensity.isOnDualGrid).rename("df0")
    )
    mbData.addVar(
        ddf0(derivDistribution).onDualGrid(elDensity.isOnDualGrid).rename("ddf0")
    )

    eeModel = mc.Model("e-e_coll_l>0")
    eeModel.setModelboundData(mbData)
    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)

    # 8*pi*f0*fl term

    normConst = (
        8
        * np.pi
        * gamma0norm
        * norms["density"]
        * norms["time"]
        / norms["velGrid"] ** 3
    )

    diag = mc.DiagonalStencil(evolvedHarmonics=evolvedHarmonics)
    eeModel.ddt[distribution] += (
        normConst
        * (mbData["f0"] * mbData["logLee"])
        @ diag(distribution).rename("8pif0fl")
    )

    for harmonic in evolvedHarmonics:
        l = lNums[harmonic - 1]

        # velocity diffusion terms with fl

        v1 = grid.profile(np.array([1 / v for v in vGrid]), "V", "\\frac{1}{v}")

        normConst = (
            gamma0norm / 3 * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )

        velDiff = stencils.D2DV2Stencil(
            harmonic, harmonic, adfAtZero=(0, 0) if l > 1 else (1 / vGrid[1], 0)
        )

        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["I2"])
            @ (v1 * velDiff(distribution)).rename("diffTermI" + str(harmonic))
        )

        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["J-1"])
            @ (v1 * velDiff(distribution)).rename("diffTermJ" + str(harmonic))
        )

        # velocity deriv terms with fl

        v2 = grid.profile(np.array([1 / v**2 for v in vGrid]), "V", "\\frac{1}{v^2}")

        ddv = stencils.DDVStencil(harmonic, harmonic)

        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["IJSum"])
            @ (v2 * ddv(distribution)).rename("dfdv" + str(harmonic))
        )

    # -l(l+1)/2 terms
    hProfile = []
    for l in lNums:
        hProfile.append(l * (l + 1.0) / 2.0)
    h = grid.profile(np.array(hProfile), "H", "\\frac{l(l+1)}{2}")
    v3 = grid.profile(np.array([1 / v**3 for v in vGrid]), "V", "\\frac{1}{v^3}")
    normConst = (
        -gamma0norm / 3 * norms["density"] * norms["time"] / norms["velGrid"] ** 3
    )

    eeModel.ddt[distribution] += (
        normConst
        * (mbData["logLee"] * mbData["IJSum"])
        @ (h * (v3 * diag(distribution))).rename("termLL")
    )

    #     # I/J(f_l) terms

    for harmonic in evolvedHarmonics:
        l = lNums[harmonic - 1]

        # d2f0/dv2 terms
        # C1*I_{l+2} term
        C = (l + 1) * (l + 2) / ((2 * l + 1) * (2 * l + 3))  # C1

        normConst = (
            C
            * gamma0norm
            / 2
            * norms["density"]
            * norms["time"]
            / norms["velGrid"] ** 3
        )

        Il2 = stencils.ShkarofskyIStencil(harmonic, harmonic, l + 2)
        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["ddf0"])
            @ (v1 * Il2(distribution)).rename("C1Il+2_h=" + str(harmonic))
        )

        # C1*J_{-l-1} term
        Jl1 = stencils.ShkarofskyJStencil(harmonic, harmonic, -l - 1)

        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["ddf0"])
            @ (v1 * Jl1(distribution)).rename("C1J-l-1_h=" + str(harmonic))
        )

        # C2*I_l term
        C = -(l - 1) * l / ((2 * l + 1) * (2 * l - 1))  # C2
        normConst = (
            C
            * gamma0norm
            / 2
            * norms["density"]
            * norms["time"]
            / norms["velGrid"] ** 3
        )

        Il = stencils.ShkarofskyIStencil(harmonic, harmonic, l)
        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["ddf0"])
            @ (v1 * Il(distribution)).rename("C2Il_h=" + str(harmonic))
        )

        # C2*J1-l term
        J1l = stencils.ShkarofskyJStencil(harmonic, harmonic, 1 - l)
        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["ddf0"])
            @ (v1 * J1l(distribution)).rename("C2J1-l_h=" + str(harmonic))
        )

        # df0/dv terms

        # C3*I_{l+2} term
        C = -((l + 1) * l / 2 + l + 1) / ((2 * l + 1) * (2 * l + 3))  # C3
        normConst = (
            C * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )

        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["df0"])
            @ (v2 * Il2(distribution))
            .rename("C3Il+2_h=" + str(harmonic))
            .withSkippingPattern()
        )

        # C4*J_{-l-1} term
        C = (-(l + 1) * l / 2 + l + 2) / ((2 * l + 1) * (2 * l + 3))  # C4
        normConst = (
            C * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )

        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["df0"])
            @ (v2 * Jl1(distribution))
            .rename("C4J-l-1_h=" + str(harmonic))
            .withSkippingPattern()
        )

        # C5*I_l term
        C = ((l + 1) * l / 2 + l - 1) / ((2 * l + 1) * (2 * l - 1))  # C5
        normConst = (
            C * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )

        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["df0"])
            @ (v2 * Il(distribution))
            .rename("C5Il_h=" + str(harmonic))
            .withSkippingPattern()
        )

        # C6*J1-l term
        C = -((l + 1) * l / 2 - l) / ((2 * l + 1) * (2 * l - 1))  # C6
        normConst = (
            C * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3
        )

        eeModel.ddt[distribution] += (
            normConst
            * (mbData["logLee"] * mbData["df0"])
            @ (v2 * J1l(distribution))
            .rename("C6J1-l_h=" + str(harmonic))
            .withSkippingPattern()
        )

    return eeModel


def ampereMaxwellKineticElTerm(
    distribution: Variable, norms: Dict[str, float]
) -> mc.MatrixTerm:
    """Return kinetic electron contribution matrix term to Ampere-Maxwell equation for E-field. This uses a moment stencil, taking the appropriate moment of the l=1 harmonic. Assumes default normalization.

    Args:
        distribution (Variable): Electron distribution - implicit
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)

    Returns:
        MatrixTerm: Matrix term to be added to a model object
    """
    elCharge = 1.60218e-19
    epsilon0 = 8.854188e-12  # vacuum permittivity
    normConst = (
        elCharge
        / (3 * epsilon0)
        * norms["density"]
        * norms["time"]
        * norms["velGrid"]
        / norms["EField"]
    )

    return normConst * stencils.MomentStencil(1, 2)(distribution).rename("kin_AM_term")


def diffusiveHeatingTerm(
    grid: Grid,
    norms: Dict[str, float],
    distribution: Variable,
    density: Variable,
    heatingProfile: Profile,
    timeSignal: mc.TimeSignalData = mc.TimeSignalData(),
) -> mc.MatrixTerm:
    """Return diffusive kinetic electron heating matrix term with a given spatial heating profile. The term is proportional to 1/v**2 * d/dv(v**2 * df_0/dv) Assumes default normalization.

    Args:
        grid (Grid): Used grid object
        norms (Dict[str,float]): Normalisation dictionary (expects standard normalisation from RMKContext)
        distribution (Variable): Electron distribution - implicit
        density (Variable): Electron density
        heatingProfile (Profile): Spatial heating profile
        timeSignal (mc.TimeSignalData, optional): Optional time signal component. Defaults to mc.TimeSignalData() - constant signal.

    Returns:
        MatrixTerm: Heating matrix term ready for addition to ReMKiT1D model object
    """

    assert (
        heatingProfile.dim == "X"
    ), "heatingProfile in diffusiveHeatingTerm must be a spatial profile"
    elCharge = 1.60218e-19
    elMass = 9.10938e-31

    vGrid = grid.vGrid
    dv = grid.vWidths
    vBoundary = [vGrid[i] + dv[i] / 2 for i in range(len(dv))]

    v2 = grid.profile(np.array([1 / v**2 for v in vGrid]), "V", "\\frac{1}{v^2}")
    v2b = grid.profile(np.array([v**2 for v in vBoundary]), "V", "v^2")

    normConst = elCharge / (3 * elMass) * norms["eVTemperature"] / norms["velGrid"] ** 2

    stencil = stencils.D2DV2Stencil(1, 1, diffCoeff=v2b)

    return (
        normConst
        * density ** (-1)
        * (timeSignal * (heatingProfile * (v2 * stencil(distribution)))).rename(
            "diff_heating"
        )
    )


def logicalBCModel(
    grid: Grid,
    distribution: Variable,
    ionCurrent: Variable,
    density: Variable,
    densityDual: Union[Variable, None] = None,
    densityOnBoundary: Union[Variable, None] = None,
    totalCurrent: Union[Variable, None] = None,
    bisTol: float = 1e-12,
    leftBoundary=False,
    evolvedHarmonics: Optional[List[int]] = None,
) -> mc.Model:
    """Return logical boundary condition model using matrix terms. Assumes default normalization. Extrapolates the distribution function to the boundary using a supplied extrapolation rule, and allows for a non-zero current through the sheath. Should not be called to evolve odd harmonics if the distribution is staggered.

    Args:
        grid (Grid): Used grid object
        distribution (Variable): Electron distribution - implicit
        ionCurrent (Variable): Scalar ion current variable at the boundary (used to match ion and electron fluxes)
        density (Variable): Density variable
        densityDual (Union[Variable,None], optional): Dual density variable. Defaults to None, assuming non-staggered variables.
        densityOnBoundary (Union[Variable,None], optional): Density on the boundary used for extrapolation. Defaults to None - extrapolating to the last cell centre before the boundary.
        totalCurrent (Union[Variable,None], optional): Total current variable. Defaults to None, setting the current to 0.
        bisTol (float, optional): Bisection tolerance for determining the cut-off velocity. Defaults to 1e-12.
        leftBoundary (bool, optional): Set to true if this is the left boundary. Defaults to False.
        evolvedHarmonics (Optional[List[int]], optional): List of evolved harmonics. Defaults to None, evolving all harmonics.

    Returns:
        Model: Logical boundary condition model ready for addition into a ReMKiT1D context
    """
    mbData = mc.LBCModelboundData(
        grid,
        ionCurrent,
        distribution,
        density,
        densityDual,
        densityOnBoundary,
        totalCurrent,
        bisTol,
        leftBoundary,
    )
    lbcModel = mc.Model("lbc_" + ("L" if leftBoundary else "R"))
    lbcModel.setModelboundData(mbData)

    usedHarmonics = list(range(1, grid.numH + 1))
    if evolvedHarmonics is not None:
        usedHarmonics = evolvedHarmonics

    lGrid = grid.lGrid

    oddL = [l % 2 == 1 for l in lGrid]
    oddLHarmonics = [i + 1 for i, x in enumerate(oddL) if x]
    evenLHarmonics = [i + 1 for i, x in enumerate(oddL) if not x]

    for h in usedHarmonics:
        l = lGrid[h - 1]

        # l-1 harmonic

        if l > 0:
            if leftBoundary:
                normConst = l / ((2 * l - 1))

                stencil = stencils.LBCStencil(
                    h,
                    grid.getH(l - 1),
                    distribution,
                    density,
                    densityDual,
                    densityOnBoundary,
                    leftBoundary=True,
                )

                lbcModel.ddt[distribution] += normConst * stencil(distribution).rename(
                    "lbcMinus" + str(h)
                )

            else:
                normConst = -l / ((2 * l - 1))

                stencil = stencils.LBCStencil(
                    h,
                    grid.getH(l - 1),
                    distribution,
                    density,
                    densityDual,
                    densityOnBoundary,
                    leftBoundary=False,
                    decompHarmonics=cast(List, oddLHarmonics),
                )

                lbcModel.ddt[distribution] += normConst * stencil(distribution).rename(
                    "lbcMinus_odd" + str(h)
                )

                stencil = stencils.LBCStencil(
                    h,
                    grid.getH(l - 1),
                    distribution,
                    density,
                    densityDual,
                    densityOnBoundary,
                    leftBoundary=False,
                    decompHarmonics=cast(List, evenLHarmonics),
                )

                lbcModel.ddt[distribution] += normConst * stencil(distribution).rename(
                    "lbcMinus_even" + str(h)
                )

        # l+1 harmonic
        if l < max(lGrid):
            if leftBoundary:
                normConst = (l + 1) / ((2 * l + 3))

                stencil = stencils.LBCStencil(
                    h,
                    grid.getH(l + 1),
                    distribution,
                    density,
                    densityDual,
                    densityOnBoundary,
                    leftBoundary=True,
                )

                lbcModel.ddt[distribution] += normConst * stencil(distribution).rename(
                    "lbcPlus" + str(h)
                )

            else:
                normConst = -(l + 1) / ((2 * l + 3))

                stencil = stencils.LBCStencil(
                    h,
                    grid.getH(l + 1),
                    distribution,
                    density,
                    densityDual,
                    densityOnBoundary,
                    leftBoundary=False,
                    decompHarmonics=cast(List, oddLHarmonics),
                )

                lbcModel.ddt[distribution] += normConst * stencil(distribution).rename(
                    "lbcPlus_odd" + str(h)
                )

                stencil = stencils.LBCStencil(
                    h,
                    grid.getH(l + 1),
                    distribution,
                    density,
                    densityDual,
                    densityOnBoundary,
                    leftBoundary=False,
                    decompHarmonics=cast(List, evenLHarmonics),
                )

                lbcModel.ddt[distribution] += normConst * stencil(distribution).rename(
                    "lbcPlus_even" + str(h)
                )

    return lbcModel


def dvEnergyTerm(grid: Grid, distribution: Variable, k: int = 0) -> mc.MatrixTerm:
    """Return velocity space drag-like heating/cooling matrix term: proportional to 1/v**2 * d/dv(Df) where D is a velocity space vector proportional to v**k * dv, with k set by the user, controlling which velocity cells get the bulk of the energy source. Assumes default normalization. In order to use this term, it needs to be multiplied by the heating/cooling rate. **NOTE**: The rate should be positive for cooling and negative for heating

    Args:
        grid (Grid): Used grid object.
        distribution (Variable): Distribution variable - implicit
        k (int, optional): Diffusion coefficient velocity power. Defaults to 0.

    Returns:
        MatrixTerm: Kinetic drag-like energy source/sing term (needs to be multiplied by the rate)
    """

    vGrid = grid.vGrid
    dv = grid.vWidths
    interp = grid.profile(np.zeros(len(dv)), "V")

    vProfile = grid.profile(
        np.array([1 / (v**2) for v in vGrid]), "V", "\\frac{1}{v^2}"
    )

    drag = dv * np.ones(len(vGrid))
    vSum = vGrid**k * np.zeros(
        len(drag)
    )  # ones if exact energy source is required, 0 if exactly no particle source is required (either way the error is negligible)
    vSum[:-1] = vGrid[:-1] ** (2 + k) / (vGrid[1:] ** 2 - vGrid[:-1] ** 2)
    drag = drag * vSum

    ddv = stencils.DDVStencil(1, 1, grid.profile(drag, "V"), interpCoeffs=interp)

    return vProfile * ddv(distribution).rename("drag_heating")


def standardBaseFluid(
    species: Species,
    density: Variable,
    flux: Variable,
    flowSpeed: Variable,
    temperature: Variable,
    eField: Variable,
    energyDensity: Optional[Variable] = None,
    heatflux: Optional[Variable] = None,
    viscosity: Optional[Variable] = None,
    viscosityLimitMult: Optional[Variable] = None,
) -> mc.Model:
    """Generates a standard base fluid model for a species with given variables.

    Variables passed into this function should be defined on the regular grid and have dual grid variables.

    The model will include the continuity equation, momentum equation, and energy equation (if energyDensity present), with the default reflective boundary conditions (should be specified in a separate model).

    The implicit temperature calculation is also added if the energyDensity is present.

    No heat flux, viscosity, or sources are added.

    If the heatflux variable is present it will be assumed to be stationary and will have its identity term and the corresponding bulk divergence terms in the energy equation added.

    If the viscosity variable is present it will be assumed to be stationary and will have its identity term and the corresponding bulk divergence terms in the momentum and energy equation added. If viscosity limiter is present it is assumed to simply multiply the viscosity.

    Assumes default norms.

    Args:
        species (Species): Species for which the base model is generated
        density (Variable): Density variable
        flux (Variable): Particle flux variable
        flowSpeed (Variable): Flow speed variable corresponding to flux/density
        temperature (Variable): Temperature variable
        eField (Variable): Electric field (used only for charged particles). Must have both a regular and dual grid equivalent.
        energyDensity (Optional[Variable], optional): Energy density variable (total). Defaults to None, not adding the energy equation.
        heatflux (Optional[Variable], optional): Heatflux variable, if present adds its identity term. Defaults to None.
        viscosity (Optional[Variable], optional): Viscosity variable, if present adds its identity term. Defaults to None.
        viscosityLimitMult (Optional[Variable], optional): Multiplicative variable for the viscosity limiter. Defaults to None.

    Returns:
        Model: Base fluid model ready to be extended or added to ReMKiT1D context
    """

    assert (
        isinstance(density.dual, Variable) and not density.isOnDualGrid
    ), "standardBaseFluid model argument 'density' must live on the regular grid and have a .dual Variable on the dual grid"

    assert (
        isinstance(flux.dual, Variable) and not flux.isOnDualGrid
    ), "standardBaseFluid model argument 'flux' must live on the regular grid and have a .dual Variable on the dual grid"

    assert (
        isinstance(flowSpeed.dual, Variable) and not flowSpeed.isOnDualGrid
    ), "standardBaseFluid model argument 'flowSpeed' must live on the regular grid and have a .dual Variable on the dual grid"

    assert (
        isinstance(temperature.dual, Variable) and not temperature.isOnDualGrid
    ), "standardBaseFluid model argument 'temperature' must live on the regular grid and have a .dual Variable on the dual grid"

    assert (
        isinstance(eField.dual, Variable) and not eField.isOnDualGrid
    ), "standardBaseFluid model argument 'eField' must live on the regular grid and have a .dual Variable on the dual grid"

    if energyDensity is not None:
        assert (
            isinstance(energyDensity.dual, Variable) and not energyDensity.isOnDualGrid
        ), "standardBaseFluid model argument 'energyDensity' must live on the regular grid and have a .dual Variable on the dual grid"

    if heatflux is not None:
        assert (
            isinstance(heatflux.dual, Variable) and not heatflux.isOnDualGrid
        ), "standardBaseFluid model argument 'heatflux' must live on the regular grid and have a .dual Variable on the dual grid"

    if viscosity is not None:
        assert (
            isinstance(viscosity.dual, Variable) and not viscosity.isOnDualGrid
        ), "standardBaseFluid model argument 'viscosity' must live on the regular grid and have a .dual Variable on the dual grid"

    if viscosityLimitMult is not None:
        assert (
            isinstance(viscosityLimitMult.dual, Variable)
            and not viscosityLimitMult.isOnDualGrid
        ), "standardBaseFluid model argument 'viscosityLimitMult' must live on the regular grid and have a .dual Variable on the dual grid"

    newModel = mc.Model("fluidBase_" + species.name)
    elMass = 9.10938e-31
    amu = 1.6605390666e-27  # atomic mass unit
    speciesMass = species.atomicA * amu
    massRatio = elMass / speciesMass

    # Continuity equation flux divergence
    newModel.ddt[density] += -stencils.StaggeredDivStencil()(
        cast(Variable, flux.dual)
    ).rename("cont_divFlux")

    # Momentum equation pressure gradient

    newModel.ddt[cast(Variable, flux.dual)] += (
        -massRatio
        / 2
        * stencils.StaggeredGradStencil()(temperature * density).rename(
            "momentum_gradPressure"
        )
    )

    # Momentum advection

    newModel.ddt[cast(Variable, flux.dual)] += -stencils.CentralDiffDivStencil(
        cast(Variable, flowSpeed.dual)
    )(cast(Variable, flux.dual)).rename("momentum_divFlux")

    diag = mc.DiagonalStencil()
    # Lorentz force
    if abs(species.charge) > 1e-6:  # Species effectively neutral below this

        newModel.ddt[cast(Variable, flux.dual)] += (
            species.charge
            * massRatio
            * density.dual
            * diag(cast(Variable, eField.dual)).rename("momentum_lorentzForce")
        )

    if energyDensity is not None:
        # Implicit temperature calculation

        # Identity term
        newModel.ddt[temperature] += -diag(temperature).rename("temperature_identity")

        # 2/3 W/n term

        newModel.ddt[temperature] += (
            2 / 3 * density ** (-1) * diag(energyDensity).rename("temperature_w")
        )

        # kinetic energy term

        newModel.ddt[temperature] += (
            -2
            / (3 * massRatio)
            * diag(
                cast(Variable, flux.dual) ** 2 / cast(Variable, density.dual) ** 2
            ).rename("temperature_U2")
        )

        # Energy advection

        newModel.ddt[energyDensity] += -stencils.StaggeredDivStencil()(
            cast(Variable, energyDensity.dual)
            * cast(Variable, flux.dual)
            / cast(Variable, density.dual)
        ).rename("energy_wAdv")

        # Pressure advection

        newModel.ddt[energyDensity] += -stencils.StaggeredDivStencil()(
            cast(Variable, temperature.dual) * cast(Variable, flux.dual)
        ).rename("energy_pAdv")

        # Lorentz Force work
        if abs(species.charge) > 1e-6:  # Species effectively neutral below this charge

            newModel.ddt[energyDensity] += (
                2
                * species.charge
                * diag(cast(Variable, flux.dual) * cast(Variable, eField.dual)).rename(
                    "energy_lorentzWork"
                )
            )

        # Heatflux terms

        if heatflux is not None:
            # Identity term

            newModel.ddt[cast(Variable, heatflux.dual)] += -diag(
                cast(Variable, heatflux.dual)
            ).rename("heatflux_identity")

            # Heatflux divergence in energy equation

            newModel.ddt[energyDensity] += -stencils.StaggeredDivStencil()(
                cast(Variable, heatflux.dual)
            ).rename("energy_divq")

    # Viscosity
    if viscosity is not None:
        # Identity term

        newModel.ddt[viscosity] += -diag(viscosity).rename("viscosity_identity")

        # Viscosity divergence in momentum equation

        if viscosityLimitMult is not None:

            newModel.ddt[cast(Variable, flux.dual)] += (
                -massRatio
                / 2
                * stencils.StaggeredDivStencil()(viscosityLimitMult * viscosity).rename(
                    "momentum_divpi"
                )
            )

        else:

            newModel.ddt[cast(Variable, flux.dual)] += (
                -massRatio
                / 2
                * stencils.StaggeredDivStencil()(viscosity).rename("momentum_divpi")
            )

        if energyDensity is not None:
            # Viscosity advection/heating in energy equation

            if viscosityLimitMult is not None:

                newModel.ddt[energyDensity] += -stencils.StaggeredDivStencil()(
                    cast(Variable, viscosityLimitMult.dual)
                    * cast(Variable, viscosity.dual)
                    * cast(Variable, flux.dual)
                    / cast(Variable, density.dual)
                ).rename("energy_divpiu")

            else:

                newModel.ddt[energyDensity] += -stencils.StaggeredDivStencil()(
                    cast(Variable, viscosity.dual)
                    * cast(Variable, flux.dual)
                    / cast(Variable, density.dual)
                ).rename("energy_divpiu")

    return newModel


def bohmBoundaryModel(
    species: Species,
    density: Variable,
    flux: Variable,
    flowSpeed: Variable,
    temperature: Variable,
    sonicSpeed: Variable,
    energyDensity: Optional[Variable] = None,
    sheathGamma: Optional[Variable] = None,
    boundaryFlowSpeed: Optional[Variable] = None,
    viscosity: Optional[Variable] = None,
    viscosityLimitMult: Optional[Variable] = None,
    leftBoundary=False,
) -> mc.Model:
    """Adds Bohm outflow boundary conditions on the continuity, momentum, and energy equations for a given species. Assumes default norms.

    Args:
        species (Species): Species for which the model is generated
        density (Variable): Density variable
        flux (Variable): Particle flux variable
        flowSpeed (Variable): Flow speed variable corresponding to flux/density
        temperature (Variable): Temperature variable
        sonicSpeed (Variable): Sonic speed corresponding to the outflow Bohm speed
        energyDensity (Optional[Variable], optional): Energy density variable (total). Defaults to None, not adding the energy equation contribution.
        sheathGamma (Optional[Variable], optional): Sheath heat transmission coefficient. Defaults to None, but is required if energyDensity is not None.
        boundaryFlowSpeed (Optional[Variable], optional): Boundary flow speed (scalar) used for kinetic energy outflow BC. Defaults to None, omitting the explicit kinetic energy outflow.
        viscosity (Optional[Variable], optional): Viscosity variable, if present adds viscous heating due to the viscosity boundary condition. Defaults to None.
        viscosityLimitMult (Optional[Variable], optional): Multiplicative viscosity limiter. Defaults to None, omitting the limiter.
        leftBoundary (bool, optional): Set to true if this is the left boundary. Defaults to False.

    Returns:
        Model: Bohm boundary condition model for the given species
    """
    newModel = mc.Model("bohmBoundary_" + species.name)
    elMass = 9.10938e-31
    amu = 1.6605390666e-27  # atomic mass unit
    speciesMass = species.atomicA * amu
    massRatio = elMass / speciesMass

    # Continuity BC

    bcDiv = stencils.BCDivStencil(flowSpeed, sonicSpeed, isLeft=leftBoundary)

    newModel.ddt[density] += -bcDiv(density).rename("continuity_Bohm")

    # Momentum BC

    newModel.ddt[cast(Variable, flux.dual)] += -bcDiv(cast(Variable, flux.dual)).rename(
        "momentum_Bohm"
    )

    if energyDensity is not None:

        assert (
            sheathGamma is not None
        ), "sheathGamma must be present in bohmBoundaryModel if energyDensity is evolved"
        # Energy BC

        newModel.ddt[energyDensity] += -sheathGamma * bcDiv(
            temperature * density
        ).rename("energy_BCGamma")

        # Kinetic energy BC

        if boundaryFlowSpeed is not None:

            newModel.ddt[energyDensity] += (
                -(massRatio ** (-1))
                * boundaryFlowSpeed**2
                * bcDiv(density).rename("energy_BCKin")
            )

        # Viscous heating BC

        if viscosity is not None:

            if viscosityLimitMult is not None:

                newModel.ddt[energyDensity] += -bcDiv(
                    viscosityLimitMult * viscosity
                ).rename("energy_BCVisc")

            else:

                newModel.ddt[energyDensity] += -bcDiv(viscosity).rename("energy_BCVisc")

    return newModel
