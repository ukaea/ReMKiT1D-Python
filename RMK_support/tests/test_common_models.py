from RMK_support.grid import Grid, Profile
import RMK_support.common_models as cm
import RMK_support.derivations as dv
import RMK_support.model_construction as mc
import RMK_support.sk_normalization as sk
import RMK_support.stencils as stencils
import RMK_support.variable_container as vc
import numpy as np
import pytest


elCharge = 1.60218e-19
elMass = 9.10938e-31
amu = 1.6605390666e-27  # atomic mass unit
ionMass = 2.014 * amu  # deuterium mass
epsilon0 = 8.854188e-12  # vacuum permittivity
heavySpeciesMass = 2.014  # in amus


@pytest.fixture
def grid():
    return Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        1,
        0,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )


@pytest.fixture
def norms() -> dict:
    return sk.calculateNorms(10.0, 1e19, 1)


def test_simple_source_term(grid: Grid):

    sourceProfile = Profile(np.ones(grid.numX), dim="X")

    timeSignal = mc.TimeSignalData()

    var = vc.Variable("evolvedVar", grid)

    sourceTerm = cm.simpleSourceTerm(var, sourceProfile, timeSignal)
    sourceTerm.evolvedVar = var

    assert (
        sourceTerm.dict()
        == (
            var**-1
            * mc.MatrixTerm(
                "custom",
                mc.DiagonalStencil(),
                evolvedVar=var,
                implicitVar=var,
                profiles={"X": sourceProfile},
                T=timeSignal,
            )
        ).dict()
    )

    # Bad case - profile must be in X

    with pytest.raises(AssertionError) as e_info:
        cm.simpleSourceTerm(var, Profile(np.ones(grid.numV), dim="V"), timeSignal)
    assert e_info.value.args[0] == "simpleSourceTerm requires a spatial source profile"


def test_advection(grid: Grid):

    n, n_dual = vc.varAndDual("n", grid)
    G, G_dual = vc.varAndDual("G", grid)

    newModel = cm.advection(n, G_dual)

    bulkDiv = -mc.MatrixTerm(
        "bulk_div",
        stencils.StaggeredDivStencil(),
        evolvedVar=n,
        implicitVar=G_dual,
    )

    assert newModel.dict() == {
        "type": "customModel",
        "termTags": [bulkDiv.name],
        "termGenerators": {"tags": []},
        bulkDiv.name: bulkDiv.dict(),
    }

    # Using flux on regular grid

    newModel = cm.advection(n, G)

    centralDiv = -mc.MatrixTerm(
        "bulk_div",
        stencils.CentralDiffDivStencil(),
        evolvedVar=n,
        implicitVar=G,
    )

    assert newModel.dict() == {
        "type": "customModel",
        "termTags": [centralDiv.name],
        "termGenerators": {"tags": []},
        centralDiv.name: centralDiv.dict(),
    }

    # Using outflow boundary condition (zero lower bound)

    u, u_dual = vc.varAndDual("u", grid)

    divBCLeft = -mc.MatrixTerm(
        "div_BC_left",
        stencils.BCDivStencil(u, isLeft=True),
        evolvedVar=n,
        implicitVar=n,
    )

    divBCRight = -mc.MatrixTerm(
        "div_BC_right",
        stencils.BCDivStencil(u),
        evolvedVar=n,
        implicitVar=n,
    )

    for leftOutflow in [False, True]:

        newModel = cm.advection(
            n, G_dual, outflow=(leftOutflow, True), advectionSpeed=u
        )

        result = {
            "type": "customModel",
            "termTags": [bulkDiv.name, divBCRight.name],
            "termGenerators": {"tags": []},
            bulkDiv.name: bulkDiv.dict(),
            divBCRight.name: divBCRight.dict(),
        }

        if leftOutflow:
            result["termTags"] = [bulkDiv.name, divBCLeft.name, divBCRight.name]
            result[divBCLeft.name] = divBCLeft.dict()

        assert newModel.dict() == result

    # Bad cases

    # If flux is a MultiplicativeArgument (e.g. product of density and flow speed)
    # then its scalar multiplier must equal 1

    tempG_dual = 0.5 * n_dual * u_dual

    with pytest.raises(AssertionError) as e_info:
        cm.advection(n, tempG_dual)
    assert (
        e_info.value.args[0]
        == "flux cannot have non-trivial scalar multiplier in advection"
    )

    # If flux is a MultiplicativeArgument, all its components must live on the same grid
    tempG = n_dual * u

    with pytest.raises(AssertionError) as e_info:
        cm.advection(n, tempG)
    assert (
        e_info.value.args[0]
        == "If flux in advection is a MultiplicativeArgument all components must live on the same grid"
    )

    # Outflow BC without advection speed

    with pytest.raises(AssertionError) as e_info:
        cm.advection(n, G_dual, outflow=(False, True))
    assert (
        e_info.value.args[0]
        == "advectionSpeed on the regular grid must be provided to advection if there is any outflow"
    )

    # Outflow advection speed must be on regular grid

    with pytest.raises(AssertionError) as e_info:
        cm.advection(n, G_dual, outflow=(False, True), advectionSpeed=u_dual)
    assert (
        e_info.value.args[0]
        == "advectionSpeed in advection model must be on regular grid"
    )


def test_pressure_grad(grid: Grid):

    G, G_dual = vc.varAndDual("G", grid)
    P, P_dual = vc.varAndDual("P", grid)
    normConst = 2.0

    newModel = cm.pressureGrad(G_dual, P, normConst)

    bulkGrad = -normConst * mc.MatrixTerm(
        "bulk_grad",
        stencils.StaggeredGradStencil(),
        evolvedVar=G_dual,
        implicitVar=P,
    )

    assert newModel.dict() == {
        "type": "customModel",
        "termTags": [bulkGrad.name],
        "termGenerators": {"tags": []},
        bulkGrad.name: bulkGrad.dict(),
    }

    # Using flux on regular (non-periodic) grid

    newModel = cm.pressureGrad(G, P, normConst)

    bulkGradReg = -normConst * mc.MatrixTerm(
        "bulk_grad",
        stencils.CentralDiffGradStencil(),
        evolvedVar=G,
        implicitVar=P,
    )

    gradBCLeft = -normConst * mc.MatrixTerm(
        "grad_BC_left",
        stencils.BCGradStencil(isLeft=True),
        evolvedVar=G,
        implicitVar=P,
    )

    gradBCRight = -normConst * mc.MatrixTerm(
        "grad_BC_right",
        stencils.BCGradStencil(),
        evolvedVar=G,
        implicitVar=P,
    )

    assert newModel.dict() == {
        "type": "customModel",
        "termTags": [bulkGradReg.name, gradBCLeft.name, gradBCRight.name],
        "termGenerators": {"tags": []},
        bulkGradReg.name: bulkGradReg.dict(),
        gradBCLeft.name: gradBCLeft.dict(),
        gradBCRight.name: gradBCRight.dict(),
    }

    # Bad cases

    # Pressure must live on regular grid

    with pytest.raises(AssertionError) as e_info:
        cm.pressureGrad(G_dual, P_dual, normConst)
    assert e_info.value.args[0] == "pressure in pressureGrad must be on regular grid"

    # If pressure is a MultiplicativeArgument (e.g. product of density and temperature)
    # then its scalar multiplier must equal 1

    tempP = 1.5 * vc.Variable("n", grid) * vc.Variable("T", grid)

    with pytest.raises(AssertionError) as e_info:
        cm.pressureGrad(G_dual, tempP, normConst)
    assert (
        e_info.value.args[0]
        == "pressure cannot have non-trivial scalar multiplier in pressureGrad"
    )

    # If pressure is a MultiplicativeArgument, all its args must live on the regular grid

    tempP_dual = vc.Variable("n_dual", grid, isOnDualGrid=True) * vc.Variable(
        "T_dual", grid, isOnDualGrid=True
    )

    with pytest.raises(AssertionError) as e_info:
        cm.pressureGrad(G_dual, tempP_dual, normConst)
    assert (
        e_info.value.args[0]
        == "If pressure in pressureGrad is a MultiplicativeArgument all components must live on the regular grid"
    )


def test_ampere_maxwell(grid: Grid, norms: dict):

    E = vc.Variable("E", grid)

    Ge = vc.Variable("Ge", grid)
    Gi = vc.Variable("Gi", grid)

    e = dv.Species("e", 0, charge=-1, associatedVars=[Ge])
    ion = dv.Species("i", -1, charge=+1, associatedVars=[Gi])

    amModel = cm.ampereMaxwell(E, [Ge, Gi], [e, ion], norms)

    result = {
        "type": "customModel",
        "termTags": [],
        "termGenerators": {"tags": []},
    }

    normConst = (
        elCharge
        / epsilon0
        * norms["density"]
        * norms["time"]
        * norms["speed"]
        / norms["EField"]
    )

    termTags = []

    for i, flux in enumerate([Ge, Gi]):
        current = f"current_{flux.name}"

        termTags.append(current)

        species = [e, ion][i]

        amTerm = (
            -species.charge
            * normConst
            * mc.MatrixTerm(
                current,
                mc.DiagonalStencil(),
                evolvedVar=E,
                implicitVar=flux,
            )
        )

        result[current] = amTerm.dict()

    result["termTags"] = termTags

    assert amModel.dict() == result


def test_lorentz_force(grid: Grid, norms: dict):

    E = vc.Variable("E", grid)

    ne = vc.Variable("ne", grid)
    ni = vc.Variable("ni", grid)

    Ge = vc.Variable("Ge", grid)
    Gi = vc.Variable("Gi", grid)

    e = dv.Species("e", 0, atomicA=elMass / amu, charge=-1, associatedVars=[ne, Ge])
    ion = dv.Species(
        "i", -1, atomicA=heavySpeciesMass, charge=+1, associatedVars=[ni, Gi]
    )

    lForceModel = cm.lorentzForces(E, [Ge, Gi], [ne, ni], [e, ion], norms)

    result = {
        "type": "customModel",
        "termTags": [],
        "termGenerators": {"tags": []},
    }

    termTags = []

    for i, species in enumerate([e, ion]):

        density = [ne, ni][i]
        flux = [Ge, Gi][i]

        tag = f"lorentz_{flux.name}"
        termTags.append(tag)

        lForceNormConst = (
            elCharge
            * species.charge
            / (amu * species.atomicA)
            * norms["EField"]
            * norms["time"]
            / norms["speed"]
        )

        lForceTerm = (
            lForceNormConst
            * density
            * mc.MatrixTerm(
                tag,
                mc.DiagonalStencil(),
                evolvedVar=flux,
                implicitVar=E,
            )
        )

        result[tag] = lForceTerm.dict()

    result["termTags"] = termTags

    assert lForceModel.dict() == result


def test_lorentz_force_work(grid: Grid, norms: dict):

    E = vc.Variable("E", grid)

    We = vc.Variable("We", grid)
    Wi = vc.Variable("Wi", grid)

    Ge = vc.Variable("Ge", grid)
    Gi = vc.Variable("Gi", grid)

    e = dv.Species("e", 0, atomicA=elMass / amu, charge=-1, associatedVars=[We, Ge])
    ion = dv.Species(
        "i", -1, atomicA=heavySpeciesMass, charge=+1, associatedVars=[Wi, Gi]
    )

    termTags = []

    lWorkModel = mc.Model("lorentz_force_work")

    for i, species in enumerate([e, ion]):

        energy = [We, Wi][i]
        flux = [Ge, Gi][i]

        tag = f"lorentz_work_{energy.name}"
        termTags.append(tag)

        lWorkNormConst = (
            species.charge
            * norms["EField"]
            * norms["time"]
            * norms["speed"]
            / norms["eVTemperature"]
        )

        lWorkModel.ddt[energy] += lWorkNormConst * mc.DiagonalStencil()(
            flux * E
        ).rename(tag)

    assert (
        cm.lorentzForceWork(E, [Ge, Gi], [We, Wi], [e, ion], norms).dict()
        == lWorkModel.dict()
    )


def test_implicit_temperature(grid: Grid, norms: dict):

    ne = vc.Variable("ne", grid)
    ni = vc.Variable("ni", grid)

    Ge = vc.Variable("Ge", grid)
    Gi = vc.Variable("Gi", grid)

    We = vc.Variable("We", grid)
    Wi = vc.Variable("Wi", grid)

    Te = vc.Variable("Te", grid, isStationary=True)
    Ti = vc.Variable("Ti", grid, isStationary=True)

    e = dv.Species("e", 0, atomicA=elMass / amu, charge=-1, associatedVars=[ne, Ge])
    ion = dv.Species(
        "i", -1, atomicA=heavySpeciesMass, charge=+1, associatedVars=[ni, Gi]
    )

    # Use 1 degree of freedom to avoid irrational numbers in constants
    dof = 1

    diag = mc.DiagonalStencil()

    # If specified, the kinetic energy term(s) can be set to evolve T only at certain X grid cells
    evolvedXCells = [1]
    diagU2 = mc.DiagonalStencil(evolvedXCells=evolvedXCells)

    kwargs = {
        "degreesOfFreedom": dof,
        "evolvedXU2Cells": evolvedXCells,
    }

    newModel = cm.implicitTemperatures(
        [We, Wi], [ne, ni], [Te, Ti], [e, ion], norms, [Ge, Gi], **kwargs
    )

    # Electron terms

    identity_Te = -mc.MatrixTerm(
        "identity_Te",
        diag,
        evolvedVar=Te,
        implicitVar=Te,
    )

    W_term_Te = (
        (2 / dof)
        * (ne**-1)
        * mc.MatrixTerm(
            "W_term_Te",
            diag,
            evolvedVar=Te,
            implicitVar=We,
        )
    )

    normU2e = (
        -amu * e.atomicA / (3 * elCharge) * norms["speed"] ** 2 / norms["eVTemperature"]
    )

    U2_term_Te = (
        normU2e
        * (ne**-2)
        * Ge
        * mc.MatrixTerm(
            "U2_term_Te",
            diagU2,
            evolvedVar=Te,
            implicitVar=Ge,
        )
    )

    # Ion terms

    identity_Ti = -mc.MatrixTerm(
        "identity_Ti",
        diag,
        evolvedVar=Ti,
        implicitVar=Ti,
    )

    W_term_Ti = (
        (2 / dof)
        * (ni**-1)
        * mc.MatrixTerm(
            "W_term_Ti",
            diag,
            evolvedVar=Ti,
            implicitVar=Wi,
        )
    )

    normU2i = (
        -amu
        * ion.atomicA
        / (3 * elCharge)
        * norms["speed"] ** 2
        / norms["eVTemperature"]
    )

    U2_term_Ti = (
        normU2i
        * (ni**-2)
        * Gi
        * mc.MatrixTerm(
            "U2_term_Ti",
            diagU2,
            evolvedVar=Ti,
            implicitVar=Gi,
        )
    )

    assert newModel.dict() == {
        "type": "customModel",
        "termGenerators": {"tags": []},
        "termTags": [
            identity_Te.name,
            W_term_Te.name,
            U2_term_Te.name,
            identity_Ti.name,
            W_term_Ti.name,
            U2_term_Ti.name,
        ],
        identity_Te.name: identity_Te.dict(),
        W_term_Te.name: W_term_Te.dict(),
        U2_term_Te.name: U2_term_Te.dict(),
        identity_Ti.name: identity_Ti.dict(),
        W_term_Ti.name: W_term_Ti.dict(),
        U2_term_Ti.name: U2_term_Ti.dict(),
    }

    # Bad cases

    # Temperature variable(s) not stationary

    TeTest = vc.Variable("Te", grid, isStationary=False)

    with pytest.raises(AssertionError) as e_info:
        cm.implicitTemperatures([We], [ne], [TeTest], [e], norms, [Ge], **kwargs)
    assert (
        e_info.value.args[0]
        == "Temperatures in implicitTemperatures are expected to be stationary"
    )


def test_kinetic_advection(grid: Grid):
    f = vc.Variable("f", grid, isDistribution=True)

    result = {
        "type": "customModel",
        "termTags": [],
        "termGenerators": {"tags": []},
    }

    termTags = []

    lNums = [grid.lGrid[i - 1] for i in range(1, grid.numH + 1)]
    mNums = [grid.mGrid[i - 1] for i in range(1, grid.numH + 1)]

    vProfile = grid.profile(grid.vGrid, dim="V")

    # By default, the kinAdvX model evolves distribution f at all harmonics

    evolvedHarmonics = list(range(1, grid.numH + 1))

    for harmonic in evolvedHarmonics:

        if lNums[harmonic - 1] > 0:
            normConst = -(lNums[harmonic - 1] - mNums[harmonic - 1]) / (
                2.0 * lNums[harmonic - 1] - 1.0
            )

            tag = f"adv_minus_{harmonic}"

            result[tag] = (
                (
                    normConst
                    * mc.MatrixTerm(
                        tag,
                        stencil=stencils.DistGradStencil(
                            harmonic,
                            grid.getH(
                                lNum=lNums[harmonic - 1] - 1,
                                mNum=mNums[harmonic - 1],
                                im=grid.imaginaryHarmonic[harmonic - 1],
                            ),
                        ),
                        evolvedVar=f,
                        implicitVar=f,
                        profiles={"V": vProfile},
                    )
                )
                .withFixedMatrix()
                .dict()
            )

            termTags.append(tag)

        if lNums[harmonic - 1] < grid.lMax:
            normConst = -(lNums[harmonic - 1] + mNums[harmonic - 1] + 1.0) / (
                2.0 * lNums[harmonic - 1] + 3.0
            )

            tag = f"adv_plus_{harmonic}"

            result[tag] = (
                (
                    normConst
                    * mc.MatrixTerm(
                        tag,
                        stencil=stencils.DistGradStencil(
                            harmonic,
                            grid.getH(
                                lNum=lNums[harmonic - 1] + 1,
                                mNum=mNums[harmonic - 1],
                                im=grid.imaginaryHarmonic[harmonic - 1],
                            ),
                        ),
                        evolvedVar=f,
                        implicitVar=f,
                        profiles={"V": vProfile},
                    )
                )
                .withFixedMatrix()
                .dict()
            )

            termTags.append(tag)

    result["termTags"] = termTags

    assert cm.kinAdvX(f, grid).dict() == result

    # 1st harmonic only

    harmonic = 1

    result = dict(
        {
            "type": "customModel",
            "termTags": [],
            "termGenerators": {"tags": []},
        }
    )

    tag = f"adv_plus_{harmonic}"

    result[tag] = (
        (
            (-1.0 / 3.0)
            * mc.MatrixTerm(
                tag,
                stencil=stencils.DistGradStencil(
                    harmonic,
                    grid.getH(
                        lNum=lNums[0] + 1,
                        mNum=mNums[0],
                        im=grid.imaginaryHarmonic[0],
                    ),
                ),
                evolvedVar=f,
                implicitVar=f,
                profiles={"V": vProfile},
            )
        )
        .withFixedMatrix()
        .dict()
    )

    result["termTags"] = [tag]

    assert cm.kinAdvX(f, grid, evolvedHarmonics=[harmonic]).dict() == result

    # Bad case - using non-distribution variable

    with pytest.raises(AssertionError) as e_info:
        cm.kinAdvX(vc.Variable("notDist", grid, isDistribution=False), grid)
    assert (
        e_info.value.args[0] == "kinAdvX distribution must be a distribution variable"
    )


def test_kinetic_advectionEx(norms: dict):

    # Use grid with lMax = 1, mMax = 2

    grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        1,
        0,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    f = vc.Variable("f", grid, isDistribution=True)

    E = vc.Variable("E", grid)

    cm.advectionEx(f, E, grid, norms)

    H1 = vc.Variable("H1", grid)
    H2 = vc.Variable("H2", grid)
    G1 = vc.Variable("G1", grid)

    result = {
        "type": "customModel",
        "termTags": ["eAdv_H_1", "eAdv_G_2"],
        "termGenerators": {"tags": []},
        "modelboundData": {
            "modelboundDataType": "varlikeData",
            "dataNames": ["G1", "H1", "G2", "H2"],
            "G1": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "G1",
                "requiredVarNames": ["f"],
            },
            "H1": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "H1",
                "requiredVarNames": ["f"],
            },
            "G2": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "G2",
                "requiredVarNames": ["f"],
            },
            "H2": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "H2",
                "requiredVarNames": ["f"],
            },
        },
        "eAdv_G_2": (
            (1.0)
            * G1
            @ mc.MatrixTerm(
                "eAdv_G_2",
                stencil=mc.DiagonalStencil(evolvedHarmonics=[2]),
                evolvedVar=f,
                implicitVar=E,
            )
        ).dict(),
        "eAdv_H_1": (
            (1 / 3)
            * H2
            @ mc.MatrixTerm(
                "eAdv_H_1",
                stencil=mc.DiagonalStencil(evolvedHarmonics=[1]),
                evolvedVar=f,
                implicitVar=E,
            )
        ).dict(),
    }

    assert cm.advectionEx(f, E, grid, norms).dict() == result


def test_eeCollIsotropic(grid: Grid, norms: dict):
    distribution = vc.Variable("f", grid, isDistribution=True)

    elTemperature = vc.Variable("Te", grid)

    elDensity = vc.Variable("ne", grid)

    textbook = dv.Textbook(grid)

    model = cm.eeCollIsotropic(
        distribution,
        elTemperature,
        elDensity,
        norms,
        grid,
        textbook,
    )

    velocityProfile = grid.profile(np.array([1.0 / v**2 for v in grid.vGrid]), dim="V")

    normConst = (
        elCharge**4
        / (4 * np.pi * elMass**2 * epsilon0**2)
        * norms["density"]
        * norms["time"]
        / norms["velGrid"] ** 3
    )

    result = {
        "type": "customModel",
        "termTags": ["drag_term", "diff_term"],
        "termGenerators": {"tags": []},
        "modelboundData": {
            "modelboundDataType": "varlikeData",
            "dataNames": ["f0", "dragCCL", "diffCCL", "weightCCL", "logLee"],
            "f0": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "f0",
                "requiredVarNames": ["f"],
            },
            "dragCCL": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "cclDragCoeff",
                "requiredVarNames": ["f0"],
            },
            "diffCCL": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "cclDiffusionCoeff",
                "requiredVarNames": ["f0", "weightCCL"],
            },
            "weightCCL": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "cclWeight",
                "requiredVarNames": ["dragCCL", "diffCCL"],
            },
            "logLee": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "logLee",
                "requiredVarNames": ["Te", "ne"],
            },
        },
        "drag_term": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": velocityProfile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normConst},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["logLee"],
                "requiredMBRowVarPowers": [1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "ddvStencil",
                "modelboundC": "dragCCL",
                "modelboundInterp": "weightCCL",
                "rowHarmonic": 1,
                "colHarmonic": 1,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "diff_term": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": velocityProfile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normConst},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["logLee"],
                "requiredMBRowVarPowers": [1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "vDiffusionStencil",
                "modelboundA": "diffCCL",
                "rowHarmonic": 1,
                "colHarmonic": 1,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
    }

    assert model.dict() == result


def test_eiCollIsotropic(grid: Grid, norms: dict):

    distribution = vc.Variable("f", grid, isDistribution=True)

    elTemperature = vc.Variable("Te", grid)

    elDensity = vc.Variable("ne", grid)

    textbook = dv.Textbook(grid)

    ionTemperature = vc.Variable("Ti", grid)

    ionDensity = vc.Variable("ni", grid)

    ionSpecies = dv.Species("ni", -1, atomicA=1.0, charge=1.0)

    velocityProfile = grid.profile(np.array([1.0 / v**2 for v in grid.vGrid]), dim="V")

    gamma0norm = (
        elCharge**4
        / (4 * np.pi * elMass**2 * epsilon0**2)
        * ionSpecies.charge**2
        * elMass
        / (ionSpecies.atomicA * amu)
    )

    normConst = gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    vBoundary = [grid.vGrid[i] + grid.vWidths[i] / 2 for i in range(len(grid.vWidths))]

    innerV = grid.profile(
        np.array([1.0 / (2 * v) for v in vBoundary]), "V", latexName="\\frac{1}{2v}"
    )

    result = {
        "type": "customModel",
        "termTags": ["drag_term", "diff_term"],
        "termGenerators": {"tags": []},
        "modelboundData": {
            "modelboundDataType": "varlikeData",
            "dataNames": ["logLei"],
            "logLei": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "logLeini",
                "requiredVarNames": ["Te", "ne"],
            },
        },
        "drag_term": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": velocityProfile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normConst},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": ["ni"],
                "requiredRowVarPowers": [1.0],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["logLei"],
                "requiredMBRowVarPowers": [1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "ddvStencil",
                "modelboundC": "none",
                "modelboundInterp": "none",
                "rowHarmonic": 1,
                "colHarmonic": 1,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "diff_term": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": velocityProfile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normConst},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": ["Ti", "ni"],
                "requiredRowVarPowers": [1.0, 1.0],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["logLei"],
                "requiredMBRowVarPowers": [1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "vDiffusionStencil",
                "modelboundA": "none",
                "rowHarmonic": 1,
                "colHarmonic": 1,
                "fixedA": innerV.data.tolist(),
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
    }

    assert (
        cm.eiCollIsotropic(
            grid,
            textbook,
            norms,
            distribution,
            elTemperature,
            elDensity,
            ionTemperature,
            ionDensity,
            ionSpecies,
        ).dict()
        == result
    )

    # Using ion energy variable adds additional ion energy evolution terms

    ionEnVar = vc.Variable("Wi", grid)

    resultWithIonEnergy = dict(result)

    resultWithIonEnergy.update(
        {
            "termTags": ["drag_term", "diff_term", "diff_term_en", "drag_term_en"],
            "diff_term_en": {
                "termType": "matrixTerm",
                "evolvedVar": "Wi",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": -1},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 2,
                    "colHarmonic": 1,
                    "termName": "diff_term",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "drag_term_en": {
                "termType": "matrixTerm",
                "evolvedVar": "Wi",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": -1},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 2,
                    "colHarmonic": 1,
                    "termName": "drag_term",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
        }
    )

    assert (
        cm.eiCollIsotropic(
            grid,
            textbook,
            norms,
            distribution,
            elTemperature,
            elDensity,
            ionTemperature,
            ionDensity,
            ionSpecies,
            ionEnVar,
        ).dict()
        == resultWithIonEnergy
    )


def test_stationaryIonEIColl(grid: Grid, norms: dict):

    textbook = dv.Textbook(grid)

    distribution = vc.Variable("f", grid, isDistribution=True)

    elTemperature = vc.Variable("Te", grid)

    elDensity = vc.Variable("ne", grid)

    ionDensity = vc.Variable("ni", grid)

    ionSpecies = dv.Species("ni", -1, atomicA=1.0, charge=1.0)

    evolvedHarmonics = list(range(1, grid.numH + 1))

    model = cm.stationaryIonEIColl(
        grid,
        textbook,
        norms,
        distribution,
        ionDensity,
        elDensity,
        elTemperature,
        ionSpecies,
        evolvedHarmonics,
    )

    hProfile = []
    for l in grid.lGrid:
        hProfile.append(l * (l + 1.0) / 2.0)
    harmonicProfile = grid.profile(np.array(hProfile), "H")
    velocityProfile = grid.profile(np.array([1.0 / v**3 for v in grid.vGrid]), dim="V")

    gamma0norm = (
        elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2) * ionSpecies.charge**2
    )

    normConst = -gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    result = {
        "type": "customModel",
        "termTags": ["ei_colls_stationary"],
        "termGenerators": {"tags": []},
        "modelboundData": {
            "modelboundDataType": "varlikeData",
            "dataNames": ["logLei"],
            "logLei": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "logLeini",
                "requiredVarNames": [elTemperature.name, elDensity.name],
            },
        },
        "ei_colls_stationary": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": harmonicProfile.data.tolist(),
            "velocityProfile": velocityProfile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normConst},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [ionDensity.name],
                "requiredRowVarPowers": [1.0],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["logLei"],
                "requiredMBRowVarPowers": [1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": evolvedHarmonics,
                "evolvedVCells": [],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
    }

    assert model.dict() == result


def test_flowingIonEIColl(grid: Grid, norms: dict):

    textbook = dv.Textbook(grid)

    distribution = vc.Variable("f", grid, isDistribution=True)

    elTemperature = vc.Variable("Te", grid)

    elDensity = vc.Variable("ne", grid)

    ionDensity = vc.Variable("ni", grid)

    ionFlowSpeed = vc.Variable("ui", grid)

    ionSpecies = dv.Species("ni", -1, atomicA=1.0, charge=1.0)

    evolvedHarmonics = list(range(2, grid.numH + 1, 2))

    model = cm.flowingIonEIColl(
        grid,
        textbook,
        norms,
        distribution,
        ionDensity,
        ionFlowSpeed,
        elDensity,
        elTemperature,
        ionSpecies,
        evolvedHarmonics,
    )

    v1Profile = grid.profile(np.array([1.0 / v**1 for v in grid.vGrid]), dim="V")
    v2Profile = grid.profile(np.array([1.0 / v**2 for v in grid.vGrid]), dim="V")
    v3Profile = grid.profile(np.array([1.0 / v**3 for v in grid.vGrid]), dim="V")

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)
    gamma0norm = gamma0norm * ionSpecies.charge**2

    adfAtZero = [1 / grid.vGrid[1], 0]

    normConst = (
        gamma0norm / 3 * norms["density"] * norms["time"] / norms["velGrid"] ** 3
    )

    l = 1

    normLL2 = -(l * (l + 1.0) / 2.0) * normConst

    C1 = (l + 1) * (l + 2) / ((2 * l + 1) * (2 * l + 3))
    normC1 = (
        C1 * gamma0norm / 2 * norms["density"] * norms["time"] / norms["velGrid"] ** 3
    )

    C2 = -(l - 1) * l / ((2 * l + 1) * (2 * l - 1))
    normC2 = (
        C2 * gamma0norm / 2 * norms["density"] * norms["time"] / norms["velGrid"] ** 3
    )

    C3 = -((l + 1) * l / 2 + l + 1) / ((2 * l + 1) * (2 * l + 3))
    normC3 = C3 * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    C4 = (-(l + 1) * l / 2 + l + 2) / ((2 * l + 1) * (2 * l + 3)) + l / (2 * l + 1)
    normC4 = C4 * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    C5 = ((l + 1) * l / 2 + l - 1) / ((2 * l + 1) * (2 * l - 1)) - (l + 1) / (2 * l + 1)
    normC5 = C5 * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    C6 = -((l + 1) * l / 2 - l) / ((2 * l + 1) * (2 * l - 1))
    normC6 = C6 * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    result = {
        "type": "customModel",
        "termTags": [
            "diffTermI2",
            "diffTermJ2",
            "dfdv2",
            "termLL2",
            "C1Il+2_h=2",
            "C1J-l-1_h=2",
            "C2Il_h=2",
            "C2J1-l_h=2",
            "C3Il+2_h=2",
            "C4J-l-1_h=2",
            "C5Il_h=2",
            "C6J1-l_h=2",
        ],
        "termGenerators": {"tags": []},
        "modelboundData": {
            "modelboundDataType": "varlikeData",
            "dataNames": [
                "CII0",
                "CII2",
                "CIJ-1",
                "CII0sh",
                "CII2sh",
                "CIJ-1sh",
                "IJSum",
                "IJSum2",
                "df0",
                "ddf0",
                "logLei",
                "CII1",
                "CII3",
                "CIJ-2",
                "CIJ0",
            ],
            "CII0": {
                "isDistribution": True,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "CII0",
                "requiredVarNames": ["ui"],
            },
            "CII2": {
                "isDistribution": True,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "CII2",
                "requiredVarNames": ["ui"],
            },
            "CIJ-1": {
                "isDistribution": True,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "CIJ-1",
                "requiredVarNames": ["ui"],
            },
            "CII0sh": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "f0",
                "requiredVarNames": ["CII0"],
            },
            "CII2sh": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "f0",
                "requiredVarNames": ["CII2"],
            },
            "CIJ-1sh": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "f0",
                "requiredVarNames": ["CIJ-1"],
            },
            "IJSum": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "sumTerm",
                "requiredVarNames": ["CII2sh", "CIJ-1sh", "CII0sh"],
            },
            "IJSum2": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "sumTerm2",
                "requiredVarNames": ["CII2sh", "CIJ-1sh"],
            },
            "df0": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "df0/dv",
                "requiredVarNames": ["f"],
            },
            "ddf0": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "d2f0/dv2",
                "requiredVarNames": ["f"],
            },
            "logLei": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "logLeini",
                "requiredVarNames": ["Te", "ne"],
            },
            "CII1": {
                "isDistribution": True,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "CII1",
                "requiredVarNames": ["ui"],
            },
            "CII3": {
                "isDistribution": True,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "CII3",
                "requiredVarNames": ["ui"],
            },
            "CIJ-2": {
                "isDistribution": True,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "CIJ-2",
                "requiredVarNames": ["ui"],
            },
            "CIJ0": {
                "isDistribution": True,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "CIJ0",
                "requiredVarNames": ["ui"],
            },
        },
        "diffTermI2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normConst},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": ["ni"],
                "requiredRowVarPowers": [1.0],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CII2sh", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "vDiffusionStencil",
                "modelboundA": "none",
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "adfAtZero": adfAtZero,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "diffTermJ2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normConst},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": ["ni"],
                "requiredRowVarPowers": [1.0],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CIJ-1sh", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "vDiffusionStencil",
                "modelboundA": "none",
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "adfAtZero": adfAtZero,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "dfdv2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normConst},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": ["ni"],
                "requiredRowVarPowers": [1.0],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["IJSum2", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "ddvStencil",
                "modelboundC": "none",
                "modelboundInterp": "none",
                "rowHarmonic": 2,
                "colHarmonic": 2,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "termLL2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v3Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normLL2},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": ["ni"],
                "requiredRowVarPowers": [1.0],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["IJSum", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "C1Il+2_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "ni",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC1},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CII3", "ddf0", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "C1J-l-1_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "ni",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC1},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CIJ-2", "ddf0", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
        "C2Il_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "ni",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC2},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CII1", "ddf0", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
        "C2J1-l_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "ni",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC2},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CIJ0", "ddf0", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
        "C3Il+2_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "ni",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC3},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CII3", "df0", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
        "C4J-l-1_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "ni",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC4},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CIJ-2", "df0", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
        "C5Il_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "ni",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC5},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CII1", "df0", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
        "C6J1-l_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "ni",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC6},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["CIJ0", "df0", "logLei"],
                "requiredMBRowVarPowers": [1.0, 1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
    }

    assert model.dict() == result

    # Using ionFlux argument

    ionFlux = vc.Variable("Gi", grid)

    modelWithIonFlux = cm.flowingIonEIColl(
        grid,
        textbook,
        norms,
        distribution,
        ionDensity,
        ionFlowSpeed,
        elDensity,
        elTemperature,
        ionSpecies,
        evolvedHarmonics,
        ionFlux,
    )

    elIonMassRatio = elMass / (ionSpecies.atomicA * amu)

    normIonFriction = -elIonMassRatio / 3

    resultWithIonFlux = dict(result)

    resultWithIonFlux.update(
        {
            "termTags": [
                "diffTermI2",
                "diffTermJ2",
                "dfdv2",
                "termLL2",
                "C1Il+2_h=2",
                "C1J-l-1_h=2",
                "C2Il_h=2",
                "C2J1-l_h=2",
                "C3Il+2_h=2",
                "C4J-l-1_h=2",
                "C5Il_h=2",
                "C6J1-l_h=2",
                "diffTermI2Ion",
                "diffTermJ2Ion",
                "dfdv2Ion",
                "termLL2Ion",
                "C1Il+2_h=2Ion",
                "C1J-l-1_h=2Ion",
                "C2Il_h=2Ion",
                "C2J1-l_h=2Ion",
                "C3Il+2_h=2Ion",
                "C4J-l-1_h=2Ion",
                "C5Il_h=2Ion",
                "C6J1-l_h=2Ion",
            ],
            "diffTermI2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "diffTermI2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "diffTermJ2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "diffTermJ2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "dfdv2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "dfdv2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "termLL2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "f",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "termLL2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C1Il+2_h=2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "ni",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C1Il+2_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C1J-l-1_h=2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "ni",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C1J-l-1_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C2Il_h=2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "ni",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C2Il_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C2J1-l_h=2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "ni",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C2J1-l_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C3Il+2_h=2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "ni",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C3Il+2_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C4J-l-1_h=2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "ni",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C4J-l-1_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C5Il_h=2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "ni",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C5Il_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
            "C6J1-l_h=2Ion": {
                "termType": "matrixTerm",
                "evolvedVar": "Gi",
                "implicitVar": "ni",
                "spatialProfile": [],
                "harmonicProfile": [],
                "velocityProfile": [],
                "evaluatedTermGroup": 0,
                "implicitGroups": [1],
                "generalGroups": [],
                "customNormConst": {"multConst": normIonFriction},
                "timeSignalData": {
                    "timeSignalType": "none",
                    "timeSignalPeriod": 0.0,
                    "timeSignalParams": [],
                    "realTimePeriod": False,
                },
                "varData": {
                    "requiredRowVarNames": [],
                    "requiredRowVarPowers": [],
                    "requiredColVarNames": [],
                    "requiredColVarPowers": [],
                    "requiredMBRowVarNames": [],
                    "requiredMBRowVarPowers": [],
                    "requiredMBColVarNames": [],
                    "requiredMBColVarPowers": [],
                },
                "stencilData": {
                    "stencilType": "termMomentStencil",
                    "momentOrder": 1,
                    "colHarmonic": 2,
                    "termName": "C6J1-l_h=2",
                },
                "skipPattern": False,
                "fixedMatrix": False,
            },
        }
    )

    assert modelWithIonFlux.dict() == resultWithIonFlux

    # Bad case - evolvedHarmonics cannot include l=1

    with pytest.raises(AssertionError) as e_info:
        cm.flowingIonEIColl(
            grid,
            textbook,
            norms,
            distribution,
            ionDensity,
            ionFlowSpeed,
            elDensity,
            elTemperature,
            ionSpecies,
            evolvedHarmonics=[1],
        )
    assert (
        e_info.value.args[0]
        == "flowingIonEIColl cannot be used to evolve harmonic with index 1"
    )

    # Bad case - evolvedHarmonics cannot include l=1

    with pytest.raises(AssertionError) as e_info:
        cm.flowingIonEIColl(
            grid,
            textbook,
            norms,
            distribution,
            ionDensity,
            ionFlowSpeed,
            elDensity,
            elTemperature,
            ionSpecies,
            evolvedHarmonics=[1],
        )
    assert (
        e_info.value.args[0]
        == "flowingIonEIColl cannot be used to evolve harmonic with index 1"
    )


def test_eeCollHigherL(grid: Grid, norms: dict):
    textbook = dv.Textbook(grid)

    distribution = vc.Variable("f", grid, isDistribution=True)

    elTemperature = vc.Variable("Te", grid)

    elDensity = vc.Variable("ne", grid)

    ionSpecies = dv.Species("ni", -1, atomicA=1.0, charge=1.0)

    evolvedHarmonics = list(range(2, grid.numH + 1, 2))

    model = cm.eeCollHigherL(
        grid, textbook, norms, distribution, elTemperature, elDensity, evolvedHarmonics
    )

    v1Profile = grid.profile(np.array([1.0 / v**1 for v in grid.vGrid]), dim="V")
    v2Profile = grid.profile(np.array([1.0 / v**2 for v in grid.vGrid]), dim="V")
    v3Profile = grid.profile(np.array([1.0 / v**3 for v in grid.vGrid]), dim="V")

    gamma0norm = elCharge**4 / (4 * np.pi * elMass**2 * epsilon0**2)
    gamma0norm = gamma0norm * ionSpecies.charge**2

    adfAtZero = [1 / grid.vGrid[1], 0]

    normf0fl = (
        8
        * np.pi
        * gamma0norm
        * norms["density"]
        * norms["time"]
        / norms["velGrid"] ** 3
    )

    normfl = gamma0norm / 3 * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    normLL = -gamma0norm / 3 * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    l = 1

    C1 = (l + 1) * (l + 2) / ((2 * l + 1) * (2 * l + 3))
    normC1 = (
        C1 * gamma0norm / 2 * norms["density"] * norms["time"] / norms["velGrid"] ** 3
    )

    C2 = -(l - 1) * l / ((2 * l + 1) * (2 * l - 1))
    normC2 = (
        C2 * gamma0norm / 2 * norms["density"] * norms["time"] / norms["velGrid"] ** 3
    )

    C3 = -((l + 1) * l / 2 + l + 1) / ((2 * l + 1) * (2 * l + 3))
    normC3 = C3 * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    C4 = (-(l + 1) * l / 2 + l + 2) / ((2 * l + 1) * (2 * l + 3))
    normC4 = C4 * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    C5 = ((l + 1) * l / 2 + l - 1) / ((2 * l + 1) * (2 * l - 1))
    normC5 = C5 * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    C6 = -((l + 1) * l / 2 - l) / ((2 * l + 1) * (2 * l - 1))  # C6
    normC6 = C6 * gamma0norm * norms["density"] * norms["time"] / norms["velGrid"] ** 3

    result = {
        "type": "customModel",
        "termTags": [
            "8pif0fl",
            "diffTermI2",
            "diffTermJ2",
            "dfdv2",
            "termLL",
            "C1Il+2_h=2",
            "C1J-l-1_h=2",
            "C2Il_h=2",
            "C2J1-l_h=2",
            "C3Il+2_h=2",
            "C4J-l-1_h=2",
            "C5Il_h=2",
            "C6J1-l_h=2",
        ],
        "termGenerators": {"tags": []},
        "modelboundData": {
            "modelboundDataType": "varlikeData",
            "dataNames": ["f0", "I0", "I2", "J-1", "IJSum", "logLee", "df0", "ddf0"],
            "f0": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "f0",
                "requiredVarNames": ["f"],
            },
            "I0": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "I0",
                "requiredVarNames": ["f0"],
            },
            "I2": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "I2",
                "requiredVarNames": ["f0"],
            },
            "J-1": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "J-1",
                "requiredVarNames": ["f0"],
            },
            "IJSum": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": True,
                "derivationPriority": 0,
                "ruleName": "sumTerm",
                "requiredVarNames": ["I2", "J-1", "I0"],
            },
            "logLee": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": False,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "logLee",
                "requiredVarNames": ["Te", "ne"],
            },
            "df0": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "df0/dv",
                "requiredVarNames": ["f"],
            },
            "ddf0": {
                "isDistribution": False,
                "isScalar": False,
                "isSingleHarmonic": True,
                "isDerivedFromOtherData": False,
                "derivationPriority": 0,
                "ruleName": "d2f0/dv2",
                "requiredVarNames": ["f"],
            },
        },
        "8pif0fl": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": [],
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normf0fl},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["logLee", "f0"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "diffTermI2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normfl},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["I2", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "vDiffusionStencil",
                "modelboundA": "none",
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "adfAtZero": adfAtZero,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "diffTermJ2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normfl},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["J-1", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "vDiffusionStencil",
                "modelboundA": "none",
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "adfAtZero": adfAtZero,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "dfdv2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normfl},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["IJSum", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "ddvStencil",
                "modelboundC": "none",
                "modelboundInterp": "none",
                "rowHarmonic": 2,
                "colHarmonic": 2,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "termLL": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [0.0, 1.0],
            "velocityProfile": v3Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normLL},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["IJSum", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "diagonalStencil",
                "evolvedXCells": [],
                "evolvedHarmonics": [2],
                "evolvedVCells": [],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "C1Il+2_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC1},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["ddf0", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": False,
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "integralIndex": 3,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "C1J-l-1_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC1},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["ddf0", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": True,
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "integralIndex": -2,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "C2Il_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC2},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["ddf0", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": False,
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "integralIndex": 1,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "C2J1-l_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v1Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC2},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["ddf0", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": True,
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "integralIndex": 0,
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "C3Il+2_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC3},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["df0", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": False,
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "integralIndex": 3,
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
        "C4J-l-1_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC4},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["df0", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": True,
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "integralIndex": -2,
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
        "C5Il_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC5},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["df0", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": False,
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "integralIndex": 1,
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
        "C6J1-l_h=2": {
            "termType": "matrixTerm",
            "evolvedVar": "f",
            "implicitVar": "f",
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": v2Profile.data.tolist(),
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normC6},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": ["df0", "logLee"],
                "requiredMBRowVarPowers": [1.0, 1.0],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": True,
                "rowHarmonic": 2,
                "colHarmonic": 2,
                "integralIndex": 0,
            },
            "skipPattern": True,
            "fixedMatrix": False,
        },
    }

    assert model.dict() == result


def test_ampere_maxwell_kinetic_term(grid: Grid, norms: dict):

    distribution = vc.Variable("f", grid, isDistribution=True)

    amTerm = cm.ampereMaxwellKineticElTerm(distribution, norms)

    # Must assign an evolved var to the term to convert to dict()
    amTerm.evolvedVar = distribution

    normConst = (
        elCharge / (3 * epsilon0) * norms["density"] * norms["time"] / norms["EField"]
    )

    assert amTerm.dict() == {
        "termType": "matrixTerm",
        "evolvedVar": distribution.name,
        "implicitVar": distribution.name,
        "spatialProfile": [],
        "harmonicProfile": [],
        "velocityProfile": [],
        "evaluatedTermGroup": 0,
        "implicitGroups": [1],
        "generalGroups": [],
        "customNormConst": {"multConst": normConst},
        "timeSignalData": {
            "timeSignalType": "none",
            "timeSignalPeriod": 0.0,
            "timeSignalParams": [],
            "realTimePeriod": False,
        },
        "varData": {
            "requiredRowVarNames": [],
            "requiredRowVarPowers": [],
            "requiredColVarNames": [],
            "requiredColVarPowers": [],
            "requiredMBRowVarNames": [],
            "requiredMBRowVarPowers": [],
            "requiredMBColVarNames": [],
            "requiredMBColVarPowers": [],
        },
        "stencilData": {
            "stencilType": "momentStencil",
            "momentOrder": 1,
            "momentHarmonic": 2,
        },
        "skipPattern": False,
        "fixedMatrix": False,
    }


def test_diffusive_heating_term(grid: Grid, norms: dict):

    distribution = vc.Variable("f", grid, isDistribution=True)

    density = vc.Variable("ne", grid)

    heatingProfile = Profile(np.ones(grid.numX), dim="X")

    heatingTerm = cm.diffusiveHeatingTerm(
        grid,
        norms,
        distribution,
        density,
        heatingProfile,
    )

    # Must assign an evolved var to the term to convert to dict()
    heatingTerm.evolvedVar = distribution

    dv = grid.vWidths

    vBoundary = [grid.vGrid[i] + dv[i] / 2 for i in range(len(dv))]

    v2Profile = Profile(np.array([1.0 / v**2 for v in grid.vGrid]), dim="V")

    v2bProfile = Profile(np.array([v**2 for v in vBoundary]), dim="V")

    normConst = elCharge / (3 * elMass) * norms["eVTemperature"] / norms["velGrid"] ** 2

    assert heatingTerm.dict() == {
        "termType": "matrixTerm",
        "evolvedVar": distribution.name,
        "implicitVar": distribution.name,
        "spatialProfile": heatingProfile.data.tolist(),
        "harmonicProfile": [],
        "velocityProfile": v2Profile.data.tolist(),
        "evaluatedTermGroup": 0,
        "implicitGroups": [1],
        "generalGroups": [],
        "customNormConst": {"multConst": normConst},
        "timeSignalData": {
            "timeSignalType": "none",
            "timeSignalPeriod": 0.0,
            "timeSignalParams": [],
            "realTimePeriod": False,
        },
        "varData": {
            "requiredRowVarNames": [density.name],
            "requiredRowVarPowers": [-1.0],
            "requiredColVarNames": [],
            "requiredColVarPowers": [],
            "requiredMBRowVarNames": [],
            "requiredMBRowVarPowers": [],
            "requiredMBColVarNames": [],
            "requiredMBColVarPowers": [],
        },
        "stencilData": {
            "stencilType": "vDiffusionStencil",
            "modelboundA": "none",
            "rowHarmonic": 1,
            "colHarmonic": 1,
            "fixedA": v2bProfile.data.tolist(),
        },
        "skipPattern": False,
        "fixedMatrix": False,
    }

    # Bad case - heating profile is not in spatial coordinate X

    with pytest.raises(AssertionError) as e_info:
        cm.diffusiveHeatingTerm(
            grid,
            norms,
            distribution,
            density,
            heatingProfile=v2Profile,
        )
    assert (
        e_info.value.args[0]
        == "heatingProfile in diffusiveHeatingTerm must be a spatial profile"
    )


def test_logicalBCmodel(grid: Grid):

    distribution = vc.Variable("f", grid, isDistribution=True)

    elDensity = vc.Variable("ne", grid)

    ionCurrent = vc.Variable("J", grid, isDerived=True, isScalar=True)

    # By default
    # - No dual grid or boundary value density is used
    # - Total current at boundary is zero
    # - Bisection tolerance for velocity cutoff is 1e-12
    # - BC is applied to the right boundary only
    # - All harmonics are evolved

    model = cm.logicalBCModel(
        grid,
        distribution,
        ionCurrent,
        elDensity,
    )

    # l+1 harmonic term constants at right boundary for 1st harmonic (l=0)
    l = 0
    normPlus0 = -(l + 1) / ((2 * l + 3))

    # l-1 harmonic term constants at right boundary for 2nd harmonic (l=1)
    l = 1
    normMinus1 = -l / ((2 * l - 1))

    result = {
        "type": "customModel",
        "termTags": [
            "lbcPlus_odd1",
            "lbcPlus_even1",
            "lbcMinus_odd2",
            "lbcMinus_even2",
        ],
        "termGenerators": {"tags": []},
        "modelboundData": {
            "modelboundDataType": "modelboundLBCData",
            "ionCurrentVarName": ionCurrent.name,
            "totalCurrentVarName": "none",
            "bisectionTolerance": 1e-12,
            "leftBoundary": False,
            "ruleName": "rightDistExt",
            "requiredVarNames": [distribution.name, elDensity.name],
        },
        "lbcPlus_odd1": {
            "termType": "matrixTerm",
            "evolvedVar": distribution.name,
            "implicitVar": distribution.name,
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": [],
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normPlus0},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": [],
                "requiredMBRowVarPowers": [],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "scalingLogicalBoundaryStencil",
                "rowHarmonic": 1,
                "colHarmonic": 2,
                "leftBoundary": True,
                "includedDecompHarmonics": [2],
                "ruleName": "leftDistExt",
                "requiredVarNames": [distribution.name, elDensity.name],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "lbcPlus_even1": {
            "termType": "matrixTerm",
            "evolvedVar": distribution.name,
            "implicitVar": distribution.name,
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": [],
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normPlus0},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": [],
                "requiredMBRowVarPowers": [],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "scalingLogicalBoundaryStencil",
                "rowHarmonic": 1,
                "colHarmonic": 2,
                "leftBoundary": True,
                "includedDecompHarmonics": [1],
                "ruleName": "leftDistExt",
                "requiredVarNames": [distribution.name, elDensity.name],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "lbcMinus_odd2": {
            "termType": "matrixTerm",
            "evolvedVar": distribution.name,
            "implicitVar": distribution.name,
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": [],
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normMinus1},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": [],
                "requiredMBRowVarPowers": [],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "scalingLogicalBoundaryStencil",
                "rowHarmonic": 2,
                "colHarmonic": 1,
                "leftBoundary": True,
                "includedDecompHarmonics": [2],
                "ruleName": "leftDistExt",
                "requiredVarNames": [distribution.name, elDensity.name],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "lbcMinus_even2": {
            "termType": "matrixTerm",
            "evolvedVar": distribution.name,
            "implicitVar": distribution.name,
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": [],
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normMinus1},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": [],
                "requiredMBRowVarPowers": [],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "scalingLogicalBoundaryStencil",
                "rowHarmonic": 2,
                "colHarmonic": 1,
                "leftBoundary": True,
                "includedDecompHarmonics": [1],
                "ruleName": "leftDistExt",
                "requiredVarNames": [distribution.name, elDensity.name],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
    }

    assert model.dict() == result

    # Using evolvedHarmonics l=0 and l=1 at left boundary

    modelLeft = cm.logicalBCModel(
        grid,
        distribution,
        ionCurrent,
        elDensity,
        leftBoundary=True,
        evolvedHarmonics=[0, 1],
    )

    # l-1 harmonic term constants at right boundary for 2nd harmonic (l=1)
    l = 1
    normMinus0 = +l / ((2 * l - 1))

    # l+1 harmonic term constants at right boundary for 1st harmonic (l=0)
    l = 0
    normPlus1 = +(l + 1) / ((2 * l + 3))

    resultLeft = {
        "type": "customModel",
        "termTags": ["lbcMinus0", "lbcPlus1"],
        "termGenerators": {"tags": []},
        "modelboundData": {
            "modelboundDataType": "modelboundLBCData",
            "ionCurrentVarName": ionCurrent.name,
            "totalCurrentVarName": "none",
            "bisectionTolerance": 1e-12,
            "leftBoundary": True,
            "ruleName": "leftDistExt",
            "requiredVarNames": [distribution.name, elDensity.name],
        },
        "lbcMinus0": {
            "termType": "matrixTerm",
            "evolvedVar": distribution.name,
            "implicitVar": distribution.name,
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": [],
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normMinus0},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": [],
                "requiredMBRowVarPowers": [],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "scalingLogicalBoundaryStencil",
                "rowHarmonic": 0,
                "colHarmonic": 1,
                "leftBoundary": True,
                "ruleName": "leftDistExt",
                "requiredVarNames": [distribution.name, elDensity.name],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
        "lbcPlus1": {
            "termType": "matrixTerm",
            "evolvedVar": distribution.name,
            "implicitVar": distribution.name,
            "spatialProfile": [],
            "harmonicProfile": [],
            "velocityProfile": [],
            "evaluatedTermGroup": 0,
            "implicitGroups": [1],
            "generalGroups": [],
            "customNormConst": {"multConst": normPlus1},
            "timeSignalData": {
                "timeSignalType": "none",
                "timeSignalPeriod": 0.0,
                "timeSignalParams": [],
                "realTimePeriod": False,
            },
            "varData": {
                "requiredRowVarNames": [],
                "requiredRowVarPowers": [],
                "requiredColVarNames": [],
                "requiredColVarPowers": [],
                "requiredMBRowVarNames": [],
                "requiredMBRowVarPowers": [],
                "requiredMBColVarNames": [],
                "requiredMBColVarPowers": [],
            },
            "stencilData": {
                "stencilType": "scalingLogicalBoundaryStencil",
                "rowHarmonic": 1,
                "colHarmonic": 2,
                "leftBoundary": True,
                "ruleName": "leftDistExt",
                "requiredVarNames": [distribution.name, elDensity.name],
            },
            "skipPattern": False,
            "fixedMatrix": False,
        },
    }

    assert modelLeft.dict() == resultLeft
