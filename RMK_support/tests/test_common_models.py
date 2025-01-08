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

    assert (
        cm.simpleSourceTerm(var, sourceProfile, timeSignal).withEvolvedVar(var).dict()
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
