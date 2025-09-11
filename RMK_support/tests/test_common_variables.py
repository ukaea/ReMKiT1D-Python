import numpy as np
import pytest

import RMK_support.common_variables as cv
from RMK_support import Grid, RMKContext, Species


@pytest.fixture
def context():
    rk = RMKContext()
    rk.grid = Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        1,
        0,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
    )

    rk.species.add(Species("e", 0), Species("n", 1))

    return rk


@pytest.mark.parametrize(
    "associateOnCreation,addOnCreation",
    [
        pytest.param(
            True,
            True,
            id="default",
        ),
        pytest.param(
            True,
            False,
            id="associateOnly",
        ),
        pytest.param(
            False,
            True,
            id="addOnly",
        ),
    ],
)
def test_standard_variable_factory(
    context: RMKContext, associateOnCreation: bool, addOnCreation: bool
):

    rk = context

    associateOnCreation = False
    addOnCreation = False

    factory = cv.StandardFluidVariables(
        rk, rk.species["e"], associateOnCreation, addOnCreation
    )

    n = factory.density()

    assert n.name == "ne"
    assert n.units == "norm. density"
    assert n.normConst == rk.normDensity
    assert n.unitsSI == "$m^{-3}$"
    assert (n.name in rk.variables.varNames) == addOnCreation
    assert (n.name in rk.species["e"].associatedVarNames) == associateOnCreation

    G = factory.flux()

    assert G.name == "Ge_dual"
    assert G.normConst == rk.normDensity * rk.norms["speed"]
    assert (G.name in rk.variables.varNames) == addOnCreation
    assert (G.name in rk.species["e"].associatedVarNames) == associateOnCreation

    T = factory.temperature()

    assert T.isStationary
    assert T.normConst == rk.normTemperature
    assert (T.name in rk.variables.varNames) == addOnCreation
    assert (T.name in rk.species["e"].associatedVarNames) == associateOnCreation

    # Use factory to create a new species 'n' and associate variables to it
    factory.species = rk.species["n"]

    p = factory.pressure()

    assert p.name == "pn"
    assert p.isDerived
    assert p.normConst == rk.normTemperature * rk.normDensity
    assert (p.name in rk.variables.varNames) == addOnCreation
    assert (p.name in rk.species["n"].associatedVarNames) == associateOnCreation
    # Pressure requires n and T as dependents - check if these were automatically added
    assert ("nn" in rk.variables.varNames) == addOnCreation
    assert ("Tn" in rk.variables.varNames) == addOnCreation
    if associateOnCreation:
        assert factory.species["density"].name == "nn"
        assert factory.species["temperature"].name == "Tn"

    u = factory.flowSpeed()

    assert u.name == "un_dual"
    assert u.isDerived
    assert u.normConst == rk.norms["speed"]
    assert (u.name in rk.variables.varNames) == addOnCreation
    assert (u.name in rk.species["n"].associatedVarNames) == associateOnCreation

    W = factory.energyDensity()

    assert W.name == "Wn"
    assert W.normConst == rk.normTemperature * rk.normDensity
    assert (W.name in rk.variables.varNames) == addOnCreation
    assert (W.name in rk.species["n"].associatedVarNames) == associateOnCreation

    q = factory.heatflux()

    assert q.name == "qn_dual"
    assert q.isStationary
    assert q.isOnDualGrid
    assert q.normConst == rk.norms["heatFlux"]
    assert (q.name in rk.variables.varNames) == addOnCreation
    assert (q.name in rk.species["n"].associatedVarNames) == associateOnCreation

    pi = factory.viscosity()

    assert pi.name == "pin"
    assert pi.isStationary
    assert (pi.name in rk.variables.varNames) == addOnCreation
    assert (pi.name in rk.species["n"].associatedVarNames) == associateOnCreation

    E = cv.electricField("E", rk)

    assert E.normConst == rk.norms["EField"]

    dndt = cv.timeDerivative("dndt", rk.norms["time"], n)

    assert dndt.units == "norm. density / time norm."
    assert dndt.normConst == rk.normDensity / rk.norms["time"]
    assert dndt.unitsSI == "$m^{-3}/s$"
