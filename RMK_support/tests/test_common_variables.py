import numpy as np
import pytest

from RMK_support import Grid, RMKContext, Species
import RMK_support.common_variables as cv


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


def test_standard_variable_factory(context):

    rk = context

    factory = cv.StandardFluidVariables(rk, rk.species["e"])

    n = factory.density()

    assert n.name == "ne"
    assert "ne" in rk.variables.varNames
    assert "ne" in rk.species["e"].associatedVarNames
    assert n.units == "norm. density"
    assert n.normConst == rk.normDensity
    assert n.unitsSI == "$m^{-3}$"

    G = factory.flux()

    assert G.name == "Ge_dual"
    assert "Ge" in rk.variables.varNames
    assert G.normConst == rk.normDensity * rk.norms["speed"]

    factory.species = rk.species["n"]

    T = factory.temperature()
    assert T.isStationary
    assert T.normConst == rk.normTemperature

    p = factory.pressure()
    assert p.name == "pn"
    assert p.isDerived
    assert p.normConst == rk.normTemperature * rk.normDensity

    u = factory.flowSpeed()
    assert u.name == "un_dual"
    assert u.isDerived
    assert u.normConst == rk.norms["speed"]

    W = factory.energyDensity()
    assert W.name in rk.species["n"].associatedVarNames
    assert W.normConst == rk.normTemperature * rk.normDensity

    q = factory.heatflux()
    assert q.isStationary
    assert q.isOnDualGrid
    assert q.normConst == rk.norms["heatFlux"]

    pi = factory.viscosity()
    assert pi.isStationary
    assert pi.name == "pin"

    E = cv.electricField("E", rk)
    assert E.normConst == rk.norms["EField"]

    assert factory.species["density"].name == "nn"
    assert factory.species["temperature"].name == "Tn"
    assert factory.species["heatflux"].name == "qn"

    dndt = cv.timeDerivative("dndt", rk.norms["time"], n)
    assert dndt.units == "norm. density / time norm."
    assert dndt.normConst == rk.normDensity / rk.norms["time"]
    assert dndt.unitsSI == "$m^{-3}/s$"
