from RMK_support.grid import Grid
import numpy as np
import pytest
from RMK_support.variable_container import (
    VariableContainer,
    Variable,
    node,
    varAndDual,
    varFromNode,
    MPIContext,
)


@pytest.fixture
def grid():
    return Grid(
        np.geomspace(5.0, 0.2, 128),
        np.geomspace(0.01, 0.8, 120),
        1,
        0,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
        isLengthInMeters=True,
    )


@pytest.fixture
def vCont(grid):
    return VariableContainer(grid, autoAddDuals=False)


def test_fluid_zero_var(grid):
    var = Variable("var", grid)

    assert all(var.data == 0)
    assert var.isFluid
    assert var.dataArr.attrs == {
        "isDerived": False,
        "isDistribution": False,
        "units": "normalized units",
        "isStationary": False,
        "isScalar": False,
        "isOnDualGrid": False,
        "priority": 0,
        "derivationRule": "none",
        "isSingleHarmonic": False,
        "normSI": 1.0,
        "unitSI": "",
    }


def test_fluid_nonzero_var(grid):

    with pytest.warns(
        UserWarning,
        match="Variable on dual grid var has been initialised with non-zero data. Make sure that the rightmost cell is zeroed out or intentionally left as non-zero.",
    ):
        var = Variable(
            "var",
            grid,
            data=np.ones(grid.numX),
            isDerived=True,
            units="arb",
            isOnDualGrid=True,
            normSI=2.0,
            unitSI="a",
        )

    assert all(var.dataArr.data == np.ones(grid.numX))
    assert var.properties == {
        "isDerived": True,
        "isDistribution": False,
        "units": "arb",
        "isStationary": False,
        "isScalar": False,
        "isOnDualGrid": True,
        "isSingleHarmonic": False,
        "priority": 0,
        "derivationRule": "none",
        "normSI": 2.0,
        "unitSI": "a",
    }

    var.switchUnits()
    assert all(var.data == 2 * np.ones(grid.numX))
    assert var.units == "a"
    var.switchUnits()
    assert all(var.data == np.ones(grid.numX))
    assert var.units == "arb"

    var.values = 3 * np.ones(grid.numX)
    assert all(var.values == 3 * np.ones(grid.numX))


def test_var_and_dual(grid):

    a, a_dual = varAndDual("a", grid, isStationary=True)

    assert a.dual.name == "a_dual"
    assert a.isStationary
    assert a.isCommunicated
    assert a.inOutput
    assert a_dual.isOnDualGrid

    assert not a.isDerived
    assert a_dual.derivation.name == "gridToDual"
    assert a_dual.derivationArgs == ["a"]

    c = a.onDualGrid().rename("c")
    assert c.name == "c"
    assert c.dual.name == "c_dual"
    assert c.isOnDualGrid

    b_dual, b = varAndDual("b", grid, primaryOnDualGrid=True)

    assert b.derivation.name == "dualToGrid"
    assert b_dual.dual.name == "b"
    assert not b.isOnDualGrid

    d = Variable("d", grid).withDual("dd")
    assert d.dual.derivation.name == "gridToDual"
    assert d.dual.name == "dd"


def test_node_var(grid):

    with pytest.warns(
        UserWarning,
        match="derivationArgs set for variable a which is not derived. Ignoring...",
    ):
        a = Variable("a", grid, derivationArgs=["c"])
    with pytest.warns(
        UserWarning,
        match="Variable b has derivation rule set, but is explicitly set to not be derived. Overriding to isDerived=True!",
    ):
        b = varFromNode("b", grid, node(a), isDerived=False)

    assert b.isDerived
    assert b.derivationArgs == ["a"]

    with pytest.warns(
        UserWarning,
        match="derivationArgs set for variable b which is produced by a NodeDerivation. Ignoring in favour of node leaf variables.",
    ):
        b = varFromNode("b", grid, node(a), derivationArgs=["a"])


def test_distribution(grid):

    f, f_dual = varAndDual("f", grid, isDistribution=True)

    assert f_dual.derivation.name == "distributionInterp"
    assert f.isDistribution

    assert f.dims == ["x_dual", "h", "v"]

    h = Variable("H", grid, isSingleHarmonic=True)
    assert h.dims == ["x", "v"]
    assert h.isSingleHarmonic


def test_multiplicative_argument(grid):

    a, b, c, d = (Variable(name, grid) for name in ["a", "b", "c", "d"])

    mult = a**2 / b

    assert mult.argMultiplicity == {"a": 2, "b": -1}
    assert mult.firstArg.name == "a"

    mult *= b
    assert mult.argMultiplicity == {"a": 2}
    mult *= 2
    assert mult.scalar == 2

    mult *= a
    assert mult.argMultiplicity["a"] == 3

    mult /= a
    assert mult.argMultiplicity["a"] == 2

    mult *= 2 * c
    assert mult.args["c"].name == "c"
    assert mult.scalar == 4

    mult *= mult
    assert mult.scalar == 16
    assert mult.argMultiplicity["a"] == 4

    mult = mult**3

    assert mult.scalar == 16**3
    assert mult.argMultiplicity["c"] == 6

    mult.scalar = 1
    assert mult.scalar == 1

    mult /= d**2
    assert mult.argMultiplicity["d"] == -2

    m2 = -mult
    assert m2.scalar == -1

    m2 = a * 2
    assert m2.scalar == 2

    m2 /= 2
    assert m2.scalar == 1

    m2 = a * mult
    assert m2.firstArg.name == "c"

    m2 = a * b
    assert m2.firstArg.name == "b"

    m2 = -a
    assert m2.scalar == -1

    m2 = a / 2
    assert m2.scalar == 0.5

    m2 = a / mult
    assert m2.argMultiplicity["a"] == -11


def test_vs_init(grid, vCont):
    assert all(vCont.dataset.coords["x"].data == grid.xGrid)
    assert all(vCont.dataset.coords["h"].data == np.array([0, 1]))
    assert all(vCont.dataset.coords["v"].data == grid.vGrid)

    vCont.add(*varAndDual("a", grid))
    vCont.add(Variable("b", grid, isScalar=True, isDerived=True))

    assert vCont["b"].isScalar

    vCont["b"] = vCont["a"].onDualGrid()
    assert not vCont["b"].isScalar
    assert vCont["b"].isFluid
    assert vCont["b"].isOnDualGrid

    with pytest.warns(
        UserWarning,
        match="Variable b already in VariableContainer. Overwriting.",
    ):
        vCont.setVar("b")


def test_var_evaluate(grid, vCont):

    vCont.add(Variable("b", grid, data=np.ones(grid.numX)))

    a = varFromNode("a", grid, 2 * node(vCont["b"]))
    assert all(vCont["b"].evaluate(vCont.dataset) == np.ones(grid.numX))
    assert all(a.evaluate(vCont.dataset) == 2 * np.ones(grid.numX))


def test_json_dump(grid, vCont):
    testCont = vCont

    testCont.setVar("var0")
    testCont.setVar("var1", data=np.ones(grid.numX), isDerived=True, units="arb")
    testCont.setVar(
        "var2",
        data=np.ones((grid.numX, grid.numH, grid.numV)),
        isDistribution=True,
        units="arb",
    )
    testCont.setVar(
        "var3",
        isScalar=True,
        isDerived=True,
        units="arb",
    )

    assert [var.name for var in testCont.implicitVars] == ["var0", "var2"]
    assert [var.name for var in testCont.derivedVars] == ["time", "var1", "var3"]

    assert testCont.varNames == [var.name for var in testCont.variables]

    expectedOutput = {
        "variables": {
            "implicitVariables": {
                "names": ["var0", "var2"],
                "var0": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.zeros(grid.numX).tolist(),
                },
                "var2": {
                    "isDistribution": True,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.ones(grid.numX * grid.numH * grid.numV).tolist(),
                },
            },
            "derivedVariables": {
                "names": ["time", "var1", "var3"],
                "time": {
                    "isDistribution": False,
                    "isScalar": True,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.zeros(1).tolist(),
                },
                "var1": {
                    "isDistribution": False,
                    "isScalar": False,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.ones(grid.numX).tolist(),
                },
                "var3": {
                    "isDistribution": False,
                    "isScalar": True,
                    "isStationary": False,
                    "isOnDualGrid": False,
                    "priority": 0,
                    "initVals": np.zeros(1).tolist(),
                },
            },
        }
    }

    assert expectedOutput == testCont.dict()


def test_mpi_context(grid, vCont):

    mpiCont = MPIContext(2, 2, 1)

    assert mpiCont.numProcs == 4
    mpiCont.numProcsX = 4
    mpiCont.numProcsH = 4
    assert mpiCont.numProcs == 16

    assert mpiCont.fluidProcs == [0, 4, 8, 12]

    vCont.add(*varAndDual("a", grid))
    vCont.add(Variable("b", grid, isScalar=True, isDerived=True, scalarHostProcess=12))

    vCont["a_dual"].isCommunicated = False

    vCont.setVar("f", isDistribution=True)

    assert mpiCont.dict(vCont) == {
        "numProcsX": 4,
        "numProcsH": 4,
        "xHaloWidth": 1,
        "commData": {
            "varsToBroadcast": ["a", "f"],
            "haloExchangeVars": ["a", "f"],
            "scalarVarsToBroadcast": ["b"],
            "scalarBroadcastRoots": [12],
        },
    }
