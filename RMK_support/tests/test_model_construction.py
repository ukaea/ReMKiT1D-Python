import RMK_support.derivations as dv
import RMK_support.model_construction as mc
from RMK_support import Variable, Grid, varAndDual
import numpy as np
import pytest


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


def test_matrix_term_arithmetic(grid):

    diag = mc.DiagonalStencil()

    a, b, c, d, e = (Variable(name, grid) for name in "abcde")

    term = 5 * (
        mc.TimeSignalData()
        * (grid.profile(np.ones(grid.numX)) * (a * b * (c @ diag(d * e, c))))
    ).withEvolvedVar(a).withFixedMatrix().withSkippingPattern().regroup([1, 2], [1])

    term.evaluatedTermGroup = 1
    term.copyTermName = "dummy"
    assert term.fixedMatrix
    assert term.skipPattern
    assert term.multConst == 5
    assert term.implicitVar.name == "e"
    assert term.evolvedVar.name == "a"
    assert term.dict() == {
        "termType": "matrixTerm",
        "evolvedVar": "a",
        "implicitVar": "e",
        "spatialProfile": np.ones(grid.numX).tolist(),
        "harmonicProfile": [],
        "velocityProfile": [],
        "evaluatedTermGroup": 1,
        "implicitGroups": [1, 2],
        "generalGroups": [1],
        "customNormConst": {"multConst": 5},
        "timeSignalData": (mc.TimeSignalData().dict()),
        "varData": {
            "requiredRowVarNames": ["b", "a"],
            "requiredRowVarPowers": [1.0, 1.0],
            "requiredColVarNames": ["d"],
            "requiredColVarPowers": [1.0],
            "requiredMBRowVarNames": ["c"],
            "requiredMBRowVarPowers": [1.0],
            "requiredMBColVarNames": ["c"],
            "requiredMBColVarPowers": [1.0],
        },
        "stencilData": {
            "stencilType": "diagonalStencil",
            "evolvedXCells": [],
            "evolvedHarmonics": [],
            "evolvedVCells": [],
        },
        "skipPattern": True,
        "fixedMatrix": True,
        "multCopyTermName": "dummy",
    }


def test_term_collection(grid):

    a, b, c, d, e = (Variable(name, grid) for name in "abcde")

    termC = (
        mc.DiagonalStencil()(a).rename("a").regroup([2])
        - mc.DiagonalStencil()(b).rename("b")
        - (mc.DiagonalStencil()(d).rename("d") + mc.DiagonalStencil()(e).rename("e"))
        + (
            mc.DiagonalStencil()(c).rename("c").regroup(generalGroups=[2])
            + mc.DiagonalStencil()(c).rename("cc")
        )
    )

    assert termC["a"].implicitVar.name == "a"
    assert "b" in termC.termNames
    del termC["b"]
    assert "b" not in termC.termNames

    assert len(termC.filterByGroup([2]).termNames) == 1
    assert termC.filterByGroup([2]).termNames[0] == "a"
    assert len(termC.filterByGroup([2], general=True).termNames) == 1
    assert termC.filterByGroup([2], general=True).termNames[0] == "c"


def test_model(grid):

    model = mc.Model("m")

    a, b, c, d = (Variable(name, grid) for name in "abcd")

    model.ddt[a] += mc.DiagonalStencil()(a).rename("a")
    model.addTerm("c", -mc.DiagonalStencil()(c).withEvolvedVar(a))
    model.ddt[b] += -model.ddt[a].withSuffix("_b")
    model.ddt[c] += mc.DiagonalStencil()(d).rename("d").regroup([2])

    assert all(var in model.evolvedVars for var in ["b", "a", "c"])
    assert model.ddt[a].termNames == ["a", "c"]
    assert model.ddt[b].termNames == ["a_b", "c_b"]
    assert model.ddt[c].termNames == ["d"]

    model.isIntegrable = False
    assert not model.isIntegrable

    newModel = model.filterByGroup([2])
    assert newModel.evolvedVars == ["c"]
    assert newModel.ddt[c].termNames == ["d"]

    newModel = model.onlyEvolving(a)
    assert newModel.evolvedVars == ["a"]
    assert newModel.ddt[a].termNames == ["a", "c"]

    assert newModel.dict() == {
        "type": "customModel",
        "termTags": ["a", "c"],
        "termGenerators": {"tags": []},
        "a": model.ddt[a]["a"].dict(),
        "c": model.ddt[a]["c"].dict(),
    }

    mCollection = mc.ModelCollection()

    mCollection.add(model, newModel.rename("new"))

    assert mCollection["new"].name == "new"
    assert mCollection.getTermsThatEvolveVar(c) == [("m", "d")]
    assert mCollection.numGroups() == (2, 1)

    assert mCollection.dict() == {
        "tags": ["m", "new"],
        "m": model.dict(),
        "new": newModel.dict(),
    }

    assert mCollection.onlyEvolving(c).dict() == {
        "tags": ["m"],
        "m": model.onlyEvolving(c).dict(),
    }

    newCollection = mCollection.filterByGroup([2])

    assert mCollection.dict() == {
        "tags": ["m", "new"],
        "m": model.dict(),
        "new": newModel.dict(),
    }

    newCollection = mCollection.filterByGroup([3])
    assert len(newCollection.models) == 0


def test_derivation_term(grid):

    a, b = (Variable(name, grid) for name in "ab")
    term = mc.DerivationTerm(
        "a", dv.DerivationClosure(dv.SimpleDerivation("deriv", 2.0, [2.0]), a), mbVar=b
    )

    termC = term + term.rename("d")

    assert termC.termNames == ["a", "d"]

    assert term.withEvolvedVar(a).dict() == {
        "termType": "derivationTerm",
        "evolvedVar": "a",
        "generalGroups": [1],
        "requiredMBVarName": "b",
        "ruleName": "deriv",
        "requiredVarNames": ["a"],
    }


def test_varlike_mb_data(grid):

    mbData = mc.VarlikeModelboundData()
    a, b = (
        Variable(
            name,
            grid,
            derivation=dv.SimpleDerivation("a", 1.0, [1.0]),
            derivationArgs=["c"],
        ).withDual()
        for name in "ab"
    )

    mbData.addVar(a, b)

    model = mc.Model("model")
    model.setModelboundData(mbData)
    assert mbData.varNames == ["a", "a_dual", "b", "b_dual"]
    assert model.mbData.dict() == {
        "modelboundDataType": "varlikeData",
        "dataNames": ["a", "a_dual", "b", "b_dual"],
        "a": {
            "isDistribution": False,
            "isScalar": False,
            "isSingleHarmonic": False,
            "isDerivedFromOtherData": False,
            "derivationPriority": 0,
            "ruleName": "a",
            "requiredVarNames": ["c"],
        },
        "b": {
            "isDistribution": False,
            "isScalar": False,
            "isSingleHarmonic": False,
            "isDerivedFromOtherData": False,
            "derivationPriority": 0,
            "ruleName": "a",
            "requiredVarNames": ["c"],
        },
        "a_dual": {
            "isDistribution": False,
            "isScalar": False,
            "isSingleHarmonic": False,
            "isDerivedFromOtherData": True,
            "derivationPriority": 0,
            "ruleName": "gridToDual",
            "requiredVarNames": ["a"],
        },
        "b_dual": {
            "isDistribution": False,
            "isScalar": False,
            "isSingleHarmonic": False,
            "isDerivedFromOtherData": True,
            "derivationPriority": 0,
            "ruleName": "gridToDual",
            "requiredVarNames": ["b"],
        },
    }


def test_lbc_mb_data(grid):

    j = Variable("j", grid, isScalar=True, isDerived=True)
    f = Variable("f", grid, isDistribution=True)
    n, n_dual = varAndDual("n", grid)
    n_b = Variable("nb", grid, isScalar=True, isDerived=True)
    j_tot = Variable("j_tot", grid, isScalar=True, isDerived=True)

    mbData = mc.LBCModelboundData(grid, j, f, n, n_dual, n_b, j_tot, leftBoundary=True)

    assert mbData.varNames == ["gamma", "potential", "coVel", "shTemp"]
    assert mbData["gamma"].isScalar

    assert mbData.dict() == {
        "modelboundDataType": "modelboundLBCData",
        "ionCurrentVarName": "j",
        "totalCurrentVarName": "j_tot",
        "bisectionTolerance": 1e-12,
        "leftBoundary": True,
        "ruleName": "leftDistExt",
        "requiredVarNames": ["f", "n", "n_dual", "nb"],
    }

    assert mbData.__deriv__.dict() == {
        "type": "distScalingExtrapDerivation",
        "extrapolateToBoundary": True,
        "staggeredVars": True,
        "leftBoundary": True,
    }
