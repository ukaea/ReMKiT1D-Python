import RMK_support.stencils as st
from RMK_support import Variable, Grid, varAndDual
from RMK_support.derivations import SimpleDerivation, DerivationClosure
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


def test_staggered_div_grad():

    div = st.StaggeredDivStencil()

    assert div.dict() == {"stencilType": "staggeredDifferenceStencil"}

    grad = st.StaggeredGradStencil()

    assert grad.dict() == {
        "stencilType": "staggeredDifferenceStencil",
        "ignoreJacobian": True,
    }


def test_bc_stencils(grid):

    u = Variable("u", grid)
    c = Variable("c", grid)

    bcDiv = st.BCDivStencil(u, c)

    assert bcDiv.dict() == {
        "stencilType": "boundaryStencil",
        "fluxJacVar": "u",
        "lowerBoundVar": "c",
        "leftBoundary": False,
    }

    bcGrad = st.BCGradStencil()

    assert bcGrad.dict() == {
        "stencilType": "boundaryStencil",
        "leftBoundary": False,
        "ignoreJacobian": True,
    }


def test_central_diff_div_grad(grid):

    u = Variable("u", grid)

    div = st.CentralDiffDivStencil(u)

    assert div.dict() == {
        "stencilType": "centralDifferenceInterpolated",
        "interpolatedVarName": "u",
    }

    grad = st.CentralDiffGradStencil()

    assert grad.dict() == {
        "stencilType": "centralDifferenceInterpolated",
        "ignoreJacobian": True,
    }


def test_diffusion(grid):

    n = Variable("n", grid)

    deriv = DerivationClosure(SimpleDerivation("D", 1.0, [1.0]), n)

    diff = st.DiffusionStencil(deriv)

    assert diff.dict() == {
        "stencilType": "diffusionStencil",
        "ruleName": "D",
        "requiredVarNames": ["n"],
        "doNotInterpolateDiffCoeff": False,
        "ignoreJacobian": False,
    }


def test_moment_stencil():

    m = st.MomentStencil(1, 1)

    assert m.dict() == {
        "stencilType": "momentStencil",
        "momentOrder": 1,
        "momentHarmonic": 1,
    }


def test_dist_grad_stencil():

    d = st.DistGradStencil(1, 1)

    assert d.dict() == {
        "stencilType": "kineticSpatialDiffStencil",
        "rowHarmonic": 1,
        "colHarmonic": 1,
    }


def test_ddv_stencil(grid):

    ddv0 = st.DDVStencil(1, 1)

    assert ddv0.dict() == {
        "stencilType": "ddvStencil",
        "modelboundC": "none",
        "modelboundInterp": "none",
        "rowHarmonic": 1,
        "colHarmonic": 1,
    }

    ddv1 = st.DDVStencil(
        1,
        1,
        grid.profile(np.ones(grid.numV), "V"),
        grid.profile(np.ones(grid.numV), "V"),
        (0.1, 0.1),
    )

    assert ddv1.dict() == {
        "stencilType": "ddvStencil",
        "modelboundC": "none",
        "modelboundInterp": "none",
        "rowHarmonic": 1,
        "colHarmonic": 1,
        "fixedInterp": np.ones(grid.numV).tolist(),
        "fixedC": np.ones(grid.numV).tolist(),
        "cfAtZero": [0.1, 0.1],
    }

    ddv2 = st.DDVStencil(
        1,
        1,
        Variable("C", grid, isSingleHarmonic=True),
        Variable("interp", grid, isSingleHarmonic=True),
    )

    assert ddv2.dict() == {
        "stencilType": "ddvStencil",
        "modelboundC": "C",
        "modelboundInterp": "interp",
        "rowHarmonic": 1,
        "colHarmonic": 1,
    }


def test_d2dv2_stencil(grid):

    d2dv0 = st.D2DV2Stencil(1, 1)

    assert d2dv0.dict() == {
        "stencilType": "vDiffusionStencil",
        "modelboundA": "none",
        "rowHarmonic": 1,
        "colHarmonic": 1,
    }

    d2dv1 = st.D2DV2Stencil(1, 1, grid.profile(np.ones(grid.numV), "V"), (0.1, 0.1))

    assert d2dv1.dict() == {
        "stencilType": "vDiffusionStencil",
        "modelboundA": "none",
        "rowHarmonic": 1,
        "colHarmonic": 1,
        "fixedA": np.ones(grid.numV).tolist(),
        "adfAtZero": [0.1, 0.1],
    }

    d2dv2 = st.D2DV2Stencil(1, 1, Variable("A", grid, isSingleHarmonic=True))

    assert d2dv2.dict() == {
        "stencilType": "vDiffusionStencil",
        "modelboundA": "A",
        "rowHarmonic": 1,
        "colHarmonic": 1,
    }


def test_shkarofskyIJ_stencil():

    iStencil = st.ShkarofskyIStencil(1, 1, 1)

    assert iStencil.dict() == {
        "stencilType": "shkarofskyIJStencil",
        "JIntegral": False,
        "rowHarmonic": 1,
        "colHarmonic": 1,
        "integralIndex": 1,
    }
    jStencil = st.ShkarofskyJStencil(1, 1, 1)

    assert jStencil.dict() == {
        "stencilType": "shkarofskyIJStencil",
        "JIntegral": True,
        "rowHarmonic": 1,
        "colHarmonic": 1,
        "integralIndex": 1,
    }


def test_term_moment_stencil():

    stencil = st.TermMomentStencil(1, 1, "a")

    assert stencil.dict() == {
        "stencilType": "termMomentStencil",
        "momentOrder": 1,
        "colHarmonic": 1,
        "termName": "a",
    }


def test_boltz_stencil():

    boltz = st.FixedEnergyBoltzmannStencil(1, 1, 1, True, True)

    assert boltz.dict() == {
        "stencilType": "boltzmannStencil",
        "rowHarmonic": 1,
        "fixedEnergyIndex": 1,
        "transitionIndex": 1,
        "absorptionTerm": True,
        "detailedBalanceTerm": True,
    }

    boltzVar = st.VariableEnergyBoltzmannStencil(1, 1, True, True)

    assert boltzVar.dict() == {
        "stencilType": "variableBoltzmannStencil",
        "rowHarmonic": 1,
        "transitionIndex": 1,
        "absorptionTerm": True,
        "superelasticTerm": True,
    }


def test_lbc_stencil(grid):

    f = Variable("f", grid, isDistribution=True)
    n, n_dual = varAndDual("n", grid)
    n_b = Variable("nb", grid, isScalar=True, isDerived=True)

    stencil = st.LBCStencil(1, 1, f, n, n_dual, n_b, [1])

    assert stencil.dict() == {
        "stencilType": "scalingLogicalBoundaryStencil",
        "rowHarmonic": 1,
        "colHarmonic": 1,
        "leftBoundary": False,
        "includedDecompHarmonics": [1],
        "ruleName": "rightDistExt",
        "requiredVarNames": ["f", "n", "n_dual", "nb"],
    }


def test_custom1d_stencil(grid):

    u = Variable("u", grid)
    c = Variable("c", grid)

    ones = grid.profile(np.ones(grid.numX))
    stencil = st.CustomFluid1DStencil(
        (-1, 0, 1), (ones, ones, ones), (u, u, u), (c, None, c)
    )

    assert stencil.dict() == {
        "stencilType": "customFluid1DStencil",
        "xStencil": [-1, 0, 1],
        "columnVector1": ones.data.tolist(),
        "columnVector2": ones.data.tolist(),
        "columnVector3": ones.data.tolist(),
        "columnVarContVars": ["u", "u", "u"],
        "columnMBDataVars": ["c", "none", "c"],
    }
