from RMK_support.grid import Grid, Profile
import RMK_support.common_models as cm
import RMK_support.crm_support as crm
import RMK_support.derivations as dv
import RMK_support.model_construction as mc
import RMK_support.variable_container as vc
import RMK_support.stencils as stencils
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
    )


def test_advection(grid: Grid):

    n = vc.Variable("n", grid)
    G = vc.Variable("G", grid, isOnDualGrid=True)

    newModel = cm.advection(advectedVar=n, flux=G)

    bulk_div = -mc.MatrixTerm(
        "bulk_div",
        stencils.StaggeredDivStencil(),
        evolvedVar=n,
        implicitVar=G,
    )

    assert newModel.dict() == {
        "type": "customModel",
        "termTags": ["bulk_div"],
        "termGenerators": {"tags": []},
        "bulk_div": bulk_div.dict(),
    }
