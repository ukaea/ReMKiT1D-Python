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

    G_dual = vc.Variable("G", grid, isOnDualGrid=True)
    P, P_dual = vc.varAndDual("P", grid)
    normConst = 2.0

    newModel = cm.pressureGrad(G_dual, P, normConst)

    bulk_grad = -normConst * mc.MatrixTerm(
        "bulk_grad",
        stencils.StaggeredGradStencil(),
        evolvedVar=G_dual,
        implicitVar=P,
    )

    assert newModel.dict() == {
        "type": "customModel",
        "termTags": [bulk_grad.name],
        "termGenerators": {"tags": []},
        bulk_grad.name: bulk_grad.dict(),
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
