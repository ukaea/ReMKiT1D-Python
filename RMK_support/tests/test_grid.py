from RMK_support.grid import Grid
import numpy as np
import pytest
import json


@pytest.fixture
def xGrid():
    return np.geomspace(5.0, 0.2, 128)


@pytest.fixture
def vGrid():
    return np.geomspace(0.01, 0.8, 120)


def test_grid_x_simple(xGrid):
    grid = Grid(xGrid=xGrid)
    assert all(abs(xGrid - grid.xGrid) < 1e-15)
    assert grid.numX() == 128
    assert not grid.isLengthInMeters


def test_grid_x_widths(xGrid):
    grid = Grid(xGrid=xGrid, interpretXGridAsWidths=True)
    assert all(abs(xGrid - grid.xWidths) < 1e-12)  # Floating point errors
    gridPoints = np.zeros(len(xGrid))
    gridPoints[0] = xGrid[0] / 2

    for i in range(1, len(gridPoints)):
        gridPoints[i] = gridPoints[i - 1] + xGrid[i] / 2 + xGrid[i - 1] / 2

    assert all(abs(gridPoints - grid.xGrid) < 1e-12)

    assert not grid.isPeriodic


def test_grid_v_simple(xGrid, vGrid):
    grid = Grid(xGrid=xGrid, vGrid=vGrid, lMax=1, mMax=1)
    assert all(abs(vGrid - grid.vGrid) < 1e-15)
    assert grid.lMax == 1
    assert grid.mMax == 1
    assert grid.getH(1, 0, im=False) == 2
    assert grid.numH() == 4
    assert grid.numV() == 120

    assert grid.lGrid == [0, 1, 1, 1]
    assert grid.mGrid == [0, 0, 1, 1]
    assert grid.imaginaryHarmonic == [False, False, False, True]

    with pytest.raises(ValueError):
        a = grid.getH(3)


def test_grid_v_widths(xGrid, vGrid):
    grid = Grid(xGrid=xGrid, vGrid=vGrid, lMax=1, mMax=1, interpretVGridAsWidths=True)

    assert all(abs(vGrid - grid.vWidths) < 1e-12)

    assert grid.numV() == 120

    gridPoints = np.zeros(len(vGrid))
    gridPoints[0] = vGrid[0] / 2

    for i in range(1, len(gridPoints)):
        gridPoints[i] = gridPoints[i - 1] + vGrid[i] / 2 + vGrid[i - 1] / 2

    assert all(abs(gridPoints - grid.vGrid) < 1e-12)


def test_set_jacobian(xGrid):
    grid = Grid(xGrid=xGrid)
    grid.xJacobian = 2 * np.ones(len(xGrid) + 1)
    assert all(abs(grid.xJacobian - 2 * np.ones(len(xGrid) + 1)) < 1e-12)


def test_json_dump(xGrid, vGrid):
    grid = Grid(xGrid=xGrid, vGrid=vGrid, lMax=1, mMax=1)

    expectedOutput = {
        "xGrid": {
            "isPeriodic": False,
            "isLengthInMeters": False,
            "cellCentreCoords": xGrid.tolist(),
            "faceJacobians": np.ones(129).tolist(),
        },
        "vGrid": {"cellCentreCoords": vGrid.tolist(), "maxL": 1, "maxM": 1},
    }

    assert expectedOutput == grid.dict()

    assert grid.__repr__() == json.dumps(expectedOutput, indent=4, sort_keys=True)


def test_velocity_moment_single_harmonic(xGrid, vGrid):
    grid = Grid(xGrid=xGrid, vGrid=vGrid, interpretVGridAsWidths=True, lMax=1)

    testArray = np.ones((len(xGrid), len(vGrid)))

    sumGrid = 4 * np.pi * sum(vGrid * grid.vGrid**2)

    expectedResult = sumGrid * np.ones((len(xGrid)))

    assert all(
        abs(expectedResult - grid.velocityMoment(testArray, 0)) < 1e-15 * sumGrid
    )


def test_velocity_moment_dist(xGrid, vGrid):
    grid = Grid(xGrid=xGrid, vGrid=vGrid, interpretVGridAsWidths=True, lMax=1)

    testArray = np.ones((len(xGrid), 2, len(vGrid)))

    testArray[:, 1, :] = 2

    sumGrid = 8 * np.pi * sum(vGrid * grid.vGrid**2)

    expectedResult = sumGrid * np.ones((len(xGrid)))

    assert all(
        abs(expectedResult - grid.velocityMoment(testArray, 0, 2)) < 1e-15 * sumGrid
    )


def test_velocity_moment_velocity_vector(xGrid, vGrid):
    grid = Grid(xGrid=xGrid, vGrid=vGrid, interpretVGridAsWidths=True, lMax=1)

    testArray = np.ones(len(vGrid))

    sumGrid = 4 * np.pi * sum(vGrid * grid.vGrid**2)

    expectedResult = sumGrid * np.ones(1)

    assert all(
        abs(expectedResult - grid.velocityMoment(testArray, 0)) < 1e-15 * sumGrid
    )


def test_x_widths_dual(xGrid):
    grid = Grid(xGrid=xGrid, interpretXGridAsWidths=True)

    dx_dual = grid.dualXGridWidths(True)

    assert abs(dx_dual[0] - xGrid[0] - xGrid[1] / 2) < 1e-12
    assert abs(dx_dual[-2] - xGrid[-1] - xGrid[-2] / 2) < 1e-12

    assert abs(dx_dual[1] - xGrid[1] / 2 - xGrid[2] / 2) < 1e-12

    gridPeriodic = Grid(xGrid=xGrid, interpretXGridAsWidths=True, isPeriodic=True)

    dx_dual = gridPeriodic.dualXGridWidths(True)
    assert abs(dx_dual[0] - xGrid[0] / 2 - xGrid[1] / 2) < 1e-12
    assert abs(dx_dual[-1] - xGrid[0] / 2 - xGrid[-1] / 2) < 1e-12


def test_x_volumes(xGrid):
    grid = Grid(xGrid=xGrid, interpretXGridAsWidths=True)

    grid.xJacobian = 2 * np.ones(len(xGrid) + 1)

    V = grid.xGridCellVolumes()

    assert all(abs(V - 2 * xGrid) < 1e-12)


def test_x_volumes_dual(xGrid):
    grid = Grid(xGrid=xGrid, interpretXGridAsWidths=True)

    grid.xJacobian = 2 * np.ones(len(xGrid) + 1)

    V = grid.xGridCellVolumes()
    V_dual = grid.xGridCellVolumesDual(True)

    assert abs(V_dual[0] - V[0] - V[1] / 2) < 1e-12
    assert abs(V_dual[-2] - V[-1] - V[-2] / 2) < 1e-12

    assert abs(V_dual[1] - V[1] / 2 - V[2] / 2) < 1e-12

    gridPeriodic = Grid(xGrid=xGrid, interpretXGridAsWidths=True, isPeriodic=True)
    gridPeriodic.xJacobian = 2 * np.ones(len(xGrid) + 1)

    V_dual = gridPeriodic.xGridCellVolumesDual(True)
    assert abs(V_dual[0] - V[0] / 2 - V[1] / 2) < 1e-12
    assert abs(V_dual[-1] - V[0] / 2 - V[-1] / 2) < 1e-12


def test_domain_integral(xGrid):
    grid = Grid(xGrid=xGrid, interpretXGridAsWidths=True)

    grid.xJacobian = 2 * np.ones(len(xGrid) + 1)

    data = np.ones((len(xGrid), 3))

    assert all(grid.domainIntegral(data) == np.ones(3) * np.sum(grid.xWidths * 2))


def test_domain_integral_dual(xGrid):
    grid = Grid(xGrid=xGrid, interpretXGridAsWidths=True)

    grid.xJacobian = 2 * np.ones(len(xGrid) + 1)

    data = np.ones((len(xGrid), 3))

    dV = grid.xGridCellVolumesDual(extendedBoundaryCells=True)

    assert all(grid.domainIntegral(data) == np.ones(3) * np.sum(dV[:-1]))

def test_grid_to_dual():
    grid = Grid(xGrid=np.linspace(1,2,5),interpretXGridAsWidths=True)

    data = np.linspace(0,4,5)
    interpData = grid.gridToDual(data)
    assert all(interpData[:-1] == np.interp(grid.xGridDual,grid.xGrid,data)[:-1])

    assert interpData[-1] == 2*(data[-1]-data[-2])/(grid.xWidths[-1]+grid.xWidths[-2]) * grid.xWidths[-1]/2 + data[-1]

    grid = Grid(xGrid=np.linspace(1,2,5),interpretXGridAsWidths=True,isPeriodic=True)

    data = np.linspace(0,4,5)
    interpData = grid.gridToDual(data)
    assert all(interpData == np.interp(grid.xGridDual,grid.xGrid,data,period=grid.xGridDual[-1]))

def test_profile():
    grid = Grid(xGrid=np.linspace(1,2,5),interpretXGridAsWidths=True)

    p = grid.profile(np.linspace(0,10,5),"X",latexName="a")

    assert all(p.data == np.linspace(0,10,5))
    assert p.dim == "X"
    assert p.latex() == "a"

    p = grid.profile(np.linspace(0,10,5),"X")
    assert p.latex() == "X"
    