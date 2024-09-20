from scipy.special import legendre  # type: ignore
import numpy as np
import xarray as xr
from .grid import Grid
from .rk_wrapper import RKWrapper
from typing import List, Dict
import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
from itertools import cycle, islice


def calculateFullF(data: xr.Dataset, dataName: str, Ntheta: int) -> xr.DataArray:
    """Calculate a full distribution function from given data, assuming that all harmonics have m=0

    Args:
        data (xr.Dataset): Dataset containing the ReMKiT1D output
        dataName (str): Name of the distribution data to be used
        Ntheta (int): Number of points in the angle space

    Returns:
        xr.DataArray: Data array containing the full distribution function
    """

    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / Ntheta)
    cosTheta = np.cos(thetas)
    newData = np.zeros(
        (
            len(data.coords["time"]),
            len(data.coords["x"]),
            len(data.coords["v"]),
            len(thetas),
        )
    )
    for h in data.coords["h"]:
        poly = legendre(h)
        for theta, x in enumerate(cosTheta):
            newData[:, :, :, theta] = newData[:, :, :, theta] + data[dataName][
                :, :, h, :
            ] * poly(x)

    newDArray = xr.DataArray(
        newData,
        coords=(data.coords["time"], data.coords["x"], data.coords["v"], thetas),
        dims=("time", "x", "v", "theta"),
    )

    newDArray.theta.attrs["units"] = "rad"
    return newDArray


def getSpatialIntegral(grid: Grid, dataset: xr.Dataset, varName: str) -> np.ndarray:

    data = dataset[varName].values
    integral = np.zeros(data.shape[0])  # array in time dimension
    if len(data.shape) > 2:
        integral = np.zeros((data.shape[0], *data.shape[2:]))

    isOnDualGrid = dataset[varName].attrs["isOnDualGrid"]

    for t in range(data.shape[0]):

        integral[t] = grid.domainIntegral(data[t], isOnDualGrid)

    return integral


def termXIntegralPlot(
    wrapper: RKWrapper,
    loadedData: xr.Dataset,
    varName: str,
    extraTermNames: List[str] = [],
    logPlot=False,
    plotFrom: int = 1,
) -> Dict[str, np.ndarray]:
    """Produce a plot showing the spatial integrals of all terms evolving a given variable and return a dictionary containing those integrals together with their sum

    Args:
        wrapper (RKWrapper): Wrapper containing the model and term information needed
        loadedData (xr.Dataset): Data to be analysed (should be compatible with the wrapper's variable container)
        varName (str): Name of the evolved variable
        extraTermNames (List[str], optional): Any variable names to be added to the list of terms. Useful when using term generators which are not picked up by the wrapper and are instead evaluated by hand using group evaluators. Defaults to [].
        logPlot (bool, optional): If true the y axis will be logarithmic and all terms will have their absolute value taken. Defaults to False.
        plotFrom (int, optional): Which timestep to plot from. Defaults to 1 avoiding to plot likely useless initial values in term variables.

    Returns:
        Dict[str,np.ndarray]: Dictionary containing the spatial integrals of the terms evolving the evolved variable
    """
    assert (
        not loadedData[varName].attrs["isDistribution"]
        and not loadedData[varName].attrs["isScalar"]
    ), "termXIntegralPlot available only for fluid variables"
    termVarNames = [
        model + term for model, term in wrapper.getTermsThatEvolveVar(varName)
    ] + extraTermNames

    varIntegrals = {
        name: getSpatialIntegral(wrapper.grid, loadedData, name)
        for name in termVarNames
    }

    plt.rcParams["figure.dpi"] = 150
    _, ax = plt.subplots(1, 1, figsize=(4, 4))
    s = np.zeros(len(loadedData["time"]))
    cmap = mpl.colormaps["plasma"]
    colors = cmap(np.linspace(0, 1, len(varIntegrals.keys())))
    linestyles = list(islice(cycle(["-", "--", "-."]), len(colors)))
    ax.set_prop_cycle(color=colors, linestyle=linestyles)
    for k, v in varIntegrals.items():
        (
            ax.semilogy(loadedData["time"][plotFrom:], abs(v[plotFrom:]), label=k)
            if logPlot
            else ax.plot(loadedData["time"][plotFrom:], v[plotFrom:], label=k)
        )
        s += v
    (
        ax.semilogy(
            loadedData["time"][plotFrom:],
            abs(s[plotFrom:]),
            color="k",
            linestyle="--",
            label="Total",
        )
        if logPlot
        else ax.plot(
            loadedData["time"][plotFrom:],
            s[plotFrom:],
            color="k",
            linestyle="--",
            label="Total",
        )
    )
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    ax.set_ylabel("$\int [d($" + varName + "$)/dt] dx$")
    ax.set_xlabel("$t[t_0]$")

    varIntegrals.update({"Total": s})
    return varIntegrals
