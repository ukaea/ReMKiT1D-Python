from scipy.special import legendre  # type: ignore
import numpy as np
import xarray as xr


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
