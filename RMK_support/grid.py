import numpy as np
import json
from typing import List, Dict, Union, cast, Optional
from itertools import accumulate
from numpy.polynomial.polynomial import polyval, polyfit


class Profile:
    """Wrapper for profiles in x,h,v spaces in ReMKiT1D"""

    def __init__(
        self, data: np.ndarray, dim: str = "X", latexName: Optional[str] = None
    ):
        """Wrapper for profiles in one of ReMKiT1D's dimensions

        Args:
            data (np.ndarray): Data - should conform to the dimension
            dim (str, optional): Profile dimension - of of ["X","H","V"]. Defaults to "X".
            latexName (Optiona[s], optional): Latex representation of the profile. Defaults to None, using dim.
        """
        self.__data__ = data
        assert dim in ["X", "H", "V"], "Profile can only be in X, H, or V dimensions"
        self.__dim__ = dim
        self.__latexName__ = latexName

    @property
    def data(self):
        return self.__data__

    @property
    def dim(self):
        return self.__dim__

    def latex(self) -> str:
        return self.__latexName__ if self.__latexName__ is not None else self.__dim__


class Grid:
    """Class containing x and v-grid data"""

    def __init__(
        self,
        xGrid: np.ndarray,
        vGrid: np.ndarray = np.ones(1),
        lMax=0,
        mMax=0,
        interpretXGridAsWidths=False,
        interpretVGridAsWidths=False,
        isPeriodic=False,
        isLengthInMeters=False,
    ):
        """Grid constructor

        Args:
            xGrid (numpy.ndarray): x coordinates of each spatial grid cell centre or their widths. If using widths set interpretXGridAsWidths to True.
            vGrid (numpy.ndarray, optional): v coordinates of each velocity grid cell centres or their widths.  If using widths set interpretVGridAsWidths to True. Defaults to a single cell (effectively no v-grid)
            lMax (int, optional): Maximum l harmonic number. Defaults to 0.
            mMax (int, optional): Maximum m harmonic number. Defaults to 0.
            interpretXGridAsWidths (bool, optional): If True interprets xGrid as cell widths. Defaults to False.
            interpretVGridAsWidths (bool, optional): If True interprets vGrid as cell widths. Defaults to False.
            isPeriodic (bool, optional): If True the x grid is set to be periodic. This means that the right boundary of the rightmost cell is the left boundary of the leftmost cell. Defaults to False.
            isLengthInMeters (bool, optional): If True will instruct ReMKiT1D to use the built-in normalization to deduce the normalized coordinates of the spatial grid. CAUTION: This can lead to issues if the default normalization is not used in the rest of the simulation. Defaults to False.
        """

        # Assertions

        assert lMax >= 0, "Negative lMax passed to Grid constructor"
        assert mMax >= 0, "Negative mMax passed to Grid constructor"

        # Initialize x grid
        if interpretXGridAsWidths:
            self.__xGrid__ = np.zeros(len(xGrid))
            self.__xGrid__[0] = xGrid[0] / 2
            for i in range(1, len(xGrid)):
                self.__xGrid__[i] = (
                    self.__xGrid__[i - 1] + (xGrid[i - 1] + xGrid[i]) / 2
                )
        else:
            self.__xGrid__ = xGrid

        self.__isPeriodic__ = isPeriodic
        self.__isLengthInMeters__ = isLengthInMeters

        dx: List[float] = []
        for x in self.__xGrid__:
            dx.append(2 * (x - sum(dx)))

        self.__xWidths__ = np.array(dx)
        self.__dualXGrid__ = np.array(list(accumulate(self.xWidths)))
        # Initialize v grid
        if interpretVGridAsWidths:
            self.__vGrid__ = np.zeros(len(vGrid))
            self.__vGrid__[0] = vGrid[0] / 2
            for i in range(1, len(vGrid)):
                self.__vGrid__[i] = (
                    self.__vGrid__[i - 1] + (vGrid[i - 1] + vGrid[i]) / 2
                )
        else:
            self.__vGrid__ = vGrid

        dv: List[float] = []
        for v in self.__vGrid__:
            dv.append(2 * (v - sum(dv)))

        self.__vWidths__ = np.array(dv)
        # Set x grid cell face jacobians to default of 1
        self.__xJacobian__ = np.ones(len(xGrid) + 1)

        # Set harmonics
        self.__lMax__ = lMax
        self.__mMax__ = mMax

        self.__lGrid__ = []
        self.__mGrid__ = []
        self.__imaginaryHarmonic__ = []
        for i in range(lMax + 1):
            for j in range(min(i + 1, mMax + 1)):
                self.__lGrid__.append(i)
                self.__mGrid__.append(j)
                self.__imaginaryHarmonic__.append(False)

                if j > 0:
                    self.__lGrid__.append(i)
                    self.__mGrid__.append(j)
                    self.__imaginaryHarmonic__.append(True)

    @property
    def xGrid(self):
        return self.__xGrid__

    @property
    def xGridDual(self):
        return self.__dualXGrid__

    @property
    def xWidths(self):
        return self.__xWidths__

    @property
    def isPeriodic(self):
        return self.__isPeriodic__

    @property
    def isLengthInMeters(self):
        return self.__isLengthInMeters__

    @property
    def vGrid(self):
        return self.__vGrid__

    @property
    def vWidths(self):
        return self.__vWidths__

    @property
    def lGrid(self):
        return self.__lGrid__

    @property
    def imaginaryHarmonic(self):
        return self.__imaginaryHarmonic__

    @property
    def mGrid(self):
        return self.__mGrid__

    @property
    def lMax(self):
        return self.__lMax__

    @property
    def mMax(self):
        return self.__mMax__

    @property
    def xJacobian(self):
        return self.__xJacobian__

    @xJacobian.setter
    def xJacobian(self, values: np.ndarray):
        """xJacobian setter

        Args:
            values (numpy.ndarray): Cell face jacobian values, should be 1 element longer than grid array
        """
        # Assertions

        assert len(values) == len(
            self.__xJacobian__
        ), "Incompatible length array passed as xJacobian"

        self.__xJacobian__ = values

    @property
    def numX(self) -> int:
        """Get number of spatial cells in grid

        Returns:
            int: Number of spatial cells
        """
        return len(self.__xGrid__)

    @property
    def numV(self) -> int:
        """Get number of velocity cells in grid

        Returns:
            int: Number of velocity magnitude cells in grid
        """
        return len(self.__vGrid__)

    @property
    def numH(self) -> int:
        """Get total number of harmonics

        Returns:
            int: Total number of harmonics including all used l and m harmonics
        """
        return len(self.__lGrid__)

    def getH(self, lNum: int, mNum=0, im=False) -> int:
        """Return harmonic with given l and m numbers. Defaults to real components.

        Args:
            lNum (int): l number of harmonic
            mNum (int, optional): m number of harmonic. Defaults to 0.
            im (bool, optional): True if the required component is imaginary. Defaults to False.

        Raises:
            ValueError: If corresponding harmonic isn't found

        Returns:
            int: index of corresponding harmonic in Fortran 1 indexing
        """

        for i in range(len(self.__lGrid__)):
            if (
                self.__lGrid__[i] == lNum
                and self.__mGrid__[i] == mNum
                and self.__imaginaryHarmonic__[i] == im
            ):
                return i + 1

        raise ValueError("getH function could not find harmonic index with given input")

    def dict(self):
        """Returns dictionary form of grid to be used in json output

        Returns:
            dict: ReMKiT1D-ready dictionary form of grid data
        """
        gridData = {
            "xGrid": {
                "isPeriodic": self.__isPeriodic__,
                "isLengthInMeters": self.__isLengthInMeters__,
                "cellCentreCoords": self.__xGrid__.tolist(),
                "faceJacobians": self.__xJacobian__.tolist(),
            },
            "vGrid": {
                "cellCentreCoords": self.__vGrid__.tolist(),
                "maxL": self.__lMax__,
                "maxM": self.__mMax__,
            },
        }

        return gridData

    def dualXGridWidths(self, extendedBoundaryCells=False) -> np.ndarray:
        """Return the widths of the dual grid cells, potentially extending the boundary cells so that the dual domain covers the regular domain. Note that the returned array is the same length as the grid, with the last value set to 1 if the grid is not periodic and to dx[0]/2+dx[-1]/2 if it is

        Args:
            extendedBoundaryCells (bool, optional): If true and the grid is not periodic will extend the left and right boundary cells to completely cover the first and last dual cell. Defaults to False.

        Returns:
            np.ndarray: Dual grid cell widths (same length as regular cell widths)
        """
        dx = self.xWidths
        dx_dual = np.ones(len(dx))
        dx_dual[:-1] = dx[:-1] / 2 + dx[1:] / 2

        if self.isPeriodic:
            dx_dual[-1] = dx[0] / 2 + dx[-1] / 2

        if extendedBoundaryCells and not self.isPeriodic:
            dx_dual[0] += dx[0] / 2
            dx_dual[-2] += dx[-1] / 2

        return dx_dual

    def xGridCellVolumes(self) -> np.ndarray:
        """Return cell volumes on the regular grid

        Returns:
            np.ndarray: Regular cell volumes
        """

        dx = self.xWidths
        J = self.xJacobian

        V = np.zeros(len(dx))

        V = dx * (J[:-1] + J[1:]) / 2

        return V

    def xGridCellVolumesDual(self, extendedBoundaryCells=False) -> np.ndarray:
        """Return cell volumes on the dual grid, potentially extending the boundary cells so that the dual domain covers the regular domain. Note that the returned array is the same length as the grid, with the last value set to 1 if the grid is not periodic and to V[0]/2+V[-1]/2 if it is

        Args:
            extendedBoundaryCells (bool, optional): If true and the grid is not periodic will extend the left and right boundary cells to completely cover the first and last dual cell. Defaults to False.


        Returns:
            np.ndarray: Dual cell volumes
        """

        V = self.xGridCellVolumes()

        V_dual = np.ones(len(V))
        V_dual[:-1] = V[:-1] / 2 + V[1:] / 2

        if self.isPeriodic:
            V_dual[-1] = V[0] / 2 + V[-1] / 2

        if extendedBoundaryCells and not self.isPeriodic:
            V_dual[0] += V[0] / 2
            V_dual[-2] += V[-1] / 2

        return V_dual

    def __repr__(self) -> str:
        return json.dumps(self.dict(), indent=4, sort_keys=True)

    def velocityMoment(
        self, distFun: np.ndarray, momentOrder: int, momentHarmonic=1
    ) -> np.ndarray:
        """Return velocity moment of distribution function (x,h,v), single harmonic variable (x,v), or velocity space vector (v)

        Args:
            distFun (np.ndarray): Distribution or single harmonic variable values.
            momentOrder (int): Moment order
            momentHarmonic (int, optional): Harmonic index (Fortran 1 indexing) to take moment of in case of distribution variable. Defaults to 1.

        Returns:
            np.ndarray: Moment represented as a contracted array
        """
        assert (
            len(np.shape(distFun)) < 4
        ), "Unsupported dimensionality of distFun in velocityMoment"
        assert (
            np.shape(distFun)[-1] == self.numV
        ), "The velocity dimension of distFun does not conform to size of velocity grid"
        if len(np.shape(distFun)) == 3:
            assert (
                momentHarmonic <= self.numH
            ), "momentHarmonic out of bounds in velocityMoment"
            moment = (
                4
                * np.pi
                * np.dot(
                    distFun[:, momentHarmonic - 1, :],
                    self.vGrid ** (2 + momentOrder) * self.vWidths,
                )
            )
        else:
            moment = (
                4
                * np.pi
                * np.dot(distFun, self.vGrid ** (2 + momentOrder) * self.vWidths)
            )

        return moment

    def domainIntegral(
        self, data: np.ndarray, isOnDualGrid: bool = False
    ) -> np.ndarray:
        """Return the integral of data along the x dimension, assuming it is the first dimension of the data passed

        Args:
            data (np.ndarray): Data to be integrated along the x direction
            isOnDualGrid (bool, optional): True if the data is on the dual grid. Defaults to False.

        Returns:
            np.ndarray: Data integrated along x
        """
        assert (
            data.shape[0] == self.numX
        ), "data passed to domainIntegral must have its first dimension conform to the spatial grid"

        dV = (
            self.xGridCellVolumesDual(extendedBoundaryCells=True)
            if isOnDualGrid
            else self.xGridCellVolumes()
        )

        if isOnDualGrid and not self.isPeriodic:
            dV[-1] = 0

        return np.transpose(np.dot(data.transpose(), dV))

    def gridToDual(self, data: np.ndarray) -> np.ndarray:
        """Linearly interpolate 1D data from regular to dual grid. Linearly extrapolates to right boundary if the grid is not periodic

        Args:
            data (np.ndarray): Data to be interpolated

        Returns:
            np.ndarray: Interpolated data on dual grid
        """
        assert data.shape == (
            self.numX,
        ), "data passed to gridToDual must be 1D and on the x grid"

        x = self.xGridDual
        xp = self.xGrid

        if self.isPeriodic:
            return np.interp(x, xp, data, period=x[-1])

        return np.piecewise(
            x,
            [x <= xp[-1], x > xp[-1]],
            [
                lambda xi: np.interp(xi, xp, data),
                lambda xi: polyval(xi, polyfit(xp[-2:], data[-2:], 1)),
            ],
        )

    def dualToGrid(self, data: np.ndarray) -> np.ndarray:
        """Linearly interpolate 1D data from dual to regular grid. Linearly extrapolates to first and last point if the grid is not periodic.

        Args:
            data (np.ndarray): Data to be interpolated

        Returns:
            np.ndarray: Interpolated data on regular grid
        """
        assert data.shape == (
            self.numX,
        ), "data passed to gridToDual must be 1D and on the x grid"

        xp = self.xGridDual
        x = self.xGrid

        if self.isPeriodic:
            return np.interp(x, xp, data, period=xp[-1])

        return np.piecewise(
            x,
            [x < xp[0], (x >= xp[0]) & (x <= xp[-2]), x > xp[-2]],
            [
                lambda xi: polyval(xi, polyfit(xp[:2], data[:2], 1)),
                lambda xi: np.interp(xi, xp, data),
                lambda xi: polyval(xi, polyfit(xp[-3:-1], data[-3:-1], 1)),
            ],
        )

    def staggeredDistToGrid(self, data: np.ndarray) -> np.ndarray:
        """Interpolate odd harmonics of a distribution variable onto the regular grid while keeping even harmonics unchanged

        Args:
            data (np.ndarray): Distribution data to be interpolated

        Returns:
            np.ndarray: Distribution data where all harmonics live on the regular grid
        """
        assert data.shape == (
            self.numX,
            self.numH,
            self.numV,
        ), "data passed to staggeredDistToGrid must be 3D and on the x,h,v grids"

        result: np.ndarray = np.copy(data)
        for h, l in enumerate(self.lGrid):
            if l % 2 == 1:
                for v in range(self.numV):
                    result[:, h, v] = self.dualToGrid(data[:, h, v])

        return result

    def staggeredDistToDual(self, data: np.ndarray) -> np.ndarray:
        """Interpolate even harmonics of a distribution variable to the dual grid while keeping odd harmonics unchanged

        Args:
            data (np.ndarray): Distribution data to be interpolated

        Returns:
            np.ndarray: Distribution with all harmonics living on the dual grid
        """
        assert data.shape == (
            self.numX,
            self.numH,
            self.numV,
        ), "data passed to staggeredDistToGrid must be 3D and on the x,h,v grids"

        result: np.ndarray = np.copy(data)
        for h, l in enumerate(self.lGrid):
            if l % 2 == 0:
                for v in range(self.numV):
                    result[:, h, v] = self.gridToDual(data[:, h, v])

        return result

    def distFullInterp(self, data: np.ndarray) -> np.ndarray:
        """Interpolate a distribution function so that the odd harmonics live on the regular grid and even harmonics on the dual grid

        Args:
            data (np.ndarray): Distribution data to be interpolated

        Returns:
            np.ndarray: Interpolated distribution
        """

        return self.staggeredDistToGrid(self.staggeredDistToDual(data))

    def profile(
        self, data: np.ndarray, dim: str = "X", latexName: Union[str, None] = None
    ) -> Profile:
        """Wrapped Profile constructor with bounds checking

        Args:
            data (np.ndarray): Profile values
            dim (str, optional): Dimension profile corresponds to. Must be one of "X","H","V". Defaults to "X".
            latexName (Union[str,None], optional): Latex name of the profile used in generating run summaries. Defaults to None.

        Returns:
            Profile: Bounds-checked Profile
        """
        assert dim in ["X", "H", "V"], "Profile can only be in X, H, or V dimensions"
        dims = {"X": self.numX, "H": self.numH, "V": self.numV}
        assert (
            dims[dim],
        ) == data.shape, (
            "data passed to profile does not conform to the appropriate dimension size"
        )

        return Profile(data, dim, latexName)


def gridFromDict(gridDict: Dict[str, object]) -> Grid:
    """Initialise the grid from a dictionary object

    Args:
        gridDict (Dict[str,object]): Dictionary to initialise from

    Returns:
        Grid: Grid object loaded from dictionary representation
    """
    grid = Grid(
        np.array(cast(Dict[str, object], gridDict["xGrid"])["cellCentreCoords"]),
        np.array(cast(Dict[str, object], gridDict["vGrid"])["cellCentreCoords"]),
        cast(Dict[str, object], gridDict["vGrid"])["maxL"],
        cast(Dict[str, object], gridDict["vGrid"])["maxM"],
        isPeriodic=cast(Dict[str, object], gridDict["xGrid"])["isPeriodic"],
        isLengthInMeters=cast(Dict[str, object], gridDict["xGrid"])["isLengthInMeters"],
    )

    grid.xJacobian = np.array(
        cast(Dict[str, object], gridDict["xGrid"])["faceJacobians"]
    )
    return grid
