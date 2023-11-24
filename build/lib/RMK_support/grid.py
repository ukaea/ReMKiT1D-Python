import numpy as np
import json
from typing import List


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
            xGrid (numpy.ndarray): x coordinates of each spatial grid cell or their widths
            vGrid (numpy.ndarray): v coordinates of each velocity grid cell or their widths. Defaults to a single cell (effectively no v-grid)
            lMax (int): Maximum l harmonic number. Defaults to 0.
            mMax (int, optional): Maximum m harmonic number. Defaults to 0.
            interpretXGridAsWidths (bool, optional): If True interprets xGrid as cell widths. Defaults to False.
            interpretVGridAsWidths (bool, optional): If True interprets vGrid as cell widths. Defaults to False.
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

    def numX(self):
        return len(self.__xGrid__)

    def numV(self):
        return len(self.__vGrid__)

    def numH(self):
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

    def __repr__(self) -> str:
        return json.dumps(self.dict(), indent=4, sort_keys=True)

    def velocityMoment(
        self, distFun: np.ndarray, momentOrder: int, momentHarmonic=1
    ) -> np.ndarray:
        """Return velocity moment of distribution function or single harmonic variable

        Args:
            distFun (np.ndarray): Distribution or single harmonic variable values
            momentOrder (int): Moment order
            momentHarmonic (int, optional): Harmonic index (Fortran 1 indexing) to take moment of in case of distribution variable. Defaults to 1.

        Returns:
            np.ndarray: Moment represented as a contracted array
        """

        moment = np.zeros(np.shape(distFun)[0])
        if len(np.shape(distFun)) == 3:
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
