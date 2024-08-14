import numpy as np
import xarray as xr
from .grid import Grid
from typing import Union, List, Dict, cast, Any
import warnings


class VariableContainer:
    """Contains an xarray dataset corresponding to ReMKiT1D data and methods used to manipulate that data"""

    def __init__(self, gridObj: Grid) -> None:
        self.dataset = xr.Dataset(
            coords={"x": gridObj.xGrid, "h": range(gridObj.numH()), "v": gridObj.vGrid}
        )

    def initFromDataset(self, dataset: xr.Dataset) -> None:
        self.dataset = dataset

    def setVariable(
        self,
        name: str,
        data: Union[np.ndarray, None] = None,
        isDerived=False,
        isDistribution=False,
        units="normalized units",
        isStationary=False,
        isScalar=False,
        isOnDualGrid=False,
        priority=0,
        derivationRule: Union[None, dict] = None,
        normSI: float = 1.0,
        unitSI: str = "",
    ):
        """Sets values and attributes for variable in dataset

        Args:
            name (str): Variable names
            data (Union[numpy.ndarray,None], optional): Optional numpy array representing variable data. Defaults to None, which initializes data to 0.
            isDerived (bool, optional): True if the variable is treated as derived by ReMKiT1D. Defaults to False.
            isDistribution (bool, optional): True for distribution-like variables. Defaults to False.
            units (str, optional): Variable units. Defaults to 'normalized units'.
            isStationary (bool, optional): True if the variable is stationary (d/dt = 0). Defaults to False.
            isScalar (bool, optional): True if the variable is a scalar. Defaults to False.
            isOnDualGrid (bool, optional): True if the variable is defined on dual grid. Defaults to False.
            priority (int, optional): Variable priority used in things like derivation call in integrators. Defaults to 0 (highest priority).
            derivationRule (Union[None,dict], optional) Optional derivation rule for derived variables. Defaults to None.
            normSI (float, optional) Optional normalisation constant for converting value to SI. Defaults to 1.0.
            unitSI (str, optional) Optional associated SI unit. Defaults to "".
        """

        assert name not in ["x", "h", "v"], (
            name
            + " is not an allowed variable name as it is associated with one of the coordinate dimensions"
        )
        dims: Union[List[str], None] = ["x"]
        dataShape = [len(self.dataset.coords["x"])]
        if isDistribution:
            cast(List[str], dims).append("h")
            dataShape.append(len(self.dataset.coords["h"]))
            cast(List[str], dims).append("v")
            dataShape.append(len(self.dataset.coords["v"]))

        if isScalar:
            dims = None
            dataShape = [1]

        if data is not None:
            usedData = data
            if isOnDualGrid:
                warnings.warn(
                    "Variable on dual grid "
                    + name
                    + " has been initialised with non-zero data. Make sure that the rightmost cell is zeroed out or intentionally left as non-zero."
                )
        else:
            usedData = np.zeros(dataShape)

        derivRule: Union[Dict[Any, Any], str] = "none"

        if isDerived:
            if derivationRule is not None:
                derivRule = derivationRule

        dataArr = xr.DataArray(
            usedData,
            dims=dims,
            attrs={
                "isDerived": isDerived,
                "isDistribution": isDistribution,
                "units": units,
                "isStationary": isStationary,
                "isScalar": isScalar,
                "isOnDualGrid": isOnDualGrid,
                "priority": priority,
                "derivationRule": derivRule,
                "normSI": normSI,
                "unitSI": unitSI,
            },
        )

        self.dataset[name] = dataArr

    def getVarAttrs(self, varName: str) -> dict:
        """Get attributes associated with given variable

        Args:
            varName (str): Variable name

        Returns:
            dict: Attribute dictionary
        """

        return self.dataset[varName].attrs

    def dict(self, outputVals=True) -> dict:
        """Return dictionary form of variable container for json output

        Args:
            outputVals (bool, optional): True if initial values should be written to dictionary. Defaults to True.

        Returns:
            dict: ReMKiT1D-ready dictionary form of variable data
        """

        variableList = self.dataset.data_vars.keys()
        implicitVars = []
        derivedVars = []

        # Sort variables into derived/implicit
        for key in variableList:
            if self.dataset.data_vars[key].attrs["isDerived"]:
                derivedVars.append(key)
            else:
                implicitVars.append(key)

        # Dictionaries for output
        impDict = {"names": implicitVars}
        derDict = {"names": derivedVars}

        for var in implicitVars:
            varProps = {}
            varProps = {
                "isDistribution": self.dataset.data_vars[var].attrs["isDistribution"],
                "isScalar": self.dataset.data_vars[var].attrs["isScalar"],
                "isStationary": self.dataset.data_vars[var].attrs["isStationary"],
                "isOnDualGrid": self.dataset.data_vars[var].attrs["isOnDualGrid"],
                "priority": self.dataset.data_vars[var].attrs["priority"],
            }
            if outputVals:
                varProps["initVals"] = (
                    self.dataset.data_vars[var].data.flatten().tolist()
                )

            cast(Dict[str, object], impDict)[var] = varProps

        for var in derivedVars:
            varProps = {}
            varProps = {
                "isDistribution": self.dataset.data_vars[var].attrs["isDistribution"],
                "isScalar": self.dataset.data_vars[var].attrs["isScalar"],
                "isStationary": self.dataset.data_vars[var].attrs["isStationary"],
                "isOnDualGrid": self.dataset.data_vars[var].attrs["isOnDualGrid"],
                "priority": self.dataset.data_vars[var].attrs["priority"],
            }
            if outputVals:
                varProps["initVals"] = (
                    self.dataset.data_vars[var].data.flatten().tolist()
                )

            if self.dataset.data_vars[var].attrs["derivationRule"] != "none":
                varProps["derivationRule"] = self.dataset.data_vars[var].attrs[
                    "derivationRule"
                ]

            cast(Dict[str, object], derDict)[var] = varProps

        variableData = {
            "variables": {"implicitVariables": impDict, "derivedVariables": derDict}
        }

        return variableData
