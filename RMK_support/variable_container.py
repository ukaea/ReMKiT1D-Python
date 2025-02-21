import numpy as np
import xarray as xr
from .grid import Grid
from typing import Union, List, Dict, cast, Any, Type, Tuple, Optional
import warnings
from .derivations import (
    DerivBase,
    Derivation,
    NodeDerivation,
    DerivationArgument,
    InterpolationDerivation,
    DerivationContainer,
    Textbook,
)
from . import calculation_tree_support as ct
from typing_extensions import Self
from copy import copy, deepcopy
from pylatex import Document, Section, Subsection, Itemize, NoEscape  # type: ignore
from math import isclose


class MultiplicativeArgument:
    """A multiplicative argument composed of variables raised to powers and an optional multiplicative scalar"""

    def __init__(self, *args: Tuple[DerivationArgument, float]):

        self.__argBuffer__: List[Tuple[DerivationArgument, float]] = list(args)

        self.__scalar__: Union[float, int] = 1

    @property
    def argMultiplicity(self) -> Dict[str, float]:
        """The multiplicity of each argument (the power it is effectively being raised to)

        Returns:
            Dict[str,float]: Dictionary with variable names as keys and their powers as values
        """
        multiplicity: Dict[str, float] = {}

        for arg, power in self.__argBuffer__:
            if arg.name in multiplicity:
                multiplicity[arg.name] += power
            else:
                multiplicity[arg.name] = power
            if isclose(multiplicity[arg.name], 0.0, rel_tol=1e-4):
                multiplicity.pop(arg.name)
        return multiplicity

    @property
    def args(self):
        """Dictionary containing all of the multiplicative variables, with their names as keys"""
        return {arg.name: arg for arg, _ in self.__argBuffer__}

    @property
    def firstArg(self):
        """The first argument - used for determining implicitness in matrix terms, for example"""
        return self.__argBuffer__[0][0]

    @property
    def scalar(self):
        """The scalar component of the multiplicative argument"""
        return self.__scalar__

    @scalar.setter
    def scalar(self, val: Union[int, float]):
        self.__scalar__ = val

    def __mul__(self, rhs: Union[DerivationArgument, Self, float, int]) -> Self:
        if isinstance(rhs, DerivationArgument):
            newArgs = deepcopy(self)
            newArgs.__argBuffer__ = [(rhs, 1.0)] + newArgs.__argBuffer__
        elif isinstance(rhs, MultiplicativeArgument):
            newArgs = cast(Self, deepcopy(rhs))
            newArgs.__argBuffer__ += self.__argBuffer__
            newArgs.__scalar__ *= self.__scalar__
        elif isinstance(rhs, (float, int)):
            newArgs = deepcopy(self)
            newArgs.__scalar__ *= rhs
        else:
            return rhs.__rmul__(self)
        return newArgs

    def __rmul__(self, lhs: Union[float, int]) -> Self:
        assert isinstance(
            lhs, (float, int)
        ), "MultiplicativeArgument can only be left-multiplied by a float or int"
        return deepcopy(self) * lhs

    def __truediv__(self, rhs: Union[DerivationArgument, Self, float, int]) -> Self:
        if isinstance(rhs, DerivationArgument):
            newArgs = deepcopy(self)
            newArgs.__argBuffer__.append((rhs, -1.0))
        if isinstance(rhs, MultiplicativeArgument):
            newArgs = cast(Self, deepcopy(self))
            for arg, power in rhs.__argBuffer__:
                newArgs.__argBuffer__.append((arg, -power))
            newArgs.__scalar__ /= rhs.__scalar__
        if isinstance(rhs, (float, int)):
            newArgs = deepcopy(self)
            newArgs.__scalar__ /= rhs

        return newArgs

    def __pow__(self, rhs: Union[float, int]) -> Self:
        newArgs = cast(Self, MultiplicativeArgument())
        for arg, power in self.__argBuffer__:
            newArgs.__argBuffer__.append((arg, rhs * power))
        newArgs.__scalar__ = self.scalar**rhs
        return newArgs

    def __matmul__(self, rhs):
        return rhs @ self

    def __neg__(self):
        newArgs = deepcopy(self)
        newArgs.scalar *= -1
        return newArgs

    def latex(self, latexRemap: Dict[str, str] = {}) -> str:
        """Generate latex representation - does not include the scalar component!

        Args:
            latexRemap (Dict[str, str], optional): Optional remapping of variable names. Defaults to {}.

        Returns:
            str: LaTeX-compatible string
        """
        numerator = ""
        denominator = ""

        for key in self.argMultiplicity:
            var = (
                latexRemap[key]
                if key in latexRemap
                else "\\text{" + key.replace("_", r"\_") + "}"
            )
            if self.argMultiplicity[key] > 0:
                if isclose(self.argMultiplicity[key], 1.0, rel_tol=1e-4):
                    numerator += " " + var
                else:
                    power = (
                        f"{round(self.argMultiplicity[key])}"
                        if isclose(
                            self.argMultiplicity[key],
                            round(self.argMultiplicity[key]),
                            rel_tol=1e-2,
                        )
                        else f"{self.argMultiplicity[key]:.2f}"
                    )
                    numerator += " " + var + "^{" + power + "}"
            else:
                if isclose(self.argMultiplicity[key], -1.0, rel_tol=1e-4):
                    denominator += " " + var
                else:
                    power = (
                        f"{round(-self.argMultiplicity[key])}"
                        if isclose(
                            self.argMultiplicity[key],
                            round(self.argMultiplicity[key]),
                            rel_tol=1e-2,
                        )
                        else f"{-self.argMultiplicity[key]:.2f}"
                    )
                    denominator += " " + var + "^{" + power + "}"
        if len(denominator):
            if numerator == "":
                numerator = "1"
            return "\\frac{" + numerator + "}{" + denominator + "}"

        return numerator


class Variable(DerivationArgument):
    """ReMKiT1D variable class used for defining and manipulating variable data"""

    def __init__(self, name: str, gridObj: Grid, **kwargs) -> None:
        """Construct a variable with given name on a given grid

        Args:
            name (str): Name of the variable
            gridObj (Grid): Grid on which the variable lives

        kwargs:

            data (numpy.ndarray): Optional numpy array representing variable data. Defaults to None, which initializes data to 0.
            isDerived (bool): True if the variable is treated as derived by ReMKiT1D. Defaults to False.
            isDistribution (bool): True for distribution-like variables. Defaults to False.
            units (str): Variable units. Defaults to 'normalized units'.
            isStationary (bool): True if the variable is stationary (d/dt = 0). Defaults to False.
            isScalar (bool): True if the variable is a scalar - automatically sets the variable to derived. Defaults to False.
            isSingleHarmonic (bool): True if the variable is a single harmonic (used currently only for modelbound data). Defaults to False.
            isOnDualGrid (bool): True if the variable is defined on dual grid. Defaults to False.
            priority (int): Variable priority used in things like derivation call in integrators. Defaults to 0 (highest priority).
            derivation (Derivation): The derivation object associated with this variable. If present will make the variable automatically derived.
            derivationArgs (List[str]): Names of derivation arguments, will be modified according to any enclosed variables in the derivation itself.
            normSI (float): Optional normalisation constant for converting value to SI. Defaults to 1.0.
            unitSI (str): Optional associated SI unit. Defaults to "".
            timeDimSize (int): The size of the time dimension. Defaults to 1, omitting the dimension.
            isCommunicated (bool): Whether the variable is communicated. Defaults to True
            scalarHostProcess (int): The process on which the variable lives if it is a scalar. Defaults to 0.
            inOutput (int): True if the variable should be added to any hdf5 output. Defaults to True.
            subtype (str): Denotes subtype of variable. Useful for filtering variables and working with species. Defaults to "untyped".
        """
        self.__grid__ = gridObj

        self.__name__ = name

        self.__derivation__: Optional[Derivation] = kwargs.get("derivation", None)
        self.__isDerived__: bool = kwargs.get("isDerived", False)
        if self.__derivation__ is not None:
            self.__isDerived__ = True
            if not kwargs.get("isDerived", True):
                warnings.warn(
                    "Variable "
                    + name
                    + " has derivation rule set, but is explicitly set to not be derived. Overriding to isDerived=True!"
                )

        self.__derivationArgs__: List[str] = kwargs.get("derivationArgs", [])
        if not self.__isDerived__ and "derivationArgs" in kwargs:
            warnings.warn(
                "derivationArgs set for variable "
                + name
                + " which is not derived. Ignoring..."
            )

        if self.__derivation__ is not None:
            self.__derivationArgs__ = self.__derivation__.fillArgs(
                *tuple(self.__derivationArgs__)
            )
        if isinstance(self.__derivation__, NodeDerivation):
            self.__derivationArgs__ = ct.getLeafVars(
                cast(NodeDerivation, self.__derivation__).node
            )
            if "derivationArgs" in kwargs:
                warnings.warn(
                    "derivationArgs set for variable "
                    + name
                    + " which is produced by a NodeDerivation. Ignoring in favour of node leaf variables."
                )

        self.__isDistribution__: bool = kwargs.get("isDistribution", False)
        self.__isSingleHarmonic__: bool = kwargs.get("isSingleHarmonic", False)
        assert not (self.__isDistribution__ and self.__isSingleHarmonic__), (
            "Variable " + name + " cannot be both full distribution and single harmonic"
        )

        self.__units__ = kwargs.get("units", "normalized units")
        self.__normUnits__ = self.__units__
        self.__isStationary__: bool = kwargs.get("isStationary", False)
        if self.__isStationary__:
            assert not self.__isDerived__, (
                "INVALID: Derived variable " + name + " set as stationary"
            )
        self.__isScalar__: bool = kwargs.get("isScalar", False)
        if self.__isScalar__:
            self.__isDerived__ = True
            if not kwargs.get("isDerived", True):
                warnings.warn(
                    "Variable "
                    + name
                    + " has is scalar, but is explicitly set to not be derived. Overriding to isDerived=True!"
                )

        self.__isOnDualGrid__: bool = kwargs.get("isOnDualGrid", False)
        if "priority" in kwargs:
            assert self.__isDerived__, (
                "INVALID: Priority set for implicit variable " + name
            )
        self.__priority__: int = kwargs.get("priority", 0)
        self.__normSI__: float = kwargs.get("normSI", 1.0)
        self.__unitSI__: str = kwargs.get("unitSI", "")
        self.__normConst__: float = self.__normSI__
        self.__timeDimSize__: int = kwargs.get("timeDimSize", 0)

        assert name not in ["x", "x_dual", "h", "v", "t"], (
            name
            + " is not an allowed variable name as it is associated with one of the coordinate dimensions"
        )

        dataTemp: Union[np.ndarray, None] = kwargs.get("data", None)

        if dataTemp is not None:
            self.__data__: np.ndarray = dataTemp
            if self.__isOnDualGrid__:
                warnings.warn(
                    "Variable on dual grid "
                    + name
                    + " has been initialised with non-zero data. Make sure that the rightmost cell is zeroed out or intentionally left as non-zero."
                )
        else:
            self.__data__ = np.zeros(self.dataShape)

        assert self.__data__.shape == tuple(self.dataShape), (
            "Non-conforming data in Variable constructor for variable " + name
        )
        self.__derivRule__: Union[Dict[Any, Any], str] = (
            "none"
            if self.__derivation__ is None
            else {
                "ruleName": self.__derivation__.name,
                "requiredVarNames": self.__derivationArgs__,
            }
        )

        self.__properties__ = {
            "isDerived": self.__isDerived__,
            "isDistribution": self.__isDistribution__,
            "units": self.__units__,
            "isStationary": self.__isStationary__,
            "isScalar": self.__isScalar__,
            "isOnDualGrid": self.__isOnDualGrid__,
            "priority": self.__priority__,
            "derivationRule": self.__derivRule__,
            "isSingleHarmonic": self.__isSingleHarmonic__,
            "normSI": self.__normSI__,
            "unitSI": self.__unitSI__,
        }

        self.__isCommunicated__: bool = kwargs.get("isCommunicated", True)
        if "scalarHostProcess" in kwargs and not self.__isScalar__:
            warnings.warn("scalarHostProcess set for non-scalar variable - ignoring")
        self.__scalarHostProcess__: int = kwargs.get("scalarHostProcess", 0)

        self.__inOutput__: bool = kwargs.get("inOutput", True)

        self.__dual__: Union[Variable, None] = None
        self.__updateDataArr__ = True
        self.__dataArr__ = self.dataArr

        self.__subtype__ = kwargs.get("subtype", "untyped")

    @property
    def dual(self) -> Optional[Self]:
        """The dual variable to this variable. If no dual returns None."""
        return cast(Optional[Self], self.__dual__)

    @dual.setter
    def dual(self, var: Self):
        self.__dual__ = var
        var.__dual__ = self

    def makeDual(self, name: Optional[str] = None):
        """Create a dual variable from this variable, assigning the dual to self. If dual is already assigned will return existing dual instead of constructing new.

        Args:
            name (Optional[str], optional): Name of the dual variable. Defaults to None, setting the name to the name of this variable with the "_dual" suffix.
        """
        if self.dual is not None:
            return self.dual

        assert not self.isScalar, (
            "Scalar variable " + self.name + " cannot have a dual assigned"
        )
        assert not self.isSingleHarmonic, (
            "Single harmonic variable " + self.name + " cannot have a dual assigned"
        )

        dualDeriv = InterpolationDerivation(
            self.grid,
            True if self.isDistribution else not self.isOnDualGrid,
            self.isDistribution,
        )

        dualVar = Variable(
            name if name is not None else self.name + "_dual",
            self.grid,
            isDistribution=self.isDistribution,
            isOnDualGrid=True if self.isDistribution else not self.isOnDualGrid,
            derivation=dualDeriv,
            derivationArgs=[self.name],
            normSI=self.__normSI__,
            units=self.__normUnits__,
            unitSI=self.__unitSI__,
            isCommunicated=self.isCommunicated,
            inOutput=self.inOutput,
            subtype=self.__subtype__,
        )

        self.dual = dualVar

        return dualVar

    def withDual(self, name: Optional[str] = None):
        """Generate dual variable of self if not already associated, and return self

        Args:
            name (Optional[str], optional): Name of the dual variable. Defaults to None, setting the name to the name of this variable with the "_dual" suffix.
        """

        _ = self.makeDual(name)
        return self

    @property
    def data(self):
        return self.__data__

    @property
    def dataArr(self):
        if self.__updateDataArr__:
            self.__dataArr__ = xr.DataArray(
                self.__data__, dims=self.dims, attrs=self.__properties__
            )
            self.__updateDataArr__ = False
        return self.__dataArr__

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name: str):
        self.__name__ = name

    @property
    def subtype(self):
        return self.__subtype__

    def rename(self, name: str):
        """Create a copy of this variable and rename it. If the variable has a dual, it will be remade to avoid name clashes"""
        newVar = deepcopy(self)
        newVar.name = name
        if newVar.dual is not None:
            newVar.__dual__ = None
            _ = newVar.withDual()
        return newVar

    @property
    def values(self):
        return self.__data__

    @values.setter
    def values(self, data: np.ndarray):
        assert (
            self.__data__.shape == data.shape
        ), "data passed to values setter on Variable does not conform with dimensions"

        self.__data__ = data
        self.__updateDataArr__ = True

    @property
    def derivation(self):
        return self.__derivation__

    @property
    def derivationArgs(self):
        return self.__derivationArgs__

    @property
    def priority(self):
        return self.__priority__

    @property
    def grid(self):
        return self.__grid__

    def addTimeDim(self, data: np.ndarray):
        """Expand the variable into having a time dimension and assign new values to it. The time dimension is assumed to be the first dimension of data, with the others conforming to the variable dimensions.

        Args:
            data (np.ndarray): Data to assign to the expanded variable
        """
        if self.dims is not None:
            assert "t" not in self.dims, "t already Variable dim"
        assert (
            self.__data__.shape == (1,)
            and len(data.shape) == 1
            or self.__data__.shape == data.shape[1:]
        ), "data passed to addTimeDim does not conform with dimensions"
        self.__timeDimSize__ = data.shape[0]
        self.__data__ = data
        self.__updateDataArr__ = True

    @property
    def properties(self):
        return self.__properties__

    @property
    def isDistribution(self):
        return self.__isDistribution__

    @property
    def isSingleHarmonic(self):
        return self.__isSingleHarmonic__

    @property
    def isFluid(self):
        return (
            not self.__isDistribution__
            and not self.__isScalar__
            and not self.__isSingleHarmonic__
        )

    @property
    def isDerived(self):
        return self.__isDerived__

    @property
    def isOnDualGrid(self):
        return self.__isOnDualGrid__

    @property
    def unitsSI(self):
        return self.__unitSI__

    @unitsSI.setter
    def unitsSI(self, units: str):
        self.__unitSI__ = units
        self.__updateDataArr__ = True

    @property
    def units(self):
        return self.__units__

    @units.setter
    def units(self, units: str):
        self.__units__ = units
        self.__updateDataArr__ = True

    @property
    def normSI(self):
        return self.__normSI__

    @normSI.setter
    def normSI(self, norm: float):
        self.__normSI__ = norm
        self.__updateDataArr__ = True

    @property
    def unitsNorm(self):
        return self.__normUnits__

    @property
    def normConst(self):
        return self.__normConst__

    @property
    def dims(self) -> Optional[List[str]]:

        dims: Union[List[str], None] = ["x"] if not self.isOnDualGrid else ["x_dual"]

        if self.__isDistribution__:
            cast(List[str], dims).append("h")
            cast(List[str], dims).append("v")

        if self.__isSingleHarmonic__:
            cast(List[str], dims).append("v")

        if self.__isScalar__:
            dims = None

        if self.__timeDimSize__ > 0:
            dims = ["t"] + dims if dims is not None else ["t"]

        return dims

    @property
    def dataShape(self):

        if self.__isScalar__:
            return [1]

        dataShape = [self.grid.numX]

        if self.__isDistribution__:
            dataShape.append(self.grid.numH)
            dataShape.append(self.grid.numV)

        if self.__isSingleHarmonic__:
            dataShape.append(self.grid.numV)

        if self.__timeDimSize__ > 0:
            dataShape = (
                [self.__timeDimSize__] + dataShape
                if self.dims is not None
                else [self.__timeDimSize__]
            )

        return dataShape

    @property
    def isScalar(self):
        return self.__isScalar__

    @property
    def isStationary(self):
        return self.__isStationary__

    @property
    def isCommunicated(self):
        return self.__isCommunicated__

    @isCommunicated.setter
    def isCommunicated(self, comm=True):
        self.__isCommunicated__ = comm

    @property
    def scalarHostProcess(self):
        return self.__scalarHostProcess__

    @scalarHostProcess.setter
    def scalarHostProcess(self, proc: int):
        assert self.isScalar, (
            "Attempted to set scalarHostProcess on non-scalar variable " + self.name
        )
        self.__scalarHostProcess__ = proc

    @property
    def inOutput(self):
        return self.__inOutput__

    def onDualGrid(self, dual=True):
        """Return a copy of this variable setting it to live on the dual or regular grid. If the variable has a dual associated with it, it will be remade to avoid having duals that live on the same grid"""
        newVar = deepcopy(self)
        newVar.__isOnDualGrid__ = dual
        if newVar.dual is not None:
            newVar.__dual__ = None
            _ = newVar.withDual()
        return newVar

    def switchUnits(self):
        """Swtich between the units (norm and SI or vice-versa)"""
        if self.units == self.unitsNorm:

            self.__units__ = self.unitsSI
            self.__data__ *= self.normConst
            self.__normConst__ = 1 / self.__normSI__

        else:
            self.__units__ = self.__normUnits__
            self.__data__ *= self.normConst
            self.__normConst__ = self.__normSI__

        self.__updateDataArr__ = True

    @classmethod
    def apply(cls, deriv: DerivBase, *args: Self) -> Self:
        DerivationArgument.apply(deriv, *args)
        options = {
            "isDistribution": args[0].isDistribution,
            "isScalar": args[0].isScalar,
            "isSingleHarmonic": args[0].isSingleHarmonic,
        }
        options.update(cast(Derivation, deriv).resultProperties)
        derivationArgs = [arg.name for arg in args]
        return cast(
            Self,
            Variable(
                deriv.name,
                args[0].grid,
                derivation=deriv,
                derivationArgs=derivationArgs,
                **options,
            ),
        )

    def latex(self, latexRemap: Dict[str, str] = {}) -> str:
        """Generate LaTeX representation of the variable, including any derivation rules

        Args:
            latexRemap (Dict[str, str], optional): Optional variable name latex remap. Defaults to {}.

        Returns:
            str: LaTeX-compatible string
        """
        result = (
            "\\text{" + self.name.replace("_", r"\_") + "}"
            if self.name not in latexRemap
            else latexRemap[self.name]
        )
        if self.derivation is not None:
            remappedArgs = (
                (
                    latexRemap[arg]
                    if arg in latexRemap
                    else "\\text{" + arg.replace("_", r"\_") + "}"
                )
                for arg in self.__derivationArgs__
            )
            result += "= " + cast(Derivation, self.__derivation__).latex(*remappedArgs)

        return result

    def __pow__(self, rhs: Union[float, int]) -> MultiplicativeArgument:
        return MultiplicativeArgument((self, float(rhs)))

    def __mul__(
        self, rhs: Union[Self, MultiplicativeArgument, float, int]
    ) -> MultiplicativeArgument:
        if isinstance(rhs, MultiplicativeArgument):
            return MultiplicativeArgument((self, 1.0)) * rhs
        elif isinstance(rhs, Variable):
            return MultiplicativeArgument((rhs, 1.0), (self, 1.0))
        elif isinstance(rhs, (int, float)):
            return rhs * MultiplicativeArgument((self, 1.0))
        else:
            return rhs.__rmul__(self)

    def __rmul__(self, lhs: Union[float, int]) -> MultiplicativeArgument:
        assert isinstance(
            lhs, (int, float)
        ), "Variable can only be left-multiplied by floats or ints"
        return lhs * MultiplicativeArgument((self, 1.0))

    def __matmul__(self, rhs):
        return rhs @ self

    def __truediv__(
        self, rhs: Union[Self, MultiplicativeArgument, float, int]
    ) -> MultiplicativeArgument:
        if isinstance(rhs, MultiplicativeArgument):
            return (rhs ** (-1)) * self
        if isinstance(rhs, Variable):
            return MultiplicativeArgument((self, 1.0), (cast(Variable, rhs), -1.0))
        if isinstance(rhs, (float, int)):
            return MultiplicativeArgument((self, 1.0)) / rhs

    def __neg__(self):
        newArg = MultiplicativeArgument((self, 1.0))
        newArg.scalar = -1.0
        return newArg

    def evaluate(self, dataset: xr.Dataset) -> np.ndarray:
        """Attempt to evaluate the variable based on passed dataset. Variables with no derivations result in just their values.
        **NOTE**: Not all derivations have an implemented evaluate function

        Args:
            dataset (xr.Dataset): Dataset containing any required variable values

        Returns:
            np.ndarray: Evaluation result
        """
        if self.isDerived:
            self.__data__ = self.derivation.evaluate(
                *tuple(dataset[name].data for name in self.__derivationArgs__)
            )
            self.__updateDataArr__ = True

        return self.data


def node(var: Variable) -> ct.Node:
    """Create a Node from a Variable"""
    return ct.Node(var.name)


def varFromNode(
    name: str,
    gridObj: Grid,
    node: ct.Node,
    container: Optional[DerivationContainer] = None,
    **kwargs,
) -> Variable:
    """Wrapper for creating a variable from a node. Takes in all of the same kwargs as the Variable constructor

    Args:
        name (str): Name of the variable
        gridObj (Grid): Grid on which the variable lives
        node (ct.Node): Node used to derive the variable values
        container (Union[DerivationContainer,None], optional): Optional container into which to register the derivation, generally not used. Defaults to None.
    """
    # TODO: add checking for conflicting kwargs
    return Variable(
        name,
        gridObj,
        derivation=NodeDerivation(name, node, container=container),
        **kwargs,
    )


def varAndDual(
    name: str, gridObj: Grid, primaryOnDualGrid=False, dualSuffix="_dual", **kwargs
) -> Tuple[Variable, Variable]:
    """Return variable and its dual, making sure that the correct derivations are set. Takes in all of the same kwargs as the Variable constructor, and they refer to the primary variable.

    Args:
        name (str): Name of the regular grid variable
        gridObj (Grid): Grid on which both the variables live
        primaryOnDualGrid (bool, optional): If true the primary variable is the dual grid variable. Defaults to False.
        dualSuffix (str, optional): Name suffix for dual variable. Defaults to "_dual".

    Returns:
        Tuple[Variable,Variable]: The primary,secondary variable tuple. **NOTE**: The dual will be the first element of the tuple if it is the primary.
    """
    assert (
        "isOnDualGrid" not in kwargs
    ), "isOnDualGrid is not a valid kwarg for varAndDual"

    primaryName = name + dualSuffix if primaryOnDualGrid else name
    secondaryName = name if primaryOnDualGrid else name + dualSuffix

    isDistribution = kwargs.get("isDistribution", False)

    primary = Variable(
        primaryName, gridObj, isOnDualGrid=primaryOnDualGrid or isDistribution, **kwargs
    )

    secondary = primary.makeDual(secondaryName)

    return primary, secondary


class VariableContainer:
    """Container object for Variables"""

    def __init__(
        self, gridObj: Grid, timestamps: np.ndarray = np.array([]), autoAddDuals=True
    ) -> None:
        """Container object for Variables

        Args:
            gridObj (Grid): Grid on which all the variables will live
            timestamps (np.ndarray, optional): Timestamps in case a time dimension should be added. Defaults to np.array([]).
            autoAddDuals (bool, optional): If true will automatically add dual variables if associated at time of adding. Defaults to True.
        """
        self.__coords__ = {
            "x": gridObj.xGrid,
            "x_dual": gridObj.xGridDual,
            "h": np.array(range(gridObj.numH)),
            "v": gridObj.vGrid,
        }
        if len(timestamps) > 0:
            self.__coords__["t"] = timestamps
        self.__dataset__ = xr.Dataset(coords=self.__coords__)

        self.__timestamps__ = timestamps
        self.__dataset__.coords["x"].attrs["units"] = "$x_0$"
        self.__dataset__.coords["x_dual"].attrs["units"] = "$x_0$"
        if gridObj.isLengthInMeters:
            self.__dataset__.coords["x"].attrs["units"] = "m"
            self.__dataset__.coords["x_dual"].attrs["units"] = "m"
        self.__dataset__.coords["v"].attrs["units"] = "$v_{th}$"
        if len(timestamps) > 0:
            self.__dataset__.coords["t"].attrs["units"] = "$t_0$"
        self.__coordUnits__ = {
            self.__dataset__.coords[coord].attrs["units"]
            for coord in self.__dataset__.coords
            if "units" in self.__dataset__.coords[coord].attrs
        }

        self.__grid__ = gridObj
        self.__variables__: List[Variable] = [
            Variable(
                "time", gridObj, isDerived=True, isScalar=True, isCommunicated=False
            )
        ]

        self.__autoAddDuals__ = autoAddDuals

    def add(self, *args: Variable):
        """Add any number of variables to the container"""
        for var in args:
            assert not var.isSingleHarmonic, (
                "Single harmonic variable "
                + var.name
                + "in add - VariableContainers do not support single harmonics"
            )

            if var.name in self.varNames:
                warnings.warn(
                    "Variable "
                    + var.name
                    + " already in VariableContainer. Overwriting."
                )
                self.__variables__[self.varNames.index(var.name)] = var
            else:
                self.__variables__.append(var)

            if self.__autoAddDuals__ and var.dual is not None:
                if var.dual.name in self.varNames:
                    warnings.warn(
                        "Variable "
                        + var.dual.name
                        + " already in VariableContainer. Overwriting."
                    )
                    self.__variables__[self.varNames.index(var.dual.name)] = var.dual
                else:
                    self.__variables__.append(var.dual)

    def setVar(
        self,
        name: str,
        **kwargs,
    ):
        """Construct a variable in place in the container. Takes in the same kwargs as the Variable constructor.

        Args:
            name (str): Name of the regular grid variable
        """

        self.add(Variable(name, self.__grid__, **kwargs))

    def __getitem__(self, key: str):
        if key not in self.varNames:
            raise KeyError()
        return self.__variables__[self.varNames.index(key)]

    def __setitem__(self, key: str, var: Variable):
        if key not in self.varNames:
            self.add(var.rename(key))
        self.__variables__[self.varNames.index(key)] = var.rename(key)

    def getVarAttrs(self, varName: str) -> Dict:
        """Get attributes associated with given variable

        Args:
            varName (str): Variable name

        Returns:
            dict: Attribute dictionary
        """

        return self[varName].properties

    def dict(self, outputVals=True) -> dict:
        """Return dictionary form of variable container for json output

        Args:
            outputVals (bool, optional): True if initial values should be written to dictionary. Defaults to True.

        Returns:
            dict: ReMKiT1D-ready dictionary form of variable data
        """

        ds = self.dataset
        variableList = ds.data_vars.keys()
        implicitVars = []
        derivedVars = []

        # Sort variables into derived/implicit
        for key in variableList:
            if ds.data_vars[key].attrs["isDerived"]:
                derivedVars.append(key)
            else:
                implicitVars.append(key)

        # Dictionaries for output
        impDict = {"names": implicitVars}
        derDict = {"names": derivedVars}

        for var in implicitVars:
            varProps = {}
            varProps = {
                "isDistribution": ds.data_vars[var].attrs["isDistribution"],
                "isScalar": ds.data_vars[var].attrs["isScalar"],
                "isStationary": ds.data_vars[var].attrs["isStationary"],
                "isOnDualGrid": ds.data_vars[var].attrs["isOnDualGrid"],
                "priority": ds.data_vars[var].attrs["priority"],
            }
            if outputVals:
                varProps["initVals"] = ds.data_vars[var].data.flatten().tolist()

            cast(Dict[str, object], impDict)[var] = varProps

        for var in derivedVars:
            varProps = {}
            varProps = {
                "isDistribution": ds.data_vars[var].attrs["isDistribution"],
                "isScalar": ds.data_vars[var].attrs["isScalar"],
                "isStationary": ds.data_vars[var].attrs["isStationary"],
                "isOnDualGrid": ds.data_vars[var].attrs["isOnDualGrid"],
                "priority": ds.data_vars[var].attrs["priority"],
            }
            if outputVals:
                varProps["initVals"] = ds.data_vars[var].data.flatten().tolist()

            if ds.data_vars[var].attrs["derivationRule"] != "none":
                varProps["derivationRule"] = ds.data_vars[var].attrs["derivationRule"]

            cast(Dict[str, object], derDict)[var] = varProps

        variableData = {
            "variables": {"implicitVariables": impDict, "derivedVariables": derDict}
        }

        return variableData

    def addLatexToDoc(self, doc: Document, latexRemap: Dict[str, str] = {}) -> None:
        """Add variable section to a ReMKiT1D summary LaTeX doc

        Args:
            doc (Document): pylatex Document to add the section to
            latexRemap (Dict[str, str], optional): Optional remapping of variable names. Defaults to {}.
        """
        implicitVars = self.implicitVars
        derivedVars = self.derivedVars
        with doc.create(Section("Variables")):
            with doc.create(Subsection("Implicit variables")):
                with doc.create(Itemize()) as itemize:
                    for var in implicitVars:
                        itemize.add_item(NoEscape(f"${var.latex(latexRemap)}$"))
            with doc.create(Subsection("Derived variables")):
                with doc.create(Itemize()) as itemize:
                    for var in derivedVars:
                        itemize.add_item(NoEscape(f"${var.latex(latexRemap)}$"))

    def registerDerivs(self, textbook: Textbook):
        """Register any derivations on derived variables in given textbook

        Args:
            textbook (Textbook): Textbook to contain the derivations
        """
        for var in self.derivedVars:
            if var.derivation is not None:
                try:
                    _ = textbook[var.derivation.name]
                except KeyError:

                    textbook.register(var.derivation)

    def checkDerivationArgs(self) -> None:
        """Check whether all derived variable derivation arguments are present in the container"""
        for var in self.derivedVars:
            for name in var.derivationArgs:
                assert name in self.varNames, (
                    "Required derivation variable "
                    + name
                    + " not registered in used variable container"
                )

    @property
    def coords(self):
        return self.__coords__

    @property
    def dataset(self):
        ds = self.__dataset__.copy()

        for var in self.__variables__:
            ds[var.name] = var.dataArr

        return ds

    @property
    def varNames(self):
        return [v.name for v in self.__variables__]

    @property
    def implicitVars(self):
        return [v for v in self.__variables__ if not v.isDerived]

    @property
    def derivedVars(self):
        return [v for v in self.__variables__ if v.isDerived]

    @property
    def variables(self):
        return self.__variables__

    @property
    def timestamps(self):
        return self.__timestamps__


class MPIContext:

    def __init__(self, numProcsX: int, numProcsH=1, xHaloWidth=1):
        """Container for MPI options for ReMKiT1D

        Args:
            numProcsX (int): The number of processes in the x direction
            numProcsH (int, optional): The number of processes in the h direction. Defaults to 1.
            xHaloWidth (int, optional): The width of the halo in the x direction. Defaults to 1.
        """
        self.__numProcsX__ = numProcsX
        self.__numProcsH__ = numProcsH
        self.__xHaloWidth__ = xHaloWidth

    @property
    def numProcsX(self):
        return self.__numProcsX__

    @numProcsX.setter
    def numProcsX(self, numProcs: int):
        self.__numProcsX__ = numProcs

    @property
    def numProcsH(self):
        return self.__numProcsH__

    @numProcsH.setter
    def numProcsH(self, numProcs: int):
        self.__numProcsH__ = numProcs

    @property
    def numProcs(self):
        return self.numProcsX * self.numProcsH

    @property
    def fluidProcs(self):
        """Return a list of process indices corresponding to the lowest harmonics/fluid variables"""
        return list(range(0, self.numProcs, self.numProcsH))

    def addToComm(self, var: Variable, mpiComm=Dict[str, object]) -> None:

        if var.isDistribution:
            cast(
                List[str],
                cast(Dict[str, object], mpiComm)["varsToBroadcast"],
            ).append(var.name)
            cast(
                List[str],
                cast(Dict[str, object], mpiComm)["haloExchangeVars"],
            ).append(var.name)
        elif var.isScalar:
            cast(
                List[str],
                cast(Dict[str, object], mpiComm)["scalarVarsToBroadcast"],
            ).append(var.name)
            assert var.scalarHostProcess <= self.numProcs, (
                "Variable "
                + var.name
                + " has scalar host process out of MPI rank bounds"
            )
            cast(
                List[str],
                cast(Dict[str, object], mpiComm)["scalarBroadcastRoots"],
            ).append(var.scalarHostProcess)
        else:
            cast(
                List[str],
                cast(Dict[str, object], mpiComm)["haloExchangeVars"],
            ).append(var.name)
            if self.numProcsH > 1:
                cast(
                    List[str],
                    cast(Dict[str, object], mpiComm)["varsToBroadcast"],
                ).append(var.name)

    def dict(self, varCont: VariableContainer) -> dict:

        mpiDict = {
            "numProcsX": self.numProcsX,
            "numProcsH": self.numProcsH,
            "xHaloWidth": self.__xHaloWidth__,
        }

        mpiCommData: Dict[str, object] = {
            "varsToBroadcast": [],
            "haloExchangeVars": [],
            "scalarVarsToBroadcast": [],
            "scalarBroadcastRoots": [],
        }

        for var in varCont.variables:
            if var.isCommunicated:
                self.addToComm(var, mpiCommData)
        mpiDict["commData"] = mpiCommData

        return mpiDict
