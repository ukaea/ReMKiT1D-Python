import xarray as xr
import holoviews as hv  # type: ignore
import panel as pn
import numpy as np
from .grid import Grid
from .variable_container import VariableContainer, Variable
from . import IO_support as io
from typing import List, Dict, Tuple, Type, ClassVar
from typing_extensions import Self
import param  # type: ignore
from warnings import warn


def fluidVarX(var: Variable, time: int) -> hv.Curve:
    """Generate holoviews curve with the spatial profile of a given fluid variable

    Args:
        var (Variable): Fluid variable to plot
        time (int): Time index to plot the spatial profile for

    Returns:
        hv.Curve: Holoviews curve of fluid variable spatial profile
    """

    assert var.isFluid, "fluidVarX accepts only fluid variables"

    return hv.Curve(
        (
            var.grid.xGridDual if var.isOnDualGrid else var.grid.xGrid,
            var.dataArr[time, :],
        ),
        "$x[m]$",
        var.units,
        label=var.name,
    )


def fluidVarT(var: Variable, x: int, time: Variable) -> hv.Curve:
    """Produce a holoviews time profile of a given variable for a given spatial location

    Args:
        var (Variable): Fluid variable to plot
        x (int): Spatial index to plot the timetrace for
        time (Variable): Scalar time variable corresponding to the time values

    Returns:
        hv.Curve: Holoviews curve of fluid variable timetrace
    """

    assert var.isFluid, "fluidVarT accepts only fluid variables"
    assert time.isScalar, "The time variable in fluidVarT must be a scalar"

    return hv.Curve(
        (time.dataArr, var.dataArr[:, x]), "$t[t_0]$", var.units, label=var.name
    )


def distVarV(
    var: Variable, x: int, time: int, harmonic: int, energyGrid: bool = False
) -> hv.Curve:
    """Produce a holoviews curve of a distribution variable velocity space profile

    Args:
        var (Variable): Distribution variable to plot
        x (int): Spatial index for which to plot the profile
        time (int): Time index for which to plot the profile
        harmonic (int): Harmonic index for which to plot the profile
        energyGrid (bool, optional): If true will plot against v**2 instead of v. Defaults to False.

    Returns:
        hv.Curve: Holoviews curve of distribution variable velocity profile
    """

    assert var.isDistribution, "distVarV accepts only distribution variables"
    if energyGrid:
        curve = hv.Curve(
            (
                var.grid.vGrid**2,
                var.dataArr[time, x, harmonic, :],
            ),
            "$v^2[v_0^2]$",
            var.units,
            label=var.name,
        )
    else:

        curve = hv.Curve(
            (
                var.grid.vGrid,
                var.dataArr[time, x, harmonic, :],
            ),
            "$v[v_0]$",
            var.units,
            label=var.name,
        )

    return curve


def scalarVarT(var: Variable, time: Variable) -> hv.Curve:
    """Produce a holoviews curve of the timetrace for a given scalar variable

    Args:
        var (Variable): Scalar variable to plot
        time (Variable): Scalar time variable corresponding to the time values

    Returns:
        hv.Curve: Holoviews curve of scalar variable timetrace
    """

    assert var.isScalar, "scalarVarT accepts only scalar variables"
    assert time.isScalar, "The time variable in scalarVarT must be a scalar"
    return hv.Curve(
        (time.dataArr, var.dataArr[:]), "$t[t_0]$", var.units, label=var.name
    )


class LazyLoader:
    """Loader class allowing for the dynamic loading of Variables in a given VariableContainer from a number of different paths"""

    def __init__(self, variables: VariableContainer, runPaths: Dict[str, str]) -> None:
        """Loader allowing for lazy loading of variables from multiple paths

        Args:
            variables (VariableContainer): Variable container with variables that the loader can load
            runPaths (Dict[str,str]): Paths containing ReMKiT1D hdf5 output files. These must conform to the VariableContainer's properties.
        """
        self.__variables__ = variables
        self.__runPaths__: Dict[str, str] = runPaths
        assert (
            len(runPaths.keys()) > 0
        ), "At least one run path must be supplied to the LazyLoader"

        self.__runs__: Dict[str, VariableContainer] = {}

    def load(self, varName: str, run: str):
        """Load a variable with a given name from a given run. If it is already loaded this is a no-op.

        Args:
            varName (str): Name of the variable to load
            run (str): Key of the run path which contains the data
        """
        path = self.__runPaths__[run]
        if run not in self.__runs__:
            self.__runs__[run] = io.loadVarContFromHDF5(
                self.__variables__["time"],
                filepaths=[path + file for file in io.getOutputFilenames(path)],
            )

        if varName in self.__runs__[run].varNames:
            return

        loadedVar = io.loadVariableFromHDF5(
            self.__variables__[varName],
            filepaths=[path + file for file in io.getOutputFilenames(path)],
        )
        self.__runs__[run].add(loadedVar)

    def setRunPath(self, run: str, path: str):
        """Set the path of a given run in the loader. This can be used to add more runs.

        Args:
            run (str): Run key
            path (str): Run path
        """
        self.__runPaths__[run] = path

    def __getitem__(self, key: Tuple[str, str]) -> Variable:
        assert key[0] in self.__variables__.varNames, (
            key[0] + " not in LazyLoader variables"
        )
        assert key[1] in self.__runPaths__, key[1] + " not in LazyLoader run paths"

        self.load(key[0], key[1])

        return self.__runs__[key[1]][key[0]]

    def getDataset(self, run: str) -> xr.Dataset:
        """Retrieve the dataset associated with a given run. It will only contain the loaded variables.

        Args:
            run (str): Run key for which the dataset is extracted
        """
        return self.__runs__[run].dataset


class DashboardElement(param.Parameterized):
    """Base parameterised class used to extend dashboard element types"""

    alias = "Selector"
    __sc__: ClassVar[Dict[str, Type[Self]]] = {}

    def __init__(self, loader: LazyLoader, **params):
        self.__loader__ = loader
        super().__init__(**params)

    def __init_subclass__(cls) -> None:
        DashboardElement.__sc__[cls.alias] = cls

    def view(self):
        pass


class ElementDisplay(param.Parameterized):

    widget = param.Selector(default="Selector")
    element = param.Parameter(instantiate=False, precedence=-1)

    def __init__(self, loader: LazyLoader, **params):
        self.param["widget"].objects.append("Selector")
        for name in DashboardElement.__sc__:
            self.param["widget"].objects.append(name)
        self.__loader__ = loader
        self.__last__ = "Selector"
        self.element = DashboardElement(self.__loader__, name="Selector")
        super().__init__(**params)
        self.__layout__ = pn.Column(pn.Param(self.param, name=""), self.element.view())

    @param.depends("widget", "element.param", watch=True)
    def _update(self):
        if self.widget != self.__last__:
            if self.widget == "Selector":
                self.element = DashboardElement(self.__loader__, name="Selector")
            else:
                self.element = DashboardElement.__sc__[self.widget](
                    self.__loader__, name=self.widget
                )
        self.__last__ = self.widget
        self.__layout__[1] = self.element.view()

    @property
    def layout(self):
        return self.__layout__


class FluidMultiVariablePlot(DashboardElement):

    alias = "Fluid Multi Variable Plot"
    variables = param.ListSelector(default=[])
    run = param.Selector()
    dim = param.Selector(objects=["Fixed time", "Fixed position"])
    x_lower_limit = param.Number()
    x_upper_limit = param.Number()
    y_lower_limit = param.Number()
    y_upper_limit = param.Number()

    def __init__(self, loader: LazyLoader, **params):
        super().__init__(loader, **params)
        self.param["variables"].objects = [
            name
            for name in loader.__variables__.varNames
            if loader.__variables__[name].isFluid
        ]
        self.param["run"].objects = list(loader.__runPaths__.keys())

    def view(self):
        if self.run is None:
            self.run = self.param["run"].objects[0]
        if len(self.variables) == 0 and len(self.param["variables"].objects) > 0:
            self.variables = [self.param["variables"].objects[0]]

        if len(self.variables) > 0:
            for var in self.variables:
                self.__loader__.load(var, self.run)
            if self.dim == "Fixed position":
                val = pn.widgets.IntSlider(
                    name="x",
                    value=0,
                    start=0,
                    end=self.__loader__[(self.variables[0], self.run)].dataShape[1] - 1,
                )
            else:
                val = pn.widgets.IntSlider(
                    name="t",
                    value=0,
                    start=0,
                    end=self.__loader__[(self.variables[0], self.run)].dataShape[0] - 1,
                )
        else:
            return pn.Column(
                pn.Param(self.param, widgets={"dim": pn.widgets.RadioButtonGroup})
            )

        def loadFluidT(ind):
            curves = {
                var: fluidVarT(
                    self.__loader__[(var, self.run)],
                    ind,
                    self.__loader__[("time", self.run)],
                )
                for var in self.variables
            }
            return hv.NdOverlay(curves).opts(
                xlim=(self.x_lower_limit, self.x_upper_limit),
                ylim=(self.y_lower_limit, self.y_upper_limit),
                title="",
            )

        def loadFluidX(ind):
            curves = {
                var: fluidVarX(self.__loader__[(var, self.run)], ind)
                for var in self.variables
            }
            return hv.NdOverlay(curves).opts(
                xlim=(self.x_lower_limit, self.x_upper_limit),
                ylim=(self.y_lower_limit, self.y_upper_limit),
                title="",
            )

        if self.dim == "Fixed position":
            dmap = hv.DynamicMap(pn.bind(loadFluidT, ind=val))

        else:
            dmap = hv.DynamicMap(pn.bind(loadFluidX, ind=val))

        return pn.Row(
            pn.Column(
                pn.Param(self.param, widgets={"dim": pn.widgets.RadioButtonGroup}), val
            ),
            pn.panel(dmap),
        )


class ScalarMultiVariablePlot(DashboardElement):

    alias = "Scalar Multi Variable Plot"
    variables = param.ListSelector(default=[])
    run = param.Selector()
    x_lower_limit = param.Number()
    x_upper_limit = param.Number()
    y_lower_limit = param.Number()
    y_upper_limit = param.Number()

    def __init__(self, loader: LazyLoader, **params):
        super().__init__(loader, **params)
        self.param["variables"].objects = [
            name
            for name in loader.__variables__.varNames
            if loader.__variables__[name].isScalar
        ]
        self.param["run"].objects = list(loader.__runPaths__.keys())

    def view(self):
        if self.run is None:
            self.run = self.param["run"].objects[0]
        if len(self.variables) == 0 and len(self.param["variables"].objects) > 0:
            self.variables = [self.param["variables"].objects[0]]

        if len(self.variables) > 0:
            for var in self.variables:
                self.__loader__.load(var, self.run)

        else:
            return pn.Column(pn.Param(self.param))

        curves = {
            var: scalarVarT(
                self.__loader__[(var, self.run)],
                self.__loader__[("time", self.run)],
            )
            for var in self.variables
        }

        return pn.Row(
            pn.Column(pn.Param(self.param)),
            pn.panel(
                hv.NdOverlay(curves).opts(
                    xlim=(self.x_lower_limit, self.x_upper_limit),
                    ylim=(self.y_lower_limit, self.y_upper_limit),
                    title="",
                )
            ),
        )


class FluidMultiRunPlot(DashboardElement):

    alias = "Fluid Multi Run Plot"
    variable = param.Selector()
    runs = param.ListSelector([])
    dim = param.Selector(objects=["Fixed time", "Fixed position"])
    x_lower_limit = param.Number()
    x_upper_limit = param.Number()
    y_lower_limit = param.Number()
    y_upper_limit = param.Number()

    def __init__(self, loader: LazyLoader, **params):
        super().__init__(loader, **params)
        self.param["variable"].objects = [
            name
            for name in loader.__variables__.varNames
            if loader.__variables__[name].isFluid
        ]
        self.param["runs"].objects = list(loader.__runPaths__.keys())

    def view(self):
        if self.variable is None:
            self.variable = self.param["variable"].objects[0]
        if len(self.runs) == 0 and len(self.param["runs"].objects) > 0:
            self.runs = [self.param["runs"].objects[0]]

        if len(self.runs) > 0:
            for run in self.runs:
                self.__loader__.load(self.variable, run)
            if self.dim == "Fixed position":
                val = pn.widgets.IntSlider(
                    name="x",
                    value=0,
                    start=0,
                    end=self.__loader__[(self.variable, self.runs[0])].dataShape[1] - 1,
                )
            else:
                val = pn.widgets.IntSlider(
                    name="t",
                    value=0,
                    start=0,
                    end=self.__loader__[(self.variable, self.runs[0])].dataShape[0] - 1,
                )

        else:
            return pn.Column(
                pn.Param(self.param, widgets={"dim": pn.widgets.RadioButtonGroup})
            )

        def loadFluidT(ind):
            curves = {
                run: fluidVarT(
                    self.__loader__[(self.variable, run)],
                    ind,
                    self.__loader__[("time", run)],
                )
                for run in self.runs
            }
            return hv.NdOverlay(curves).opts(
                xlim=(self.x_lower_limit, self.x_upper_limit),
                ylim=(self.y_lower_limit, self.y_upper_limit),
                title="",
            )

        def loadFluidX(ind):
            curves = {
                run: fluidVarX(self.__loader__[(self.variable, run)], ind)
                for run in self.runs
            }

            return hv.NdOverlay(curves).opts(
                xlim=(self.x_lower_limit, self.x_upper_limit),
                ylim=(self.y_lower_limit, self.y_upper_limit),
                title="",
            )

        if self.dim == "Fixed position":
            dmap = hv.DynamicMap(pn.bind(loadFluidT, ind=val))

        else:
            dmap = hv.DynamicMap(pn.bind(loadFluidX, ind=val))

        return pn.Row(
            pn.Column(
                pn.Param(self.param, widgets={"dim": pn.widgets.RadioButtonGroup}), val
            ),
            pn.panel(dmap),
        )


class ScalarMultiRunPlot(DashboardElement):

    alias = "Scalar Multi Run Plot"
    variable = param.Selector()
    runs = param.ListSelector([])
    x_lower_limit = param.Number()
    x_upper_limit = param.Number()
    y_lower_limit = param.Number()
    y_upper_limit = param.Number()

    def __init__(self, loader: LazyLoader, **params):
        super().__init__(loader, **params)
        self.param["variable"].objects = [
            name
            for name in loader.__variables__.varNames
            if loader.__variables__[name].isScalar
        ]
        self.param["runs"].objects = list(loader.__runPaths__.keys())

    def view(self):
        if self.variable is None:
            self.variable = self.param["variable"].objects[0]
        if len(self.runs) == 0 and len(self.param["runs"].objects):
            self.runs = [self.param["runs"].objects[0]]

        if len(self.runs) > 0:
            for run in self.runs:
                self.__loader__.load(self.variable, run)

        else:
            return pn.Column(pn.Param(self.param))

        curves = {
            run: scalarVarT(
                self.__loader__[(self.variable, run)],
                self.__loader__[("time", run)],
            )
            for run in self.runs
        }

        return pn.Row(
            pn.Column(pn.Param(self.param)),
            pn.panel(
                hv.NdOverlay(curves).opts(
                    xlim=(self.x_lower_limit, self.x_upper_limit),
                    ylim=(self.y_lower_limit, self.y_upper_limit),
                    title="",
                )
            ),
        )


class DistExplorer(DashboardElement):
    alias = "Distribution Explorer"
    logarithmic_y = param.Boolean(default=False)
    energy_grid = param.Boolean(default=False)
    variable = param.Selector()
    runs = param.ListSelector([])
    auto_x_limits = param.Boolean(default=True)
    x_lower_limit = param.Number()
    x_upper_limit = param.Number()
    auto_y_limits = param.Boolean(default=True)
    y_lower_limit = param.Number()
    y_upper_limit = param.Number()

    def __init__(self, loader: LazyLoader, **params):
        super().__init__(loader, **params)
        self.param["variable"].objects = [
            name
            for name in loader.__variables__.varNames
            if loader.__variables__[name].isDistribution
        ]
        self.param["runs"].objects = list(loader.__runPaths__.keys())

    def view(self):
        if not len(self.param["variable"].objects) > 0:
            return pn.Column(pn.Param(self.param))
        if self.variable is None:
            self.variable = self.param["variable"].objects[0]
        if len(self.runs) == 0 and len(self.param["runs"].objects) > 0:
            self.runs = [self.param["runs"].objects[0]]

        if len(self.runs) > 0:
            for run in self.runs:
                self.__loader__.load(self.variable, run)
            pos = pn.widgets.IntSlider(
                name="x",
                value=0,
                start=0,
                end=self.__loader__[(self.variable, self.runs[0])].dataShape[1] - 1,
            )
            t = pn.widgets.IntSlider(
                name="t",
                value=0,
                start=0,
                end=self.__loader__[(self.variable, self.runs[0])].dataShape[0] - 1,
            )
            h = pn.widgets.IntSlider(
                name="h",
                value=0,
                start=0,
                end=self.__loader__[(self.variable, self.runs[0])].dataShape[2] - 1,
            )

        else:
            return pn.Column(pn.Param(self.param))

        def load_dist_curves(x: int, time: int, harmonic: int):

            xlims = (
                (None, None)
                if self.auto_x_limits
                else (self.x_lower_limit, self.x_upper_limit)
            )
            ylims = (
                (None, None)
                if self.auto_y_limits
                else (self.y_lower_limit, self.y_upper_limit)
            )

            curves = {
                run: distVarV(
                    self.__loader__[(self.variable, run)],
                    x,
                    time,
                    harmonic,
                    self.energy_grid,
                ).opts(framewise=True, logy=self.logarithmic_y, xlim=xlims, ylim=ylims)
                for run in self.runs
            }

            return hv.NdOverlay(curves).opts(
                title="",
            )

        dmap = hv.DynamicMap(pn.bind(load_dist_curves, pos, t, h))

        return pn.Row(pn.Column(self.param, pos, t, h), pn.panel(dmap))


class ReMKiT1DDashboard:
    def __init__(self, data: xr.Dataset, gridObj: Grid):
        """Constructor for a dashboard object

        NOTE: This is the pre-v2.0.0 dashboard. It is depracated and might be removed in a future version.

        Args:
            data (xr.Dataset): Data loaded into the dashboard to be visualized
            gridObj (Grid): Grid object used to set axis ticks and labels.
        """
        warn(
            "This is the pre-v2.0.0 dashboard. It is depracated and might be removed in a future version.",
            category=DeprecationWarning,
        )
        self.__data__ = data
        self.__fluidNames__ = list(
            data.filter_by_attrs(isDistribution=False, isScalar=False).data_vars.keys()
        )
        self.__distNames__ = list(
            data.filter_by_attrs(isDistribution=True, isScalar=False).data_vars.keys()
        )
        self.__gridObj__ = gridObj
        self.__dualGrid__ = gridObj.xGrid + gridObj.xWidths / 2

    def __load_fluid__(
        self, dataname, val, mode, logy=False, removeTitle=False, **kwargs
    ):
        assert self.__data__[dataname].coords.dims == (
            "t",
            "x",
        ) or self.__data__[dataname].coords.dims == (
            "t",
            "x_dual",
        ), "Non-fluid dataname in load_fluid"
        titlePrefix = "Variable: " + dataname + ", "
        if removeTitle:
            titlePrefix = ""
        if mode == "Fixed position":
            if dataname[-5:] == "_dual":
                curve = hv.Curve(self.__data__[dataname][:, val], label=dataname).opts(
                    framewise=True,
                    title=titlePrefix
                    + f"x = {self.__dualGrid__[val]:.2f} "
                    + self.__data__.coords["x"].units,
                    logy=logy,
                )
            else:
                curve = hv.Curve(self.__data__[dataname][:, val], label=dataname).opts(
                    framewise=True,
                    title=titlePrefix
                    + f'x = {self.__data__.coords["x"].values[val]:.2f} '
                    + self.__data__.coords["x"].units,
                    logy=logy,
                )
        if mode == "Fixed time":
            if dataname[-5:] == "_dual":
                curve = hv.Curve(
                    (
                        self.__dualGrid__,
                        self.__data__.data_vars[dataname].values[val, :],
                    ),
                    label=dataname,
                ).opts(
                    framewise=True,
                    title=titlePrefix
                    + f't = {self.__data__.coords["t"].values[val]:.2f} '
                    + self.__data__.coords["t"].units,
                    logy=logy,
                )
            else:
                curve = hv.Curve(self.__data__[dataname][val, :], label=dataname).opts(
                    framewise=True,
                    title=titlePrefix
                    + f't = {self.__data__.coords["t"].values[val]:.2f} '
                    + self.__data__.coords["t"].units,
                    logy=logy,
                )

        return curve

    def fluid2Comparison(self, logY=False):
        """Open up a dashboard view with side-by-side plots of two fluid quantites. These can be plotted at fixed positions as functions of time, or vice versa.

        Args:
            logY (bool, optional): Set the y axes on both plots to a logarithimic axis. Defaults to False.
        """
        opt = pn.widgets.RadioButtonGroup(options=["Fixed position", "Fixed time"])
        variable = pn.widgets.Select(options=self.__fluidNames__)
        val = pn.widgets.IntSlider(
            name="x", value=0, start=0, end=len(self.__data__.coords["x"].values) - 1
        )

        opt2 = pn.widgets.RadioButtonGroup(options=["Fixed position", "Fixed time"])
        variable2 = pn.widgets.Select(options=self.__fluidNames__)
        val2 = pn.widgets.IntSlider(
            name="x", value=0, start=0, end=len(self.__data__.coords["x"].values) - 1
        )

        def callback(target, event):
            if event.new == "Fixed position":
                target.name = "x"
                target.value = 0
                target.start = 0
                target.end = len(self.__data__.coords["x"].values) - 1
            if event.new == "Fixed time":
                target.name = "t"
                target.value = 0
                target.start = 0
                target.end = len(self.__data__.coords["t"].values) - 1

        opt.link(val, callbacks={"value": callback})
        opt2.link(val2, callbacks={"value": callback})

        dmap = hv.DynamicMap(
            pn.bind(self.__load_fluid__, dataname=variable, val=val, mode=opt)
        )
        dmap2 = hv.DynamicMap(
            pn.bind(self.__load_fluid__, dataname=variable2, val=val2, mode=opt2)
        )

        app = pn.Row(
            pn.Column(
                pn.WidgetBox(
                    '## <span style="color:black"> Left </span>', opt, variable, val
                ),
                pn.Column(
                    pn.WidgetBox(
                        '## <span style="color:black"> Right </span>',
                        opt2,
                        variable2,
                        val2,
                    )
                ),
            ),
            dmap.opts(framewise=True, logy=logY),
            dmap2.opts(framewise=True, logy=logY),
        )
        return app

    def distDynMap(self):
        """Open up a distribution function explorer dashboard. Allows for exploring different distribution function variables in velocity space, with the remaining coordinate values selectable."""

        class DistExplorer(param.Parameterized):
            variable = param.ObjectSelector(
                default=self.__distNames__[0], objects=self.__distNames__
            )
            t = param.Integer(
                default=0, bounds=(0, len(self.__data__.coords["t"].values) - 1)
            )
            pos = param.Integer(
                default=0, bounds=(0, len(self.__data__.coords["x"].values) - 1)
            )
            harmonic = param.Integer(
                default=0, bounds=(0, len(self.__data__.coords["h"].values) - 1)
            )
            maxV = param.Integer(
                default=len(self.__data__.coords["v"].values) - 1,
                bounds=(0, len(self.__data__.coords["v"].values) - 1),
            )
            logY = param.Boolean(default=False)
            energyGrid = param.Boolean(default=False)

            def __init__(self, data, **params):
                self.__data__ = data
                super().__init__(**params)

            @param.depends("variable", "t", "pos", "harmonic", "maxV", "energyGrid")
            def load_dist_curve(self):
                assert self.__data__[self.variable].coords.dims == (
                    "t",
                    "x",
                    "h",
                    "v",
                ) or self.__data__[self.variable].coords.dims == (
                    "t",
                    "x_dual",
                    "h",
                    "v",
                ), "Non-dist dataname in load_dist"

                if self.energyGrid:
                    curve = hv.Curve(
                        (
                            self.__data__.coords["v"][: self.maxV + 1] ** 2,
                            self.__data__[self.variable][
                                self.t, self.pos, self.harmonic, : self.maxV + 1
                            ],
                        ),
                        label=self.variable,
                    ).opts(framewise=True, logy=self.logY)
                else:
                    curve = hv.Curve(
                        self.__data__[self.variable][
                            self.t, self.pos, self.harmonic, : self.maxV + 1
                        ],
                        label=self.variable,
                    ).opts(framewise=True, logy=self.logY)

                return curve

            @param.depends("logY")
            def view(self):
                dmap = hv.DynamicMap(self.load_dist_curve)

                return dmap.opts(logy=self.logY)

        explorer = DistExplorer(
            self.__data__.filter_by_attrs(isDistribution=True, isScalar=False)
        )

        app = pn.Row(explorer.param, explorer.view)
        return app

    def fluidMultiComparison(
        self, dataNames: List[str], fixedPosition=False
    ) -> hv.HoloMap:
        """Generate a holoviews map comparing multiple different fluid variables on the same plot. By default compares data profiles at fixed points in time.

        Args:
            dataNames (List[str]): List of fluid variables names to be compared.
            fixedPosition (bool, optional): If True will plot variables as functions of time with the HoloMap slider controlling the x position. Defaults to False.

        Returns:
            hv.HoloMap: Interactive HoloMap object containing the comparison plots
        """
        if fixedPosition:
            curveDict = {
                x: hv.Overlay(
                    [
                        self.__load_fluid__(
                            name, x, mode="Fixed position", removeTitle=True
                        )
                        for name in dataNames
                    ]
                )
                for x in range(len(self.__data__.coords["x"].values))
            }
            kdims = [
                hv.Dimension(
                    ("x", "Position"),
                    unit=self.__data__.coords["x"].attrs["units"],
                    default=0,
                )
            ]
        else:
            curveDict = {
                t: hv.Overlay(
                    [
                        self.__load_fluid__(
                            name, t, mode="Fixed time", removeTitle=True
                        )
                        for name in dataNames
                    ]
                )
                for t in range(len(self.__data__.coords["t"].values))
            }
            kdims = [
                hv.Dimension(
                    ("t", "Time"),
                    unit=self.__data__.coords["t"].attrs["units"],
                    default=0,
                )
            ]

        hMap = hv.HoloMap(curveDict, kdims=kdims)

        return hMap
