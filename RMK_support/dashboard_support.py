import xarray as xr
import holoviews as hv  # type: ignore
import panel as pn
import numpy as np
from .grid import Grid
from typing import Union, List
import param  # type: ignore


class ReMKiT1DDashboard:
    def __init__(self, data: xr.Dataset, gridObj: Grid):
        """Constructor for a dashboard object

        Args:
            data (xr.Dataset): Data loaded into the dashboard to be visualized
            gridObj (Grid): Grid object used to set axis ticks and labels.
        """
        self.__data__ = data
        self.__fluidNames__ = list(
            data.filter_by_attrs(isDistribution=False, isScalar=False).data_vars.keys()
        )
        self.__distNames__ = list(
            data.filter_by_attrs(isDistribution=True, isScalar=False).data_vars.keys()
        )
        self.__gridObj__ = gridObj
        self.__dualGrid__ = gridObj.xGrid + gridObj.xWidths / 2

    def __load_dist_curve__(
        self, dataname, time, pos, harmonic, logY, energyGrid, maxV, **kwargs
    ):
        assert self.__data__[dataname].coords.dims == (
            "time",
            "x",
            "h",
            "v",
        ), "Non-dist dataname in load_dist"

        if energyGrid:
            curve = hv.Curve(
                (
                    self.__energyGrid__[: maxV + 1],
                    self.__data__[dataname][time, pos, harmonic, : maxV + 1],
                ),
                label=dataname,
            ).opts(framewise=True, logy=logY)
        else:
            curve = hv.Curve(
                self.__data__[dataname][time, pos, harmonic, : maxV + 1], label=dataname
            ).opts(framewise=True, logy=logY)

        return curve

    def __load_fluid__(
        self, dataname, val, mode, logy=False, removeTitle=False, **kwargs
    ):
        assert self.__data__[dataname].coords.dims == (
            "time",
            "x",
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
                    + f't = {self.__data__.coords["time"].values[val]:.2f} '
                    + self.__data__.coords["time"].units,
                    logy=logy,
                )
            else:
                curve = hv.Curve(self.__data__[dataname][val, :], label=dataname).opts(
                    framewise=True,
                    title=titlePrefix
                    + f't = {self.__data__.coords["time"].values[val]:.2f} '
                    + self.__data__.coords["time"].units,
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
                target.name = "time"
                target.value = 0
                target.start = 0
                target.end = len(self.__data__.coords["time"].values) - 1

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
            time = param.Integer(
                default=0, bounds=(0, len(self.__data__.coords["time"].values) - 1)
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

            @param.depends("variable", "time", "pos", "harmonic", "maxV", "energyGrid")
            def load_dist_curve(self):
                assert self.__data__[self.variable].coords.dims == (
                    "time",
                    "x",
                    "h",
                    "v",
                ), "Non-dist dataname in load_dist"

                if self.energyGrid:
                    curve = hv.Curve(
                        (
                            self.__data__.coords["v"][: self.maxV + 1] ** 2,
                            self.__data__[self.variable][
                                self.time, self.pos, self.harmonic, : self.maxV + 1
                            ],
                        ),
                        label=self.variable,
                    ).opts(framewise=True, logy=self.logY)
                else:
                    curve = hv.Curve(
                        self.__data__[self.variable][
                            self.time, self.pos, self.harmonic, : self.maxV + 1
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
                for t in range(len(self.__data__.coords["time"].values))
            }
            kdims = [
                hv.Dimension(
                    ("time", "Time"),
                    unit=self.__data__.coords["time"].attrs["units"],
                    default=0,
                )
            ]

        hMap = hv.HoloMap(curveDict, kdims=kdims)

        return hMap
