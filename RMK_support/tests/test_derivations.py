import RMK_support.derivations as dv
from RMK_support import Variable, VariableContainer, Grid, node
import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type:ignore
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
        isLengthInMeters=True,
    )


def test_species(grid):

    a = Variable("a", grid, subtype="a")

    sp1 = dv.Species("sp1", 1)
    sp2 = dv.Species("sp2", 2, 2.0, 3.0)
    sp3 = dv.Species("sp3", 3, 2.0, 3.0)
    sp1.associateVar(a, associateSubtype=True)

    species = dv.SpeciesContainer(sp1)
    species.add(sp2)

    assert (species["sp1"].name, species["sp1"].speciesID, sp1.charge, sp1.atomicA) == (
        "sp1",
        1,
        0.0,
        1.0,
    )
    assert species["sp1"]["a"].name == "a"
    assert (sp2.name, sp2.speciesID, sp2.charge, sp2.atomicA) == ("sp2", 2, 3.0, 2.0)
    species["sp4"] = sp3

    assert species["sp4"].name == "sp4"

    assert species.speciesNames == ["sp1", "sp2", "sp4"]

    species["sp2"] = dv.Species("sp2", 6, 2.0, 3.0)
    assert species["sp2"].speciesID == 6
    del species["sp4"]
    assert species.speciesNames == ["sp1", "sp2"]

    speciesDict = {"names": ["sp1", "sp2"]}
    speciesDict["sp1"] = {
        "ID": 1,
        "atomicMass": 1.0,
        "charge": 0.0,
        "associatedVars": ["a"],
    }
    speciesDict["sp2"] = {
        "ID": 6,
        "atomicMass": 2.0,
        "charge": 3.0,
        "associatedVars": [],
    }

    assert speciesDict == species.dict()


def test_generic_derivation(grid):

    deriv = dv.GenericDerivation(
        "d", 1, {"test": True}, resultProperties={"isScalar": True}
    ).rename("c")

    assert deriv.name == "c"
    deriv.name = "d"

    a = Variable("a", grid)

    assert deriv.dict() == {"test": True}

    with pytest.raises(NotImplementedError):
        _ = deriv(np.array([1.0]))

    d = deriv(a)
    assert d.name == "d"
    assert d.isDerived
    assert d.derivationArgs == ["a"]
    assert d.isScalar


def test_simple_derivation():

    deriv = dv.SimpleDerivation("a", 2.0, [2.0, 3.0])

    x = np.array([1.0, 2.0])
    y = np.array([1.5, -1.0])

    result = deriv(x, y)
    assert all(np.isclose(result, 2 * x**2 * y**3))
    assert deriv.dict() == {
        "type": "simpleDerivation",
        "multConst": 2.0,
        "varPowers": [2.0, 3.0],
    }


def test_interpolation_deriv():

    grid = Grid(
        xGrid=np.linspace(1, 2, 5),
        vGrid=np.linspace(1, 2, 5),
        lMax=1,
        interpretXGridAsWidths=True,
    )

    deriv1 = dv.InterpolationDerivation(grid)

    assert deriv1.name == "gridToDual"
    data = np.linspace(0, 4, 5)
    interpData = deriv1(data)
    assert all(
        np.isclose(interpData[:-1], np.interp(grid.xGridDual, grid.xGrid, data)[:-1])
    )

    assert np.isclose(
        interpData[-1],
        2
        * (data[-1] - data[-2])
        / (grid.xWidths[-1] + grid.xWidths[-2])
        * grid.xWidths[-1]
        / 2
        + data[-1],
    )

    deriv2 = dv.InterpolationDerivation(grid, False)
    assert deriv2.name == "dualToGrid"

    interpData = deriv2(data)
    assert all(
        np.isclose(interpData[1:-1], np.interp(grid.xGrid, grid.xGridDual, data)[1:-1])
    )

    assert np.isclose(
        interpData[-1],
        (data[-2] - data[-3]) / grid.xWidths[-2] * grid.xWidths[-1] / 2 + data[-2],
    )

    assert np.isclose(
        interpData[0],
        -(data[1] - data[0]) / grid.xWidths[1] * grid.xWidths[0] / 2 + data[0],
    )

    deriv3 = dv.InterpolationDerivation(grid, onDistribution=True)
    data = np.ones((5, 2, 5))
    for i in range(5):
        data[:, 0, i] = np.linspace(0, 4, 5)
        data[:, 1, i] = np.linspace(0, 4, 5)

    interpData = deriv3(data)

    for i in range(5):

        assert all(
            np.isclose(
                interpData[:-1, 0, i],
                np.interp(grid.xGridDual, grid.xGrid, data[:, 0, i])[:-1],
            )
        )

        assert np.isclose(
            interpData[-1, 0, i],
            2
            * (data[-1, 0, i] - data[-2, 0, i])
            / (grid.xWidths[-1] + grid.xWidths[-2])
            * grid.xWidths[-1]
            / 2
            + data[-1, 0, i],
        )

        assert all(
            np.isclose(
                interpData[1:-1, 1, i],
                np.interp(grid.xGrid, grid.xGridDual, data[:, 1, i])[1:-1],
            )
        )

        assert np.isclose(
            interpData[-1, 1, i],
            (data[-2, 1, i] - data[-3, 1, i]) / grid.xWidths[-2] * grid.xWidths[-1] / 2
            + data[-2, 1, i],
        )

        assert np.isclose(
            interpData[0, 1, i],
            -(data[1, 1, i] - data[0, 1, i]) / grid.xWidths[1] * grid.xWidths[0] / 2
            + data[0, 1, i],
        )


def test_textbook():

    grid = Grid(
        xGrid=np.linspace(1, 2, 5),
        vGrid=np.linspace(1, 2, 5),
        lMax=2,
        interpretXGridAsWidths=True,
    )

    tb = dv.Textbook(grid)

    assert tb.registeredDerivs == [
        "flowSpeedFromFlux",
        "leftElectronGamma",
        "rightElectronGamma",
        "densityMoment",
        "energyMoment",
        "cclDragCoeff",
        "cclDiffusionCoeff",
        "cclWeight",
        "fluxMoment",
        "heatFluxMoment",
        "viscosityTensorxxMoment",
        "gridToDual",
        "dualToGrid",
        "distributionInterp",
        "gradDeriv",
        "logLee",
        "maxwellianDistribution",
    ]

    deriv = dv.GenericDerivation("d", 1, properties={})
    tb.register(deriv)
    assert tb.registeredDerivs[-1] == "d"

    assert tb["d"].numArgs == 1

    assert tb["logLeiD+"].numArgs == 2

    assert tb.tempDerivSpeciesIDs == []

    tb.addSpeciesForTempDeriv(dv.Species("e", 0))

    assert tb.tempDerivSpeciesIDs == [0]

    tb.ePolyCoeff = 2.0
    tb.ionPolyCoeff = 2.0
    assert tb.ePolyCoeff == 2.0
    assert tb.ionPolyCoeff == 2.0

    tb.removeLogLeiDiscontinuity = True
    assert tb.removeLogLeiDiscontinuity

    tb.setSheathGammaSpecies(dv.Species("D", -2))
    textbookDict = {
        "standardTextbook": {
            "temperatureDerivSpeciesIDs": [0],
            "electronPolytropicCoeff": 2.0,
            "ionPolytropicCoeff": 2.0,
            "electronSheathGammaIonSpeciesID": -2,
            "removeLogLeiDiscontinuity": True,
        },
        "customDerivations": {"tags": ["d"], "d": {}},
    }
    assert textbookDict == tb.dict()


def test_closure_arithmetic(grid):

    d1 = dv.DerivationClosure(
        dv.SimpleDerivation("d1", 2.0, [1.0, 1.0]),
        Variable("a", grid, isDerived=True, isScalar=True, data=2 * np.ones(1)),
        Variable("b", grid, isDerived=True, isScalar=True, data=3 * np.ones(1)),
    )
    d2 = dv.DerivationClosure(
        dv.SimpleDerivation("d2", 1.0, [2.0]),
        Variable("a", grid, isDerived=True, isScalar=True, data=2 * np.ones(1)),
    )

    add = (2 * d1).rename("d") + (d2 + d1) + d2
    assert all(add.evaluate() == 44)
    assert add.numArgs == 0
    assert add.enclosedArgs == 6
    assert add.fillArgs() == ["a", "b", "a", "a", "b", "a"]
    assert add.name == "d_d2_d1_d2"
    assert add.__deriv__.dict() == {
        "type": "additiveDerivation",
        "derivationTags": ["d1", "d2", "d1_copy", "d2_copy"],
        "resultPower": 1.0,
        "linearCoefficients": [2.0, 1.0, 1.0, 1.0],
        "d1": {"derivationIndices": [1, 2]},
        "d1_copy": {"derivationIndices": [4, 5]},
        "d2": {"derivationIndices": [3]},
        "d2_copy": {"derivationIndices": [6]},
    }

    p = d1**2

    assert p.enclosedArgs == 2
    assert p.name == "d1_pow"

    mul = (d1 * d2).rename("d") * d1
    assert all(mul.evaluate() == 12 * 4 * 12)
    assert mul.fillArgs() == ["a", "b", "a", "a", "b"]
    assert mul.name == "dXd1"
    mul.enclosedArgs == 5
    assert mul.__deriv__.dict() == {
        "type": "multiplicativeDerivation",
        "innerDerivation": "d",
        "innerDerivIndices": [1, 2, 3],
        "innerDerivPower": 1.0,
        "outerDerivation": ("d1"),
        "outerDerivIndices": [4, 5],
        "outerDerivPower": 1.0,
        "innerDerivFuncName": "none",
    }

    mul = dv.funApply("exp", d1)
    assert mul.name == "exp_d1"
    assert mul.enclosedArgs == 2
    assert mul.__deriv__.dict() == {
        "type": "multiplicativeDerivation",
        "innerDerivation": "d1",
        "innerDerivIndices": [1, 2],
        "innerDerivPower": 1.0,
        "outerDerivation": ("none"),
        "outerDerivIndices": [],
        "outerDerivPower": 1.0,
        "innerDerivFuncName": "exp",
    }
    assert all(mul.evaluate() == np.exp(12))


def test_polynomial_deriv():

    deriv = dv.PolynomialDerivation(
        "p", 1.0, np.array([1.0, 1.5]), np.array([2.0, 0.5])
    )

    assert deriv.dict() == {
        "type": "polynomialFunctionDerivation",
        "constantPolynomialCoefficient": 1.0,
        "polynomialPowers": [2.0, 0.5],
        "polynomialCoefficients": [1.0, 1.5],
    }

    assert all(deriv.evaluate(np.array([2]), np.array([3])) == 5 + 1.5 * np.sqrt(3))


def test_range_filter_deriv(grid):

    deriv = dv.SimpleDerivation("d", 1.0, [1.0])

    vals = np.linspace(0, 10, grid.numX)
    filtered = dv.RangeFilterDerivation(
        "f", deriv, [(Variable("p", grid, data=vals), 3, 6)]
    )

    assert filtered.dict() == {
        "type": "rangeFilterDerivation",
        "ruleName": "d",
        "controlIndices": [1],
        "controlRanges": {"index1": [3, 6]},
        "derivationIndices": [2],
    }

    assert all(
        filtered(vals, np.ones(grid.numX)) == np.where((vals < 6) & (vals > 3), 1, 0)
    )

    assert filtered.fillArgs("a") == ["p", "a"]

    assert filtered.enclosedArgs == 1


def test_filtered_node_deriv(grid):

    deriv = dv.NodeDerivation("deriv", node=node(Variable("a", grid)))

    vals = np.linspace(0, 10, grid.numX)
    filtered = dv.RangeFilterDerivation(
        "f", deriv, [(Variable("p", grid, data=vals), 3, 6)]
    )

    assert filtered.enclosedArgs == 2

    assert filtered.numArgs == 0

    assert filtered.fillArgs("a") == ["p", "a"]

    assert all(
        filtered(vals, np.ones(grid.numX)) == np.where((vals < 6) & (vals > 3), 1, 0)
    )


def test_bounded_ext_deriv(grid):

    deriv = dv.BoundedExtrapolationDerivation(
        "d1", "lin", 0.1, 1, leftBoundary=True, expectedHaloWidth=1, staggeredVars=False
    )

    assert deriv.dict() == {
        "type": "boundedExtrapolationDerivation",
        "expectUpperBoundVar": False,
        "expectLowerBoundVar": False,
        "ignoreUpperBound": False,
        "ignoreLowerBound": False,
        "fixedLowerBound": 0.1,
        "fixedUpperBound": 1,
        "extrapolationStrategy": {
            "type": "linExtrapolation",
            "leftBoundary": True,
            "staggeredVars": False,
            "expectedHaloWidth": 1,
        },
    }

    deriv2 = dv.BoundedExtrapolationDerivation(
        "d2", "lin", lowerBound=Variable("c", grid), upperBound=Variable("b", grid)
    )
    assert deriv2.fillArgs("a") == ["a", "c", "b"]
    assert deriv2.enclosedArgs == 2
    assert deriv2.resultProperties["isScalar"]


def test_shkarofsky_derivs():

    CII = dv.coldIonIDeriv("CII", 0)
    assert CII.dict() == {
        "type": "coldIonIJIntegralDerivation",
        "isJIntegral": False,
        "index": 0,
    }

    CIJ = dv.coldIonJDeriv("CIJ", 0)
    assert CIJ.dict() == {
        "type": "coldIonIJIntegralDerivation",
        "isJIntegral": True,
        "index": 0,
    }
    I = dv.shkarofskyIIntegralDeriv("I", 0)
    assert I.dict() == {
        "type": "IJIntegralDerivation",
        "isJIntegral": False,
        "index": 0,
    }
    J = dv.shkarofskyJIntegralDeriv("J", 0)
    assert J.dict() == {"type": "IJIntegralDerivation", "isJIntegral": True, "index": 0}


def test_harmonic_extractor(grid):

    ext = dv.HarmonicExtractorDerivation("h", grid, 1)

    f = Variable("f", grid, isDistribution=True)

    assert ext.dict() == {"type": "harmonicExtractorDerivation", "index": 1}

    assert ext(f.data).shape == (grid.numX, grid.numV)


def test_ddv_deriv(grid):

    ddv = dv.DDVDerivation(
        "ddv",
        grid,
        1,
        grid.profile(np.ones(grid.numV), "V"),
        grid.profile(np.ones(grid.numV), "V"),
        (1, 1),
    )

    assert ddv.resultProperties["isSingleHarmonic"]
    assert ddv.dict() == {
        "type": "ddvDerivation",
        "targetH": 1,
        "outerV": np.ones(grid.numV).tolist(),
        "innerV": np.ones(grid.numV).tolist(),
        "vifAtZero": [1, 1],
    }


def test_d2dv2_deriv(grid):

    d2dv2 = dv.D2DV2Derivation(
        "d2dv2",
        grid,
        1,
        grid.profile(np.ones(grid.numV), "V"),
        grid.profile(np.ones(grid.numV), "V"),
        (1, 1),
    )

    assert d2dv2.resultProperties["isSingleHarmonic"]
    assert d2dv2.dict() == {
        "type": "d2dv2Derivation",
        "targetH": 1,
        "outerV": np.ones(grid.numV).tolist(),
        "innerV": np.ones(grid.numV).tolist(),
        "vidfdvAtZero": [1, 1],
    }


def test_moment_deriv(grid):

    deriv = dv.MomentDerivation(
        "m", grid, 1, 1, gVec=grid.profile(np.ones(grid.numV), "V")
    )

    assert deriv.dict() == {
        "type": "momentDerivation",
        "momentHarmonic": 1,
        "momentOrder": 1,
        "multConst": 1.0,
        "varPowers": [],
        "gVector": np.ones(grid.numV).tolist(),
    }

    assert deriv.resultProperties == {
        "isScalar": False,
        "isDistribution": False,
        "isSingleHarmonic": False,
    }


def test_gen_int_poly_deriv():

    deriv = dv.GenIntPolynomialDerivation(
        "p",
        np.array([[1, 1], [1, 2], [2, 2]]),
        np.array([0.5, 1.5, 0.1]),
        multConst=2.0,
        funcName="exp",
    )

    assert deriv.dict() == {
        "type": "generalizedIntPowerPolyDerivation",
        "multConst": 2.0,
        "polynomialPowers": {"index1": [1, 1], "index2": [1, 2], "index3": [2, 2]},
        "polynomialCoefficients": [0.5, 1.5, 0.1],
        "functionName": "exp",
    }

    assert all(
        deriv.evaluate(np.array([0.1]), np.array([0.3]))
        == 2 * np.exp(0.5 * 0.1 * 0.3 + 1.5 * 0.1 * 0.3**2 + 0.1 * 0.1**2 * 0.3**2)
    )


def test_loc_val_deriv(grid):

    deriv = dv.LocValExtractorDerivation("ext", grid, grid.numX - 1)

    assert deriv.dict() == {
        "type": "locValExtractorDerivation",
        "targetX": grid.numX - 1,
    }

    assert deriv.resultProperties["isScalar"]

    vals = np.ones(grid.numX)
    vals[grid.numX - 2] = 3.0

    assert deriv(vals).shape == (1,)
    assert deriv(vals)[0] == 3.0


def test_nd_deriv():

    # grids
    grid1 = np.linspace(1, 6, 10)
    grid2 = np.linspace(10, 173, 15)

    # 2D interpolation

    xg2, yg2 = np.meshgrid(grid1, grid2, indexing="ij", sparse=True)
    f = lambda x, y: x**2 + 2 * y + 1
    data2D = f(xg2, yg2)
    deriv2 = dv.NDInterpolationDerivation("interp1D", [grid1, grid2], data2D)

    assert deriv2.dict() == {
        "type": "nDLinInterpDerivation",
        "data": {
            "dims": [10, 15],
            "values": data2D.flatten(order="F").tolist(),
        },
        "grids": {
            "names": ["grid0", "grid1"],
            "grid0": grid1.tolist(),
            "grid1": grid2.tolist(),
        },
    }

    interp2 = RegularGridInterpolator((grid1, grid2), data2D)

    var1 = np.random.rand(4) * 5 + 1
    var2 = np.random.rand(4) * 163 + 10

    interpVals2 = interp2(np.array(list(zip(var1, var2))))

    assert np.all(interpVals2 == deriv2(var1, var2))
