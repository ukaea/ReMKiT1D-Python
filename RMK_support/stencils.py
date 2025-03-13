import numpy as np
from typing import Union, List, Dict, Tuple, Optional
from RMK_support.derivations import Textbook
from .variable_container import Variable, MultiplicativeArgument
from .model_construction import Stencil
from .derivations import GenericDerivation, DerivationClosure
from .grid import Profile


class StaggeredDivStencil(Stencil):
    """Staggered divergence stencil, used when row and column variables live on different grids"""

    def __init__(self):
        super().__init__(
            latexTemplate="\\nabla\\cdot\\left($0\\right)",
            properties={"stencilType": "staggeredDifferenceStencil"},
        )


class StaggeredGradStencil(Stencil):
    """Staggered gradient stencil, used when row and column variables live on different grids"""

    def __init__(self):
        super().__init__(
            latexTemplate="\\nabla\\left($0\\right)",
            properties={
                "stencilType": "staggeredDifferenceStencil",
                "ignoreJacobian": True,
            },
        )


class BCDivStencil(Stencil):
    """Boundary condition divergence stencil"""

    def __init__(
        self,
        fluxJac: Variable,
        lowerBoundVar: Optional[Variable] = None,
        isLeft: bool = False,
    ):
        """Boundary condition divergence stencil with extrapolation and optional lower bounds. Provides the boundary condition for the div(u * vars) operator, where u is the flux Jacobian/flow speed, optionally putting a lower bound on the absolute value of the flow speed (negative on the left boundary)

        Args:
            fluxJac (Variable): Flow speed variable, should live on the regular grid
            lowerBoundVar (Optional[Variable], optional): Lower bound variable - should live on the regular grid. Defaults to None.
            isLeft (bool, optional): True if this stencil is for the left boundary. Defaults to False.
        """
        stencil = {
            "stencilType": "boundaryStencil",
            "fluxJacVar": fluxJac.name,
            "leftBoundary": isLeft,
        }

        if lowerBoundVar is not None:
            stencil["lowerBoundVar"] = lowerBoundVar.name

        super().__init__(properties=stencil)

        self.__isLeft__ = isLeft
        self.__fluxJac__ = fluxJac
        assert (
            not fluxJac.isOnDualGrid
        ), "Flux Jacobian variable for BCDivStencil must live on the regular grid"
        self.__lowerBoundVar__ = lowerBoundVar

    def latex(self, arg: MultiplicativeArgument, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        LR = "L" if self.__isLeft__ else "R"
        boundVar = ""
        if self.__lowerBoundVar__ is not None:
            boundVar += ", "
            boundVar += (
                latexRemap[self.__lowerBoundVar__.name]
                if self.__lowerBoundVar__.name in latexRemap
                else "\\text{" + self.__lowerBoundVar__.name.replace("_", r"\_") + "}"
            )
        fluxJac = (
            latexRemap[self.__fluxJac__.name]
            if self.__fluxJac__.name in latexRemap
            else "\\text{" + self.__fluxJac__.name.replace("_", r"\_") + "}"
        )
        return (
            "\\nabla_{BC,"
            + LR
            + boundVar
            + "}\\cdot\\left("
            + fluxJac
            + " "
            + arg.latex(latexRemap)
            + "\\right)"
        )


class BCGradStencil(Stencil):
    """Boundary condition gradient stencil"""

    def __init__(self, isLeft: bool = False):
        LR = "L" if isLeft else "R"
        super().__init__(
            "\\nabla_{BC," + LR + "}\\left($0\\right)",
            {
                "stencilType": "boundaryStencil",
                "leftBoundary": isLeft,
                "ignoreJacobian": True,
            },
        )


class CentralDiffDivStencil(Stencil):
    """Central difference divergence stencil - to be used when both row and column variables live on the same grid"""

    def __init__(self, fluxJac: Optional[Variable] = None):
        """Central difference divergence stencil - to be used when both row and column variables live on the same grid. The column variables are linearly interpolated onto corresponding cell edges/centres, and optionally multiplied by a separately linearly interpolated variable - the flux jacobian, before the central difference is taken.

        Args:
            fluxJac (Optional[Variable], optional): Optional flux jacobian to be separately linearly interpolated and used to multiply the column variables - should live on the same grid as the row/column variables. Defaults to None.
        """
        stencil = {
            "stencilType": "centralDifferenceInterpolated",
            "interpolatedVarName": fluxJac.name if fluxJac is not None else "none",
        }
        super().__init__(properties=stencil)

        self.__fluxJac__ = fluxJac

    def latex(self, arg: MultiplicativeArgument, **kwargs):
        latexRemap: Dict[str, str] = kwargs.get("latexRemap", {})
        fluxJac = ""
        if self.__fluxJac__ is not None:
            fluxJac = (
                latexRemap[self.__fluxJac__.name]
                if self.__fluxJac__.name in latexRemap
                else "\\text{" + self.__fluxJac__.name.replace("_", r"\_") + "}"
            )
        return (
            "\\nabla_{c}\\cdot\\left("
            + fluxJac
            + " "
            + arg.latex(latexRemap)
            + "\\right)"
        )


class CentralDiffGradStencil(Stencil):
    """Central difference divergence stencil - to be used when both row and column variables live on the same grid"""

    def __init__(self):
        super().__init__(
            latexTemplate="\\nabla_{c}\\left($0\\right)",
            properties={
                "stencilType": "centralDifferenceInterpolated",
                "ignoreJacobian": True,
            },
        )


class DiffusionStencil(Stencil):
    """Spatial diffusion stencil - works only when both row and column variables are on the regular grid"""

    def __init__(
        self, deriv: DerivationClosure, diffCoeffOnDualGrid=False, ignoreJacobian=False
    ):
        """Spatial diffusion stencil - works only when both row and column variables are on the regular grid

        Args:
            deriv (DerivationClosure): A derivation closure used to calculate the diffusion coefficient
            diffCoeffOnDualGrid (bool, optional): True if the diffusion coefficient is calculated on the dual grid, otherwise it will be linearly interpolated onto cell edges. Defaults to False.
            ignoreJacobian (bool, optional): If true will ignore the Jacobian in the outer divergence part of the diffusion operator. Defaults to False.
        """
        assert (
            deriv.numArgs == 0
        ), "deriv in DiffusionStencil must be a complete closure"
        cdot = "" if ignoreJacobian else "\\cdot"
        super().__init__(
            "\\nabla " + cdot + " \\left[D \\nabla \\left($0\\right)\\right]",
            {
                "stencilType": "diffusionStencil",
                "ruleName": deriv.name,
                "requiredVarNames": deriv.fillArgs(),
                "doNotInterpolateDiffCoeff": diffCoeffOnDualGrid,
                "ignoreJacobian": ignoreJacobian,
            },
        )
        self.__deriv__ = deriv

    def registerDerivs(self, container: Textbook):
        self.__deriv__.registerComponents(container)


class MomentStencil(Stencil):
    """Stencil taking the moment of a distribution harmonic - distribution column, fluid row"""

    def __init__(self, momentOrder: int, momentHarmonic: int):
        """Stencil taking the moment of a distribution harmonic - distribution column, fluid row

        Args:
            momentOrder (int): Order of the moment being taken
            momentHarmonic (int): Harmonic index of the distribution to take the moment of
        """
        super().__init__(
            "\\langle $0 \\rangle_{h="
            + str(momentHarmonic)
            + ",n="
            + str(momentOrder)
            + "}",
            {
                "stencilType": "momentStencil",
                "momentOrder": momentOrder,
                "momentHarmonic": momentHarmonic,
            },
        )


class DistGradStencil(Stencil):
    """Distribution spatial gradient stencil"""

    def __init__(self, rowHarmonic: int, colHarmonic: int):
        """Return stencil representing the evolution of rowHarmonic due to spatial gradients of colHarmonic. If the harmonics represent l-numbers of different parities and the variables are staggered the difference will be calculated using forward/backwards staggered difference, otherwise it will be calculated using central difference with interpolation at spatial cell faces.

        Args:
            rowHarmonic (int): Index of row (evolved) harmonic
            colHarmonic (int): Indef of column (implicit) harmonic

        """
        super().__init__(
            "\\delta_{h,"
            + str(rowHarmonic)
            + "}\\nabla \\left( $0_{h'="
            + str(colHarmonic)
            + "}\\right)",
            {
                "stencilType": "kineticSpatialDiffStencil",
                "rowHarmonic": rowHarmonic,
                "colHarmonic": colHarmonic,
            },
        )


class DDVStencil(Stencil):
    """Velocity space gradient stencil"""

    def __init__(
        self,
        rowHarmonic: int,
        colHarmonic: int,
        CCoeff: Union[None, Profile, Variable] = None,
        interpCoeffs: Union[None, Profile, Variable] = None,
        cfAtZero: Optional[Tuple[float, float]] = None,
    ):
        """Stencil representing the evolution of rowHarmonic due to d(C*f_l)/dv, where l is the column harmonic. C is assumed to be defined on right velocity cell boundaries, and f_l is interpolated to those boundaries using either standard or user-defined interpolation coefficients (also defined at right cell boundaries). f_{n+1/2} is given by (1-interp(n))*f_{n} + interp(n)*f_{n+1}, where n is a velocity cell index.

        Args:
            rowHarmonic (int): Index of row (evolved) harmonic
            colHarmonic (int): Index of column (implicit) harmonic
            CCoeff (Union[None, Profile, Variable], optional): C coefficient in the above formula (on right velocity cell edges), if it's a Profile it should be a velocity space profile, and if it's a variable it's assumed to be a modelbound single harmonic variable. Defaults to None.
            interpCoeffs (Union[None, Profile, Variable], optional): Interpolation coefficient in the above formula (on right cell edges), if it's a Profile it should be a velocity space profile, and if it's a variable it's assumed to be a modelbound single harmonic variable. Defaults to None.
            cfAtZero (Optional[Tuple[float,float]], optional): Extrapolation coefficients of C*f at zero in the form A1*f(v1)+A2*f(v2). Defaults to None, resulting in (0,0).
        """

        if isinstance(CCoeff, Variable):
            assert (
                CCoeff.isSingleHarmonic
            ), "CCoeff must be a single harmonic if it's a variable"
        if isinstance(interpCoeffs, Variable):
            assert (
                interpCoeffs.isSingleHarmonic
            ), "interpCoeffs must be a single harmonic if it's a variable"
        stencil = {
            "stencilType": "ddvStencil",
            "modelboundC": "none" if not isinstance(CCoeff, Variable) else CCoeff.name,
            "modelboundInterp": (
                "none" if not isinstance(interpCoeffs, Variable) else interpCoeffs.name
            ),
            "rowHarmonic": rowHarmonic,
            "colHarmonic": colHarmonic,
        }

        if isinstance(CCoeff, Profile):
            assert CCoeff.dim == "V", "CCoeff profile must be a velocity space profile"
            stencil["fixedC"] = CCoeff.data.tolist()

        if isinstance(interpCoeffs, Profile):
            assert (
                interpCoeffs.dim == "V"
            ), "interpCoeffs profile must be a velocity space profile"
            stencil["fixedInterp"] = interpCoeffs.data.tolist()

        if cfAtZero is not None:
            stencil["cfAtZero"] = list(cfAtZero)

        if CCoeff is None:
            latexTemplate = (
                "\\frac{\\partial $0_{h'=" + str(colHarmonic) + "}}{\\partial v}"
            )
        else:
            C = CCoeff.latex() if isinstance(CCoeff, Profile) else "C_{MB}"
            latexTemplate = (
                "\\frac{\\partial}{\\partial v}\\left("
                + C
                + "$0_{h'="
                + str(colHarmonic)
                + "}\\right)"
            )
        latexTemplate = "\\delta_{h," + str(rowHarmonic) + "}" + latexTemplate
        super().__init__(latexTemplate, properties=stencil)


class D2DV2Stencil(Stencil):
    """Velocity space diffusion stencil"""

    def __init__(
        self,
        rowHarmonic: int,
        colHarmonic: int,
        diffCoeff: Union[None, Profile, Variable] = None,
        adfAtZero: Optional[Tuple[float, float]] = None,
    ):
        """Return stencil representing the evolution of rowHarmonic due to d(A*d/f_l/dv)/dv, where l is the column harmonic. A is assumed to be defined on right velocity cell boundaries.

        Args:
            rowHarmonic (int): Index of row (evolved) harmonic
            colHarmonic (int): Index of column (implicit) harmonic
            diffCoeff (Union[None, Profile, Variable], optional): A coefficient in the above formula (on right velocity cell edges), if it's a Profile it should be a velocity space profile, and if it's a variable it's assumed to be a modelbound single harmonic variable. Defaults to None.
            adfAtZero (Optional[Tuple[float,float]], optional): Extrapolation coefficients (length 2) of A*df/dv at zero in the form A1*f(v1)+A2*f(v2). Defaults to None.
        """
        if isinstance(diffCoeff, Variable):
            assert (
                diffCoeff.isSingleHarmonic
            ), "diffCoeff must be a single harmonic if it is a variable"
        stencil = {
            "stencilType": "vDiffusionStencil",
            "modelboundA": (
                "none" if not isinstance(diffCoeff, Variable) else diffCoeff.name
            ),
            "rowHarmonic": rowHarmonic,
            "colHarmonic": colHarmonic,
        }

        if isinstance(diffCoeff, Profile):
            assert (
                diffCoeff.dim == "V"
            ), "diffCoeff profile must be a velocity space profile"
            stencil["fixedA"] = diffCoeff.data.tolist()

        if adfAtZero is not None:
            stencil["adfAtZero"] = list(adfAtZero)

        if diffCoeff is None:
            latexTemplate = (
                "\\frac{\\partial^2 $0_{h=" + str(colHarmonic) + "}}{\\partial v^2}"
            )
        else:
            A = diffCoeff.latex() if isinstance(diffCoeff, Profile) else "A_{MB}"
            latexTemplate = (
                "\\frac{\\partial}{\\partial v}\\left("
                + A
                + "\\frac{\\partial $0_{h'="
                + str(colHarmonic)
                + "}}{\\partial v}\\right)"
            )
        latexTemplate = "\\delta_{h," + str(rowHarmonic) + "}" + latexTemplate
        super().__init__(latexTemplate, properties=stencil)


class ShkarofskyIStencil(Stencil):
    """Shkarofsky I integral stencil"""

    def __init__(self, rowHarmonic: int, colHarmonic: int, integralIndex: int):
        """Shkarofsky I integral stencil acting on a distribution

        Args:
            rowHarmonic (int): Index of row (evolved) harmonic
            colHarmonic (int): Index of column (implict) harmonic
            integralIndex (int): Index of I integral
        """
        super().__init__(
            "\\delta_{h,"
            + str(rowHarmonic)
            + "}I_"
            + str(integralIndex)
            + "\\left($0_{h'="
            + str(colHarmonic)
            + "}\\right)",
            {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": False,
                "rowHarmonic": rowHarmonic,
                "colHarmonic": colHarmonic,
                "integralIndex": integralIndex,
            },
        )


class ShkarofskyJStencil(Stencil):
    """Shkarofsky J integral stencil"""

    def __init__(self, rowHarmonic: int, colHarmonic: int, integralIndex: int):
        """Shkarofsky J integral stencil acting on a distribution

        Args:
            rowHarmonic (int): Index of row (evolved) harmonic
            colHarmonic (int): Index of column (implict) harmonic
            integralIndex (int): Index of J integral
        """
        super().__init__(
            "\\delta_{h,"
            + str(rowHarmonic)
            + "}J_"
            + str(integralIndex)
            + "\\left($0_{h'="
            + str(colHarmonic)
            + "}\\right)",
            {
                "stencilType": "shkarofskyIJStencil",
                "JIntegral": True,
                "rowHarmonic": rowHarmonic,
                "colHarmonic": colHarmonic,
                "integralIndex": integralIndex,
            },
        )


class TermMomentStencil(Stencil):
    """Stencil taking a moment of another term evolving the distribution function"""

    def __init__(self, colHarmonic: int, momentOrder: int, termName: str):
        """Stencil which represent taking the moment of a given a term which evolves a single harmonic of a distribution variable. The term must be contained in the same model as the term resulting from this stencil. The term whose moment is taken should be local in space and cannot perform interpolation if implicit in a distribution variable

        Args:
            colHarmonic (int): Harmonic evolved by the targeted term
            momentOrder (int): Harmonic evolved by the targeted term
            termName (str): Name of term in model whose moment is to be taken
        """
        super().__init__(
            "\\langle \\left( \\frac{\\partial $0_{h="
            + str(colHarmonic)
            + "}}{\\partial t} \\right)_{\\text{"
            + termName.replace("_", r"\_")
            + "}} \\rangle_{n="
            + str(momentOrder)
            + "}",
            {
                "stencilType": "termMomentStencil",
                "momentOrder": momentOrder,
                "colHarmonic": colHarmonic,
                "termName": termName,
            },
        )


class FixedEnergyBoltzmannStencil(Stencil):
    """Boltzmann collision integral emission/absorption term stencil for processes with a constant transition energy"""

    def __init__(
        self,
        rowHarmonic: int,
        transitionIndex: int,
        fixedEnergyIndex: int,
        absorptionTerm=False,
        detailedBalanceTerm=False,
        transitionLatex: Optional[str] = None,
    ):
        """Boltzmann collision operator for given harmonic and transition

        Args:
            rowHarmonic (int): Evolved and implicit harmonic index transitionIndex (int): Index of the transition in CRM modelbound data which this stencil is modelling
            fixedEnergyIndex (int): Index of the energy value of this transition in the CRM inelastic data
            absorptionTerm (bool, optional): Set to true if this stencil represents the absorption term as opposed to the emission term. Defaults to False.
            detailedBalanceTerm (bool, optional): et to true if this stencil is associated with a detailed balance transition. Defaults to False.
            transitionLatex (Optional[str], optional): LaTeX representation of the transition. Defaults to None, using "T=transitionIndex".
        """
        AE = "A" if absorptionTerm else "E"
        transition = (
            transitionLatex
            if transitionLatex is not None
            else "T=" + str(transitionIndex)
        )
        super().__init__(
            "\\delta_{h,"
            + str(rowHarmonic)
            + "}C^"
            + AE
            + "_{"
            + transition
            + "}\\left($0_h\\right)",
            {
                "stencilType": "boltzmannStencil",
                "rowHarmonic": rowHarmonic,
                "fixedEnergyIndex": fixedEnergyIndex,
                "transitionIndex": transitionIndex,
                "absorptionTerm": absorptionTerm,
                "detailedBalanceTerm": detailedBalanceTerm,
            },
        )


class VariableEnergyBoltzmannStencil(Stencil):
    """Boltzmann collision integral emission/absorption term stencil for processes with a variable transition energy"""

    def __init__(
        self,
        rowHarmonic: int,
        transitionIndex: int,
        absorptionTerm=False,
        superelasticTerm=False,
        transitionLatex: Optional[str] = None,
    ):
        """Boltzmann collision integral emission/absorption term stencil for processes with a variable transition energy

        Args:
            rowHarmonic (int): Evolved and implicit harmonic index
            transitionIndex (int): Index of the transition in CRM modelbound data which this stencil is modelling
            absorptionTerm (bool, optional): Set to true if this stencil represents the absorption term as opposed to the emission term. Defaults to False.
            superelasticTerm (bool, optional): Set to true if this stencil is associated with a superelastic (negative transition energy) transiton. Defaults to False.
            transitionLatex (Optional[str], optional): LaTeX representation of the transition. Defaults to None, using "T=transitionIndex".
        """
        AE = "A" if absorptionTerm else "E"
        transition = (
            transitionLatex
            if transitionLatex is not None
            else "T=" + str(transitionIndex)
        )
        super().__init__(
            "\\delta_{h,"
            + str(rowHarmonic)
            + "}C^"
            + AE
            + "_{"
            + transition
            + "}\\left($0_h\\right)",
            {
                "stencilType": "variableBoltzmannStencil",
                "rowHarmonic": rowHarmonic,
                "transitionIndex": transitionIndex,
                "absorptionTerm": absorptionTerm,
                "superelasticTerm": superelasticTerm,
            },
        )


class LBCStencil(Stencil):
    """Logical boundary condition stencil acting on distribution variable"""

    def __init__(
        self,
        rowHarmonic: int,
        colHarmonic: int,
        distFunVar: Variable,
        densityVar: Variable,
        densityVarDual: Optional[Variable] = None,
        densityVarOnBoundary: Optional[Variable] = None,
        decompHarmonics: Optional[List[int]] = None,
        leftBoundary=False,
    ):
        """Electron logical boundary condition stencil v*f_b/dx acting on rowHarmonic due to the decomposition of colHarmonic, which extrapolates the distribution through a scaling derivation

        Args:
            rowHarmonic (int):  Index of row (evolved) harmonic
            colHarmonic (int): Index of column (implicit) harmonic
            distFun (Variable): Electron distribution variable
            density (Variable): Electron density variable
            densityDual (Optional[Variable], optional): Dual electron density variable - needed when using staggered grids. Defaults to None.
            densityOnBoundary (Optional[Variable], optional): Value of the electron density on the boundary - if not present will only extrapolate the distribution to the cell centre closes to the boundary, assuming it stays constant from that point to the boundary - scalar variable. Defaults to None.
            decompHarmonics (Optional[List[int]], optional): List of harmonics included in the decomposition of column harmonic. **NOTE**: These are the actual implicit harmonics, and in case of a staggered distribution on the right boundary must all have the same l-number parity. Defaults to None, resulting in all harmonics.
            leftBoundary (bool, optional): True if on left boundary. Defaults to False.
        """

        if densityVarOnBoundary is not None:
            assert (
                densityVarOnBoundary.isScalar
            ), "LBC boundary density variable must be a scalar"
        if densityVarDual is not None:
            assert (
                densityVarDual.isOnDualGrid
            ), "Dual density variable in LBC stencil must live on dual grid"
        self.__deriv__ = GenericDerivation(
            ("left" if leftBoundary else "right") + "DistExt",
            0,
            {
                "type": "distScalingExtrapDerivation",
                "extrapolateToBoundary": isinstance(densityVarOnBoundary, Variable),
                "staggeredVars": isinstance(densityVarDual, Variable),
                "leftBoundary": leftBoundary,
            },
            "\\text{DistExt}_" + "L" if leftBoundary else "R",
        )
        reqVars = [distFunVar.name, densityVar.name]
        if densityVarDual is not None:
            reqVars.append(densityVarDual.name)
        if densityVarOnBoundary is not None:
            reqVars.append(densityVarOnBoundary.name)

        stencil = {
            "stencilType": "scalingLogicalBoundaryStencil",
            "rowHarmonic": rowHarmonic,
            "colHarmonic": colHarmonic,
            "leftBoundary": leftBoundary,
        }
        if decompHarmonics is not None:
            stencil["includedDecompHarmonics"] = decompHarmonics
        stencil.update({"ruleName": self.__deriv__.name, "requiredVarNames": reqVars})
        super().__init__(
            latexTemplate="\\delta_{h,"
            + str(rowHarmonic)
            + "}\\nabla_{LBC,"
            + ("L" if leftBoundary else "R")
            + "}\\left($0_{h'="
            + str(colHarmonic)
            + ("" if decompHarmonics is None else "{,\\text{partial}}")
            + "}\\right)",
            properties=stencil,
        )

    def registerDerivs(self, container: Textbook):
        container.register(self.__deriv__, ignoreDuplicates=True)


class CustomFluid1DStencil(Stencil):
    """A custom 1D fluid stencil based on relative stencil and column vector data"""

    def __init__(
        self,
        xStencil: Tuple[int, ...],
        fixedColumnVecs: Tuple[Profile, ...],
        varContColumnVars: Optional[Tuple[Optional[Variable], ...]] = None,
        mbDataColumnVars: Optional[Tuple[Optional[Variable], ...]] = None,
        latexTemplate: Optional[str] = None,
    ):
        """A custom 1D fluid stencil based on relative stencil and column vector data

        Args:
            xStencil (Tuple[int, ...]): Relative stencil (eg. (-1,0,1) will generate a tridiagonal stencil)
            fixedColumnVecs (Tuple[Profile, ...]): Tuple of spatial profiles, one for each entry in the stencil, corresponding to that column.
            varContColumnVars (Optional[Tuple[Optional[Variable], ...]], optional): Tuple of global fluid variables to multiply the fixed column vectors with, one for each stencil entry. If a specific column does not require a variable use None in that column. Defaults to None.
            mbDataColumnVars (Optional[Tuple[Optional[Variable], ...]], optional): Tuple of modelbound fluid variables to multiply the fixed column vectors with, one for each stencil entry. If a specific column does not require a variable use None in that column. Defaults to None.
            latexTemplate (Optional[str], optional): Stencil LaTeX template - must contain $0 - this is where the argument will go. Defaults to None.
        """
        assert len(fixedColumnVecs) == len(
            xStencil
        ), "fixedColumnVecs must be of same length as xStencil in CustomFluid1DStencil"

        stencil: Dict[str, object] = {
            "stencilType": "customFluid1DStencil",
            "xStencil": list(xStencil),
        }

        for i, vec in enumerate(fixedColumnVecs):
            assert vec.dim == "X", "fixedColumnVecs must be spatial profiles"
            stencil.update({"columnVector" + str(i + 1): vec.data.tolist()})

        if varContColumnVars is not None:

            for var in varContColumnVars:
                if var is not None:
                    assert (
                        var.isFluid
                    ), "All variables in varContColumnVars must be fluid"
            assert len(varContColumnVars) == len(
                xStencil
            ), "varContColumnVars must be of same length as xStencil in CustomFluid1DStencil"
            stencil.update(
                {
                    "columnVarContVars": [
                        var.name if var is not None else "none"
                        for var in varContColumnVars
                    ]
                }
            )
        else:
            stencil.update({"columnVarContVars": ["none" for _ in xStencil]})
        if mbDataColumnVars is not None:
            for var in mbDataColumnVars:
                if var is not None:
                    assert (
                        var.isFluid
                    ), "All variables in mbDataColumnVars must be fluid"
            assert len(mbDataColumnVars) == len(
                xStencil
            ), "mbDataColumnVars must be of same length as xStencil in CustomFluid1DStencil"
            stencil.update(
                {
                    "columnMBDataVars": [
                        var.name if var is not None else "none"
                        for var in mbDataColumnVars
                    ]
                }
            )
        else:
            stencil.update({"columnMBDataVars": ["none" for _ in xStencil]})

        super().__init__(
            (
                latexTemplate
                if latexTemplate is not None
                else "\\text{CustomStencil}\\left($0\\right)"
            ),
            properties=stencil,
        )
