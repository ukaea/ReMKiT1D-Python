import numpy as np
from typing import Union, List, Dict, cast, Tuple

from RMK_support.derivations import Textbook
from .variable_container import VariableContainer, Variable, MultiplicativeArgument
from .model_construction import Stencil
from .derivations import GenericDerivation, DerivationClosure
from .grid import Profile
import warnings


class StaggeredDivStencil(Stencil):

    def __init__(self):
        super().__init__(
            latexTemplate="\\nabla\\cdot\\left($0\\right)",
            properties={"stencilType": "staggeredDifferenceStencil"},
        )


class StaggeredGradStencil(Stencil):

    def __init__(self):
        super().__init__(
            latexTemplate="\\nabla\\left($0\\right)",
            properties={
                "stencilType": "staggeredDifferenceStencil",
                "ignoreJacobian": True,
            },
        )


class BCDivStencil(Stencil):

    def __init__(
        self,
        fluxJac: Variable,
        lowerBoundVar: Union[Variable, None] = None,
        isLeft: bool = False,
    ):

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
                else "\\text{" + self.__lowerBoundVar__.name.replace("_", "\_") + "}"
            )
        fluxJac = (
            latexRemap[self.__fluxJac__.name]
            if self.__fluxJac__.name in latexRemap
            else "\\text{" + self.__fluxJac__.name.replace("_", "\_") + "}"
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

    def __init__(self, fluxJac: Union[Variable, None] = None):

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
                else "\\text{" + self.__fluxJac__.name.replace("_", "\_") + "}"
            )
        return (
            "\\nabla_{c}\\cdot\\left("
            + fluxJac
            + " "
            + arg.latex(latexRemap)
            + "\\right)"
        )


class CentralDiffGradStencil(Stencil):

    def __init__(self):
        super().__init__(
            latexTemplate="\\nabla_{c}\\left($0\\right)",
            properties={
                "stencilType": "centralDifferenceInterpolated",
                "ignoreJacobian": True,
            },
        )


class DiffusionStencil(Stencil):

    def __init__(
        self, deriv: DerivationClosure, diffCoeffOnDualGrid=False, ignoreJacobian=False
    ):
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

    def __init__(self, momentOrder: int, momentHarmonic: int):
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

    def __init__(self, rowHarmonic: str, colHarmonic: str):
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

    def __init__(
        self,
        rowHarmonic: int,
        colHarmonic: int,
        CCoeff: Union[None, Profile, Variable] = None,
        interpCoeffs: Union[None, Profile, Variable] = None,
        cfAtZero: Union[Tuple[float, float], None] = None,
    ):
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
            stencil["fixedC"] = CCoeff.data.tolist()

        if isinstance(interpCoeffs, Profile):
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

    def __init__(
        self,
        rowHarmonic: int,
        colHarmonic: int,
        diffCoeff: Union[None, Profile, Variable] = None,
        adfAtZero: Union[Tuple[float, float], None] = None,
    ):
        stencil = {
            "stencilType": "vDiffusionStencil",
            "modelboundA": (
                "none" if not isinstance(diffCoeff, Variable) else diffCoeff.name
            ),
            "rowHarmonic": rowHarmonic,
            "colHarmonic": colHarmonic,
        }

        if isinstance(diffCoeff, Profile):
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

    def __init__(self, rowHarmonic: int, colHarmonic: int, integralIndex: int):
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

    def __init__(self, rowHarmonic: int, colHarmonic: int, integralIndex: int):
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

    def __init__(self, colHarmonic: int, momentOrder: int, termName: str):
        super().__init__(
            "\\langle \\left( \\frac{\\partial $0_{h="
            + str(colHarmonic)
            + "}}{\\partial t} \\right)_{\\text{"
            + termName.replace("_", "\_")
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

    def __init__(
        self,
        rowHarmonic: int,
        transitionIndex: int,
        fixedEnergyIndex: int,
        absorptionTerm=False,
        detailedBalanceTerm=False,
        transitionLatex: Union[None, str] = None,
    ):
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

    def __init__(
        self,
        rowHarmonic: int,
        transitionIndex: int,
        absorptionTerm=False,
        superelasticTerm=False,
        transitionLatex: Union[None, str] = None,
    ):
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

    def __init__(
        self,
        rowHarmonic: int,
        colHarmonic: int,
        distFunVar: Variable,
        densityVar: Variable,
        densityVarDual: Union[Variable, None] = None,
        densityVarOnBoundary: Union[Variable, None] = None,
        decompHarmonics: Union[List[int], None] = None,
        leftBoundary=False,
    ):

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

    def __init__(
        self,
        xStencil: Tuple[int, ...],
        fixedColumnVecs: Tuple[Profile, ...],
        varContColumnVars: Union[Tuple[Union[None, Variable], ...], None] = None,
        mbDataColumnVars: Union[Tuple[Union[None, Variable], ...], None] = None,
        latexTemplate: Union[str, None] = None,
    ):

        assert len(fixedColumnVecs) == len(
            xStencil
        ), "fixedColumnVecs must be of same length as xStencil in CustomFluid1DStencil"

        stencil = {"stencilType": "customFluid1DStencil", "xStencil": list(xStencil)}

        for i, vec in enumerate(fixedColumnVecs):
            stencil.update({"columnVector" + str(i + 1): vec.data.tolist()})

        if varContColumnVars is not None:
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
