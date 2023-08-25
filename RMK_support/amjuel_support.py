from . import amjuel_reader as ar
from . import crm_support as crm
from .rk_wrapper import RKWrapper
from . import simple_containers as sc
import numpy as np


def AMJUELDeriv(
    reaction,
    section,
    amjuelFilename="../data/amjuel.tex",
    timeNorm=0.72204953888999173e-7,
    densNorm=1e19,
    tempNorm=1,
):
    fit = ar.read2DAMJUELFitCoeffs(
        ar.loadAMJUELReactionData(reaction, section, amjuelFilename)
    )
    polyCoeffs = fit.flatten().tolist()
    polyPowers = [[j, i] for (i, j), val in np.ndenumerate(fit)]
    multConst = 1e-6 * timeNorm * densNorm / tempNorm
    funcName = "exp"
    return sc.generalizedIntPolyDerivation(polyPowers, polyCoeffs, multConst, funcName)


def AMJUELDeriv1D(
    reaction,
    section,
    coefName="b",
    amjuelFilename="../data/amjuel.tex",
    timeNorm=0.72204953888999173e-7,
    densNorm=1e19,
    tempNorm=1,
):
    fit = ar.read1DAMJUELFitCoeffs(
        ar.loadAMJUELReactionData(reaction, section, amjuelFilename), coefName=coefName
    )
    polyCoeffs = fit.flatten().tolist()
    polyPowers = [i for i, val in np.ndenumerate(fit)]
    multConst = 1e-6 * timeNorm * densNorm / tempNorm
    funcName = "exp"
    return sc.generalizedIntPolyDerivation(polyPowers, polyCoeffs, multConst, funcName)


def addAMJUELDerivs(
    derivDict: dict,
    rk: RKWrapper,
    timeNorm=0.72204953888999173e-7,
    amjuelFilename="../data/amjuel.tex",
):
    for key in derivDict.keys():
        rk.addCustomDerivation(
            key,
            AMJUELDeriv(
                derivDict[key][0],
                section=derivDict[key][1],
                amjuelFilename=amjuelFilename,
                timeNorm=timeNorm,
                tempNorm=rk.normalization["eVTemperature"] if "Energy" in key else 1,
                densNorm=rk.normalization["density"],
            ),
        )


def amjuelHydrogenAtomDerivs():
    return {
        "ionPart": ("2.1.5", "H.4"),
        "ionEnergy": ("2.1.5", "H.10"),
        "recombPart": ("2.1.8", "H.4"),
        "recombEnergy": ("2.1.8", "H.10"),
    }


def addLogVars(rk: RKWrapper, densName: str, tempName: str):
    if "lognDeriv" not in rk.customDerivs["tags"]:
        rk.addCustomDerivation(
            "lognDeriv",
            sc.generalizedIntPolyDerivation(
                [[1]], [rk.normalization["density"] / 1e14], funcName="log"
            ),
        )
        rk.addCustomDerivation(
            "logTDeriv",
            sc.generalizedIntPolyDerivation(
                [[1]], [rk.normalization["eVTemperature"]], funcName="log"
            ),
        )
    rk.addVar(
        "log" + densName,
        isDerived=True,
        derivationRule=sc.derivationRule("lognDeriv", [densName]),
        isCommunicated=True,
    )
    rk.addVar(
        "log" + tempName,
        isDerived=True,
        derivationRule=sc.derivationRule("logTDeriv", [tempName]),
        isCommunicated=True,
    )
