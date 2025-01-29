from . import amjuel_reader as ar
from .derivations import GenIntPolynomialDerivation
from .variable_container import Variable
import numpy as np
from typing import Dict, Tuple, Union


def AMJUELDeriv(
    derivName: str,
    reaction: str,
    section: str,
    amjuelFilename="../data/amjuel.tex",
    timeNorm: float = 0.72204953888999173e-7,
    densNorm: float = 1e19,
    tempNorm: float = 1,
):
    fit = ar.read2DAMJUELFitCoeffs(
        ar.loadAMJUELReactionData(reaction, section, amjuelFilename)
    )
    polyCoeffs = fit.flatten()
    polyPowers = np.array([[j, i] for (i, j), _ in np.ndenumerate(fit)])
    multConst = 1e-6 * timeNorm * densNorm / tempNorm
    funcName = "exp"
    return GenIntPolynomialDerivation(
        derivName, polyPowers, polyCoeffs, multConst, funcName
    )


def AMJUELDeriv1D(
    derivName: str,
    reaction: str,
    section: str,
    coefName="b",
    amjuelFilename="../data/amjuel.tex",
    timeNorm: float = 0.72204953888999173e-7,
    densNorm: float = 1e19,
    tempNorm: float = 1,
):
    fit = ar.read1DAMJUELFitCoeffs(
        ar.loadAMJUELReactionData(reaction, section, amjuelFilename), coefName=coefName
    )
    polyCoeffs = fit.flatten()
    polyPowers = np.array(list(range(len(fit)))).reshape((len(fit), 1))
    multConst = 1e-6 * timeNorm * densNorm / tempNorm
    funcName = "exp"
    return GenIntPolynomialDerivation(
        derivName, polyPowers, polyCoeffs, multConst, funcName
    )


def generateAMJUELDerivs(
    derivDict: Dict[str, Tuple[str, str]],
    norms: Dict[str, float],
    timeNorm: Union[float, None] = None,
    amjuelFilename="../data/amjuel.tex",
):
    usedTimeNorm = norms["time"]
    if timeNorm is not None:
        usedTimeNorm = timeNorm

    derivs: Dict[str, GenIntPolynomialDerivation] = {}
    for key in derivDict:
        derivs[key] = AMJUELDeriv(
            key,
            derivDict[key][0],
            derivDict[key][1],
            amjuelFilename=amjuelFilename,
            timeNorm=usedTimeNorm,
            tempNorm=norms["eVTemperature"] if "Energy" in key else 1,
            densNorm=norms["density"],
        )

    return derivs


def amjuelHydrogenAtomDerivs():
    return {
        "ionPart": ("2.1.5", "H.4"),
        "ionEnergy": ("2.1.5", "H.10"),
        "recombPart": ("2.1.8", "H.4"),
        "recombEnergy": ("2.1.8", "H.10"),
    }


def AMJUELLogVars(
    norms: Dict[str, float], dens: Variable, temp: Variable
) -> Tuple[Variable, Variable]:

    logn = GenIntPolynomialDerivation(
        "log" + dens.name,
        np.ones((1, 1)),
        np.ones(1) * norms["density"] / 1e14,
        funcName="log",
    )(dens)

    logT = GenIntPolynomialDerivation(
        "log" + temp.name,
        np.ones((1, 1)),
        np.ones(1) * norms["eVTemperature"],
        funcName="log",
    )(temp)

    return logn, logT
