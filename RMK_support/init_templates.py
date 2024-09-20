from .variable_container import VariableContainer
from . import simple_containers as sc
from typing import Union, List
import numpy as np


def addVarAndDual(
    vc: VariableContainer,
    varName: str,
    data: Union[np.ndarray, None] = None,
    isDerived=False,
    derivationRule: Union[None, dict] = None,
    isDistribution=False,
    isStationary=False,
    primaryOnDualGrid=False,
    units="normalized units",
    priority=0,
    dualSuffix="_dual",
    normSI: float = 1.0,
    unitSI: str = "",
) -> None:
    """Adds a variable (primary) and its dual (secondary) on the grid to given variable container. The secondary is initialized to 0

    Args:
        vc (VariableContainer): Variable container to add to
        varName (str): Name of variable on regular grid
        data (Union[numpy.ndarray,None], optional): Optional numpy array representing variable data. Defaults to None, which initializes data to 0.
        isDerived (bool, optional): True if both the primary variable is derived. Defaults to False.
        derivationRule (Union[None,dict], optional): Derivation rule for primary derived variable. Defaults to None.
        isDistribution (bool, optional): True if variable is a distribution. Defaults to False.
        isStationary (bool, optional): True if primary variable is stationary. Defaults to False.
        primaryOnDualGrid (bool, optional): True if the primary variable is on the dual grid. Defaults to False.
        units (str, optional): Units for both primary and secondary. Defaults to 'normalized units'.
        priority (int, optional): Variable priority for both primary and secondary. Defaults to 0 (highest priority).
        dualSuffix (str, optional): Suffix for the variable on the dual grid. Defaults to "_dual".
        normSI (float, optional) Optional normalisation constant for converting value to SI. Defaults to 1.0.
        unitSI (str, optional) Optional associated SI unit. Defaults to "".
    """

    primaryName = varName
    secondaryName = varName
    if primaryOnDualGrid:
        primaryName = primaryName + dualSuffix
        derivRule = "dualToGrid"
    else:
        secondaryName = secondaryName + dualSuffix
        derivRule = "gridToDual"
    onDualGrid = primaryOnDualGrid
    if isDistribution:
        onDualGrid = True
        derivRule = "distributionInterp"
    vc.setVariable(
        primaryName,
        data=data,
        isDerived=isDerived,
        derivationRule=derivationRule,
        isDistribution=isDistribution,
        isStationary=isStationary,
        isOnDualGrid=onDualGrid,
        units=units,
        priority=priority,
        normSI=normSI,
        unitSI=unitSI,
    )

    secondaryDerivationRule = sc.derivationRule(
        derivationName=derivRule, requiredVars=[primaryName]
    )
    onDualGrid = not primaryOnDualGrid
    if isDistribution:
        onDualGrid = True
    vc.setVariable(
        secondaryName,
        isDerived=True,
        derivationRule=secondaryDerivationRule,
        isDistribution=isDistribution,
        isOnDualGrid=onDualGrid,
        units=units,
        priority=priority,
        normSI=normSI,
        unitSI=unitSI,
    )
