#%%
import sys

sys.path.append('../')
from RMK_support import Node, atan, treeDerivation

def phi(X:Node) -> Node:
    """Function used when evaulating integrals of type in Chodura and Pohl 1971

    Args:
        X (Node): Related to the anisotropy coefficient through (X = alphasr - 1)

    Returns:
        Node: output of phi function
    """
    return atan(X**0.5)*X**-0.5
def K_LMN(alpha:Node,LMN:str) -> Node:
    """Calculates integral of form found in Chodura and Pohl 1971

    Args:
        X (Node): Related to the anisotropy coefficient through (X = alphasr - 1)
        LMN (str): String that states which integral is being calculated

    Raises:
        ValueError: Unknown K_LMN case

    Returns:
        Node: Result of integral
    """
    match LMN:
        case "200": 
            return X(alpha)**-1*(-1 + (1 + X(alpha))*phi(X(alpha)))
        case "002":
            return X(alpha)**-1*(1 - phi(X(alpha)))
        case "220":
            return 0.125*X(alpha)**-2*(3 + X(alpha) + (1 + X(alpha))*(X(alpha) - 3)*phi(X(alpha)))
        case "202":
            return 0.5*X(alpha)**-2*(-3 + (3 + X(alpha))*phi(X(alpha)))
        case "004":
            return X(alpha)**-2*(2 + 1/(1 + X(alpha)) - 3*phi(X(alpha)))
        case "420":
            return 0.03125*X(alpha)**-3*(-15 - 4*X(alpha) + 3*X(alpha)**2 + (15 + 9*X(alpha) - 3*X(alpha)**2 + 3*X(alpha)**3)*phi(X(alpha)))
        case "222":
            return 0.0625*X(alpha)**-3*(15 + X(alpha) + (-15 -6*X(alpha) + X(alpha)**2)*phi(X(alpha)))
        case "204":
            return 0.25*X(alpha)**-3*(-13 - 2/(1 + X(alpha)) + (15 + 3*X(alpha))*phi(X(alpha)))
        case "006":
            return 0.5*X(alpha)**-3*(8 + 9/(1 + X(alpha)) - 2/(1 + X(alpha))**2 - 15*phi(X(alpha)))
        case other:
            raise ValueError("Unknown K_LMN case")

def alphasr(TsPar:str,TsPerp:str,TrPar:str,TrPerp:str,massRatio:float) -> Node:
    """Calculates anisotropy coefficient for species s and r

    Args:
        TsPar (str): Parallel temperature of species s
        TsPerp (str): Perpendicular temperature of species s
        TrPar (str): Parallel temperature of species r
        TrPerp (str): Perpendicular temperature of species r
        massRatio (float): ratio of masses between species s and r

    Returns:
        Node: Anisotropy coefficient
    """
    TsParNode = Node(TsPar)
    TsPerpNode = Node(TsPerp)
    TrParNode = Node(TrPar)
    TrPerpNode = Node(TrPerp)

    return (TrPerpNode + massRatio*TsPerpNode)*(TrParNode + massRatio*TsParNode)

def X(alpha:Node) -> Node:
    """Calculates term X used in integrals found in Chodura and Pohl 1971

    Args:
        alpha (Node): anisotropy coefficient

    Returns:
        Node: X
    """
    return alpha - 1

def psi(alpha:Node) -> Node:
    return alpha**2*(K_LMN(X(alpha),"004") - K_LMN(X(alpha),"202")) + 0.5*alpha()*(K_LMN(X(alpha),"200") - K_LMN(X(alpha),"002"))

def xi(alpha:Node) -> Node:
    return 4*K_LMN(X(alpha),"220") - 2*K_LMN(X(alpha),"202") - K_LMN(X(alpha),"200") + K_LMN(X(alpha),"002")

def cPerp(nuee:Node,nuei:Node,nuii:Node,nuie:Node,alphae:Node,alphai:Node,isEl:bool) -> Node:
    if isEl:
        return -4*nuee*(6*alphae*K_LMN(X(alphae),"202") + 0.5*xi(X)) + 4*alphae*nuei*(-32*K_LMN(X(alphae),"222") + 4*K_LMN(X(alphae),"204") + 10*K_LMN(X(alphae),"202") - 2*K_LMN(X(alphae),"004") - K_LMN(X(alphae),"002"))
    return -4*nuii*(6*alphai*K_LMN(X(alphai),"202") + psi(alphai)) - 4*nuie*(K_LMN(X(alphae),"200") + alphae*K_LMN(X(alphae),"002"))

def cPar(nuee:Node,nuei:Node,nuii:Node,alphae:Node,alphai:Node,isEl:bool) -> Node:
    if isEl:
        return 2*nuee*psi(alphai) + 4*alphae**2*nuei*(-8*3**-1*alphae*K_LMN(X(alphae),"204") + 2*3**-1*alphae*K_LMN(X(alphae),"006") + 4*K_LMN(X(alphae),"202") - (1 - alphae*3**-1)*K_LMN(X(alphae),"004") - 0.5*K_LMN(X(alphae),"002"))
    return 2*psi(alphai)*nuii

def dPar(us:str,ur:str) -> Node:
    usNode = Node(us)
    urNode = Node(ur)

    return urNode - usNode

def ePerp(nuee:Node,nuei:Node,nuii:Node,alphae:Node,alphai:Node,isEl:bool) -> Node:
    if isEl:
        return 12*nuee*xi(X(alphae)) + 12*nuei*(16*alphae*K_LMN(X(alphae),"222") - 4*alphae*K_LMN(X(alphae),"204") - 2*(2*alphae - 1)*K_LMN(X(alphae),"202") + 2*alphae*K_LMN(X(alphae),"004") - K_LMN(X(alphae),"002"))
    return 12*xi(X(alphai))*nuii

def ePar(nuee:Node,nuei:Node,nuii:Node,nuie:Node,alphae:Node,alphai:Node,isEl:bool) -> Node:
    if isEl:
        return -12*nuee*psi(alphai) + 4*nuei*alphae*(4*alphae**2*K_LMN(X(alphae),"204") - 2*alphae**2*K_LMN(X(alphae),"006") - 6*alphae*K_LMN(X(alphae),"202") + 4*alphae*K_LMN(X(alphae),"004") - 1.5*K_LMN(X(alphae),"002"))
    return -12*psi(alphai)*nuii - 12*nuie*alphae*K_LMN(X(alphae),"002")

def kPerp(nuei:Node,alphae:Node,alphai:Node,isEl:bool) -> Node:
    # NOT FINISHED DO I INCLUDE NUii?
    if isEl:
        return -4*nuei*(2*psi(alphai) - alphae**2*(6*K_LMN(X(alphae),"202") - K_LMN(X(alphae),"002")))
    else:
        return 0
    
def kPar(nuee:Node,nuei:Node,nuii:Node,nuie:Node,alphae:Node,alphai:Node,isEl:bool) -> Node:
    # NOT FINISHED DO I INCLUDE NUii?
    if isEl:
        return -4*nuei*(-4*psi(alphai) + alphae*(2*alphae*K_LMN(X(alphae),"004") + K_LMN(X(alphae),"002")))
    return 0

# %%
