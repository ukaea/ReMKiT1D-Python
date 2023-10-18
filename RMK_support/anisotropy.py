#%%
import numpy as np
import sys

sys.path.append('../')
from RMK_support import Node, atan

def phi(X):
    return atan(X**0.5)*X**-0.5

def K_LMN(X,LMN:str) -> float:
    match LMN:
        case "200": 
            return X**-1*(-1 + (1 + X)*phi(X))
        case "002":
            return X**-1*(1 - phi(X))
        case "220":
            return 0.125*X**-2*(3 + X + (1 + X)*(X - 3)*phi(X))
        case "202":
            return 0.5*X**-2*(-3 + (3 + X)*phi(X))
        case "004":
            return X**-2*(2 + 1/(1 + X) - 3*phi(X))
        case "420":
            return 0.03125*X**-3*(-15 - 4*X + 3*X**2 + (15 + 9*X - 3*X**2 + 3*X**3)*phi(X))
        case "222":
            return 0.0625*X**-3*(15 + X + (-15 -6*X + X**2)*phi(X))
        case "204":
            return 0.25*X**-3*(-13 - 2/(1 + X) + (15 + 3*X)*phi(X))
        case "006":
            return 0.5*X**-3*(8 + 9/(1 + X) - 2/(1 + X)**2 - 15*phi(X))
        case other:
            return None

def alphasr(TsPar:str,TsPerp:str,TrPar:str,TrPerp:str,massRatio:float):
    TsParNode = Node(TsPar)
    TsPerpNode = Node(TsPerp)
    TrParNode = Node(TrPar)
    TrPerpNode = Node(TrPerp)

    return (TrPerpNode + massRatio*TsPerpNode)*(TrParNode + massRatio*TsParNode)

def X(alphasr):
    return alphasr - 1

def psi(alphasr,X):
    return alphasr()**2*(K_LMN(X,"004") - K_LMN(X,"202")) + 0.5*alphasr()*(K_LMN(X,"200") - K_LMN(X,"002"))

def xi(X):
    return 4*K_LMN(X,"220") - 2*K_LMN(X,"202") - K_LMN(X,"200") + K_LMN(X,"002")

def cPar():
    return None

def cPerp():
    return None

def dPar():
    return None

def ePar():
    return None

def ePerp():
    return None

def kPar():
    return None

def kPerp():
    return None
        

# %%
