#%%
import sys

sys.path.append('../')
from RMK_support import Node, atan

def phi(X:Node) -> Node:
    return atan(X**0.5)*X**-0.5
def K_LMN(X:Node,LMN:str) -> Node:
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
            raise ValueError("Unknown K_LMN case")

def alphasr(TsPar:str,TsPerp:str,TrPar:str,TrPerp:str,massRatio:float) -> Node:
    TsParNode = Node(TsPar)
    TsPerpNode = Node(TsPerp)
    TrParNode = Node(TrPar)
    TrPerpNode = Node(TrPerp)

    return (TrPerpNode + massRatio*TsPerpNode)*(TrParNode + massRatio*TsParNode)

def X(alphasr:Node) -> Node:
    return alphasr - 1

def psi(alphasr:Node,X:Node) -> Node:
    return alphasr()**2*(K_LMN(X,"004") - K_LMN(X,"202")) + 0.5*alphasr()*(K_LMN(X,"200") - K_LMN(X,"002"))

def xi(X:Node) -> Node:
    return 4*K_LMN(X,"220") - 2*K_LMN(X,"202") - K_LMN(X,"200") + K_LMN(X,"002")

def cPerp(nuss:Node,nusr:Node,alphass:Node,X:Node) -> Node:
    return -4*nuss*(6*alphass*K_LMN(X,"202") + 0.5*xi(X)) + 4*alphass*nusr*(-32*K_LMN(X,"222") + 4*K_LMN(X,"204") + 10*K_LMN(X,"202") - 2*K_LMN(X,"004") - K_LMN(X,"002"))

def cPar(nuss:Node,nusr:Node,alphass:Node,X:Node) -> Node:
    return 2*nuss*psi(alphass,X) + 4*alphass**2*nusr*(-8*3**-1*alphass*K_LMN(X,"204") + 2*3**-1*alphass*K_LMN(X,"006") + 4*K_LMN(X,"202") - (1 - alphass*3**-1)*K_LMN(X,"004") - 0.5*K_LMN(X,"002"))

def ePerp(nuss:Node,nusr:Node,alphass:Node,X:Node) -> Node:
    return 12*nuss*xi(X) + 12*nusr*(16*alphass*K_LMN(X,"222") - 4*alphass*K_LMN(X,"204") - 2*(2*alphass - 1)*K_LMN(X,"202") + 2*alphass*K_LMN(X,"004") - K_LMN(X,"002"))

def ePar(nuss:Node,nusr:Node,alphass:Node,X:Node) -> Node:
    return -12*nuss*psi(alphass,X) + 4*nusr*alphass*(4*alphass**2*K_LMN(X,"204") - 2*alphass**2*K_LMN(X,"006") - 6*alphass*K_LMN(X,"202") + 4*alphass*K_LMN(X,"004") - 1.5*K_LMN(X,"002"))
        

# %%
