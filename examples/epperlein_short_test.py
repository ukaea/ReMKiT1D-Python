import numpy as np

import RMK_support as rmk
import RMK_support.common_models as cm
from RMK_support import node
from RMK_support.derivations import SimpleDerivation,DerivationClosure,NodeDerivation

def esTestGenerator(**kwargs) -> rmk.RMKContext:
    """Generator for a standard Epperlein-Short test run. Initializes the electrons with a periodic perturbed temperature profile
    T = T0 + dT*sin(2pi*x/L), where T0 is 100eV and dT is 0.1eV and L is the domain length (in normalized e-i collision mfps). 

    Ions are assumed stationary, and the electron density is set to n0=10^19m^{-3}, with ni=n0/ionZ. 

    The following kwargs are available:

    jsonFilepath (str): Config filepath. Defaults to ".config.json".
    hdf5Filepath (str): Input and output hdf5 folder path. Defaults to "./RMKOutput/RMK_ES_test/".
    mpiProcsX (int): Number of MPI processes in the spatial direction. Defaults to 4.
    mpiProcsH (int): Number of MPI processes in the harmonic direction. Defaults to 1. 
    ionZ (float): Reference ion charge. Defaults to 1. 
    dx (float): Spatial cell width in units of normalized e-i mfp. Defaults to 150.
    Nx (int): Number of spatial cells. Defaults to 64.
    dv0 (float): Smallest velocity grid width in electron thermal velocity units (sqrt(2kTe/me)). Defaults to 0.0307.
    cv (float): Velocity grid width multiplier such that dv_n = cv*dv_{n-1}. Defaults to 1.025.
    Nv (int): Number of velocity cells. Defaults to 120.
    lmax (int): Highest resolved l-harmonic. Defaults to 1.
    initialTimestep (float): Used timestep in e-i collision time units. Defaults to 0.1.
    Nt (int): Number of timesteps - must be divisible by 30. Defaults to 300.
    includeDiagnosticVars (bool): If true will include all distribution evaluation terms as communicated variables. Defaults to False.

    Returns:
        RKWrapper: ReMKiT1D wrapper containing initialization data for this run.
    """
    
    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  # vacuum permittivity

    rk = rmk.RMKContext()
    rk.IOContext = rmk.IOContext(kwargs.get("jsonFilepath","./config.json"),
                                 kwargs.get("hdf5Filepath","./RMKOutput/RMK_ES_test/"))

    rk.mpiContext = rmk.MPIContext(kwargs.get("mpiProcsX",4),
                                   kwargs.get("mpiProcsH",1))

    ionZ = kwargs.get("ionZ",1)
    rk.normZ = ionZ #Setting normalisation for ion charge
    rk.normTemperature = 100.00 
    # ### Grid setup

    xGrid = kwargs.get("dx",150)*np.ones(kwargs.get("Nx",64))
    dv0 = kwargs.get("dv0",0.0307)
    cv = kwargs.get("cv",1.025)
    vGrid = [dv0]
    for i in range(1,kwargs.get("Nv",120)):
        vGrid.append(vGrid[i-1]*cv)
    lMax = kwargs.get("lmax",1)
    rk.grid = rmk.Grid(xGrid, np.array(vGrid), lMax, interpretXGridAsWidths=True, interpretVGridAsWidths=True, isPeriodic=True)
    L = sum(xGrid)

    # ### Set default species and temperature derivations

    rk.textbook = rmk.Textbook(rk.grid,[0]) 

    rk.species.add(rmk.Species("e",0))
    rk.species.add(rmk.Species("D+",-1,atomicA=2.014,charge=1.0))

    # ### Variables

    nInit = np.ones(rk.grid.numX)

    TInit = 1.0 + 0.001*np.sin(2*np.pi*rk.grid.xGrid/L)
    WInit = 3*nInit*TInit/2
    fInit = np.zeros([rk.grid.numX,rk.grid.numH,rk.grid.numV])
    for i in range(rk.grid.numX):
        fInit[i,rk.grid.getH(0)-1,:] = (np.pi*TInit[i])**(-1.5) * nInit[i]* np.exp(-rk.grid.vGrid**2/TInit[i])

    # Rescale distribution function to ensure that the numerical density moment agrees with the initial values
    numerical_dens = rk.grid.velocityMoment(fInit,0,1)
    for i in range(rk.grid.numX):
        fInit[i,rk.grid.getH(0)-1,:] = nInit[i] *fInit[i,rk.grid.getH(0)-1,:]/numerical_dens[i]
        
    f,f_dual = rmk.varAndDual("f",rk.grid,isDistribution=True,data=fInit)
    W = rmk.Variable("W",rk.grid,data=WInit,derivation=rk.textbook["energyMoment"],derivationArgs=["f"])
    n,n_dual = rmk.varAndDual("n",rk.grid,derivation=rk.textbook["densityMoment"],derivationArgs=["f"])
    ni,ni_dual = rmk.varAndDual("ni",rk.grid,data=nInit,derivation=NodeDerivation("ni",node =node(n)/ionZ))

    T,T_dual = rmk.varAndDual("T",rk.grid,derivation=rk.textbook["tempFromEnergye"],derivationArgs=["W","n","zeroVar"])
    zeroVar = rmk.Variable("zeroVar",rk.grid,isDerived=True,inOutput=False)

    E_dual,E = rmk.varAndDual("E",rk.grid,primaryOnDualGrid=True)
    q_dual,q = rmk.varAndDual("q",rk.grid,primaryOnDualGrid=True,derivation=rk.textbook["heatFluxMoment"],derivationArgs=["f"])

    logLee = rmk.Variable("logLee",rk.grid,isDerived=True)
    rk.variables.add(f,f_dual,W,n,n_dual,ni,ni_dual,T,T_dual,zeroVar,E_dual,E,q_dual,q,logLee)
    # ### Braginskii flux derivation objects

    nConstGradT = 12*np.pi**1.5*epsilon0**2/np.sqrt(elMass*elCharge)  # Comes from e-i collision time

    sqrt2 = np.sqrt(2)

    delta = (1 + 65*sqrt2/32 + 433*sqrt2/288 - 23*sqrt2/16)*ionZ + (5629/1152 - 529/128) * ionZ**2  # A30 in Makarov assuming single ion species and 0 mass ratio
    elCondConst = 125*(1+433*sqrt2*ionZ/360)/(32*delta)

    lenNorm = rk.norms["length"]
    qNorm = rk.norms["heatFlux"]

    normalizationConst = rk.normTemperature**3.5/(lenNorm*qNorm)
    kappaClosure = DerivationClosure(SimpleDerivation("kappa",-elCondConst*nConstGradT*normalizationConst,[2.5,-1.0]),T,logLee)
    gradTClosure = DerivationClosure(rk.textbook["gradDeriv"],T)
    qt = (kappaClosure*gradTClosure)(T).rename("qT")
    rk.variables.add(qt)

    rk.models.add(cm.kinAdvX(f,rk.grid).rename("adv"))

    rk.models.add(cm.advectionEx(f,E_dual,rk.grid,rk.norms).rename("E-adv"))

    amModel = rmk.Model("AM")
    amModel.ddt[E_dual] += cm.ampereMaxwellKineticElTerm(f,rk.norms)
    rk.models.add(amModel)

    rk.models.add(cm.eeCollIsotropic(f,T,n,rk.norms,rk.grid,rk.textbook).rename("e-e_0"))

    rk.models.add(cm.stationaryIonEIColl(rk.grid,rk.textbook,rk.norms,f,ni_dual,n_dual,T_dual,rk.species["D+"],evolvedHarmonics=list(range(2, rk.grid.numH+1, 2))).rename("e-i_odd"))

    rk.models.add(cm.eeCollHigherL(rk.grid,rk.textbook,rk.norms,f,T_dual,n_dual,list(range(2, rk.grid.numH+1, 2))).rename("e-e_odd"))
    
    if lMax > 1:
        rk.models.add(cm.stationaryIonEIColl(rk.grid,rk.textbook,rk.norms,f,ni,n,T,rk.species["D+"],evolvedHarmonics=list(range(3, rk.grid.numH+1, 2))).rename("e-i_even"))

        rk.models.add(cm.eeCollHigherL(rk.grid,rk.textbook,rk.norms,f,T,n,list(range(3, rk.grid.numH+1, 2))).rename("e-e_even"))

    # Extractor manipulator to retrieve the e-e Coulomb log from the e-e0 model. Priority set to 1 to ensure that qB is derived from the latest logLee value.

    rk.manipulators.add(rmk.MBDataExtractor("logLee",rk.models["e-e_0"],rk.models["e-e_0"].mbData["logLee"],priority=1))

    #Integration setup
    integrator = rmk.BDEIntegrator("BDE",absTol=10.0,convergenceVars=[f])
    integrationStep = rmk.IntegrationStep("BE",integrator)
    integrationStep.add(rk.models) 
    rk.integrationScheme = rmk.IntegrationScheme(dt=kwargs.get("initialTimestep",0.1),steps=integrationStep) 
    Nt = kwargs.get("Nt",300)
    rk.integrationScheme.setFixedNumTimesteps(Nt,Nt/30) 

    if kwargs.get("includeDiagnosticVars",False):

        rk.addTermDiagnostics(f)

    rk.setPETScOptions(cliOpts="-pc_type bjacobi -sub_pc_factor_shift_type nonzero",kspSolverType="gmres")
    
    return rk
