import numpy as np

import sys
sys.path.append('../')
import RMK_support.simple_containers as sc
import RMK_support.common_models as cm
from RMK_support import RKWrapper,Grid
import RMK_support.sk_normalization as skn

def esTestGenerator(**kwargs) -> RKWrapper:
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

    # ### Wrapper initialization

    rk = RKWrapper()

    # ### Global parameters for IO files

    rk.jsonFilepath = kwargs.get("jsonFilepath","./config.json")
    hdf5Filepath = kwargs.get("hdf5Filepath","./RMKOutput/RMK_ES_test/") 
    rk.setHDF5Path(hdf5Filepath)

    numProcsX = kwargs.get("mpiProcsX",4)
    numProcsH = kwargs.get("mpiProcsH",1)  # Number of processes in harmonic
    haloWidth = 1  # Halo width in cells

    rk.setMPIData(numProcsX, numProcsH, haloWidth)

    # ### Normalization setup
    ionZ = kwargs.get("ionZ",1)
    rk.setNormDensity(1.0e19) 
    rk.setNormTemperature(100.0)
    rk.setNormRefZ(ionZ)

    skNorms = skn.calculateNorms(rk.normalization["eVTemperature"],rk.normalization["density"],ionZ)

    # ### Grid setup

    xGrid = kwargs.get("dx",150)*np.ones(kwargs.get("Nx",64))
    dv0 = kwargs.get("dv0",0.0307)
    cv = kwargs.get("cv",1.025)
    vGrid = [dv0]
    for i in range(1,kwargs.get("Nv",120)):
        vGrid.append(vGrid[i-1]*cv)
    lMax = kwargs.get("lmax",1)
    gridObj = Grid(xGrid, np.array(vGrid), lMax, interpretXGridAsWidths=True, interpretVGridAsWidths=True, isPeriodic=True)
    L = sum(xGrid)

    # Add the grid to the wrapper
    rk.grid = gridObj


    # ### Set default species and temperature derivations

    rk.setStandardTextbookOptions([0])

    rk.addSpecies("e", 0)
    rk.addSpecies("D+", -1, atomicA=2.014, charge=1.0)

    # ### Braginskii flux derivation objects

    nConstGradT = 12*np.pi**1.5*epsilon0**2/np.sqrt(elMass*elCharge)  # Comes from e-i collision time

    sqrt2 = np.sqrt(2)

    delta = (1 + 65*sqrt2/32 + 433*sqrt2/288 - 23*sqrt2/16)*ionZ + (5629/1152 - 529/128) * ionZ**2  # A30 in Makarov assuming single ion species and 0 mass ratio
    elCondConst = 125*(1+433*sqrt2*ionZ/360)/(32*delta)

    lenNorm = skNorms["length"]
    qNorm = skNorms["heatFlux"]

    normalizationConst = rk.normalization["eVTemperature"]**3.5/(lenNorm*qNorm)
    kappaDeriv = sc.simpleDerivation(-elCondConst*nConstGradT*normalizationConst, [2.5,-1.0])
    rk.addCustomDerivation("kappa", kappaDeriv)

    qDeriv = sc.multiplicativeDerivation(innerDerivation="kappa", innerDerivationIndices=[1,2], outerDerivation="gradDeriv", outerDerivationIndices=[1])
    rk.addCustomDerivation("qT", qDeriv)
    rk.addCustomDerivation("ionDens",sc.simpleDerivation(1/ionZ,[1.0]))

    # ### Variables

    n = np.ones(gridObj.numX())
    T = 1.0 + 0.001*np.sin(2*np.pi*gridObj.xGrid/L)
    W = 3*n*T/2

    f = np.zeros([gridObj.numX(),gridObj.numH(),gridObj.numV()])
    for i in range(gridObj.numX()):
        f[i,gridObj.getH(0)-1,:] = np.pi**(-1.5) * T[i] ** (-1.5) * n[i] * np.exp(-gridObj.vGrid**2/T[i])

    # Rescale distribution function to ensure that the numerical density moment agrees with the initial values
    numerical_dens = gridObj.velocityMoment(f,0,1)
    for i in range(gridObj.numX()):
        f[i,gridObj.getH(0)-1,:] = n[i] *f[i,gridObj.getH(0)-1,:]/numerical_dens[i]

    rk.addVarAndDual("f",f,isDistribution=True,isCommunicated=True)
    rk.addVar("W",W,isDerived=True,derivationRule=sc.derivationRule("energyMoment",["f"]))
    rk.addVarAndDual("n",n,units='$10^{19} m^{-3}$',isDerived=True,derivationRule=sc.derivationRule("densityMoment",["f"]))
    rk.addVarAndDual("ni",n/ionZ,units='$10^{19} m^{-3}$',isDerived=True,derivationRule=sc.derivationRule("ionDens",["n"]))
    rk.addVar("zeroVar",isDerived=True,outputVar=False)
    rk.addVarAndDual("T",T,units='$100eV$',isDerived=True,derivationRule=sc.derivationRule("tempFromEnergye",["W","n","zeroVar"]),isCommunicated=True)
    rk.addVarAndDual("E",primaryOnDualGrid=True,isCommunicated=True)
    rk.addVarAndDual("q",isDerived=True,primaryOnDualGrid=True,derivationRule=sc.derivationRule("heatFluxMoment",["f"]))
    rk.addVarAndDual("G",isDerived=True,primaryOnDualGrid=True,derivationRule=sc.derivationRule("fluxMoment",["f"]))
    rk.addVarAndDual("qT",isDerived=True,derivationRule=sc.derivationRule("qT",["T","logLee"])) # Braginskii heatflux variable for comparison
    rk.addVar("logLee",13.63*np.ones(gridObj.numX()),isDerived=True) 
    rk.addVar("time",isScalar=True,isDerived=True)


    advModel = cm.kinAdvX(modelTag="adv", distFunName="f", gridObj=gridObj)
    rk.addModel(advModel.dict())

    cm.addExAdvectionModel(modelTag="E-adv", distFunName="f", eFieldName="E_dual", wrapper=rk, dualDistFun="f_dual")

    ampMaxModel = sc.CustomModel(modelTag="AM")

    eTerm = cm.ampereMaxwellKineticElTerm("f", "E_dual")

    ampMaxModel.addTerm("eTerm", eTerm)

    rk.addModel(ampMaxModel.dict())

    cm.addEECollIsotropic(modelTag="e-e0", distFunName="f", elTempVar="T", elDensVar="n", wrapper=rk)

    cm.addStationaryIonEIColl(modelTag="e-i_odd",
                            distFunName="f",
                            ionDensVar="ni_dual",
                            electronDensVar="n_dual",
                            electronTempVar="T_dual",
                            ionSpeciesName="D+",
                            evolvedHarmonics=list(range(2, gridObj.numH()+1, 2)),
                            wrapper=rk)

    cm.addEECollHigherL(modelTag="e-e_odd",
                        distFunName="f",
                        elTempVar="T_dual",
                        elDensVar="n_dual",
                        wrapper=rk,
                        evolvedHarmonics=list(range(2, gridObj.numH()+1, 2)),
                        dualDistFun="f_dual")
    
    if lMax > 1:
        cm.addStationaryIonEIColl(modelTag="e-i_even",
                            distFunName="f",
                            ionDensVar="ni",
                            electronDensVar="n",
                            electronTempVar="T",
                            ionSpeciesName="D+",
                            evolvedHarmonics=list(range(3, gridObj.numH()+1, 2)),
                            wrapper=rk)

        cm.addEECollHigherL(modelTag="e-e_even",
                        distFunName="f",
                        elTempVar="T",
                        elDensVar="n",
                        wrapper=rk,
                        evolvedHarmonics=list(range(3, gridObj.numH()+1, 2)))

    

    # Extractor manipulator to retrieve the e-e Coulomb log from the e-e0 model. Priority set to 1 to ensure that qB is derived from the latest logLee value.

    rk.addManipulator("logLeeExtractor",sc.extractorManipulator("e-e0","logLee","logLee",priority=1))

    # ### Integrator and timestep options
    
    integrator = sc.picardBDEIntegrator(absTol=10.0, convergenceVars=["f"])

    rk.addIntegrator("BE", integrator)

    # Set initial timestep length and numbers of allowed implicit and general groups

    initialTimestep = kwargs.get("initialTimestep",0.1)

    rk.setIntegratorGlobalData(1, 1, initialTimestep)

    bdeStep = sc.IntegrationStep("BE")

    for tag in rk.modelTags():
        bdeStep.addModel(tag)

    rk.addIntegrationStep("BE1", bdeStep.dict())

    if kwargs.get("includeDiagnosticVars",False):

        rk.addTermDiagnosisForVars(["f"])

    Nt = kwargs.get("Nt",300)
    rk.setFixedNumTimesteps(Nt)
    rk.setFixedStepOutput(Nt/30)

    rk.setPETScOptions(cliOpts="-pc_type bjacobi -sub_pc_factor_shift_type nonzero",kspSolverType="gmres")
    
    return rk
