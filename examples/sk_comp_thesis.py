
import numpy as np
import xarray as xr

import sys
sys.path.append('../')
import RMK_support.simple_containers as sc
import RMK_support.common_models as cm
import RMK_support.crm_support as crm
from RMK_support import RKWrapper,Grid
import RMK_support.amjuel_support as ams
import RMK_support.sk_normalization as skn

def generatorSKThesis(**kwargs) -> RKWrapper:
    """Generate a wrapper containing a SOL-KiT run similar to the Mijin thesis fluid version. Neutrals are diffusive, the heating profile is a step function upstream, recycling is at 100%, the neutrals have a fixed temperature (with a spurious sqrt(2) factor to match SOL-KiT's diffusion) and Ti=Te. The CRM can either use a SOL-KiT-like Janev cross-section calculation or AMJUEL rates, and similarly for CX. Both the spatial and velocity grid widths are geometric. If not initializing from a previous run, the initial profiles are based on the 2-Point Model, with 100% ionization fraction.

    The following kwargs are available:

    jsonFilepath (str): Config filepath. Defaults to ".config.json".
    hdf5Filepath (str): Input and output hdf5 folder path. Defaults to "./RMKOutput/RMK_SK_comp_staggered_thesis/".
    mpiProcs (int): Number of MPI processes in the spatial direction. Defaults to 8.
    dx0 (float): Largest spatial cell width in m. Defaults to 0.27.
    dxN (float): Smallest spatial cell width in m. Defaults to 0.0125.
    Nx (int): Number of spatial cells. Defaults to 128.
    dv0 (float): Smallest velocity grid width in electron thermal velocity units (sqrt(2kTe/me)). Defaults to 0.0307.
    dvN (float): Largest velocity grid width. Defaults to 0.4.
    Nv (int): Number of velocity cells. Defaults to 80.
    lmax (int): Highest resolved l-harmonic. Defaults to 1.
    Tn (float): Neutral temperature for the diffusion operator (in eV). Defaults to 3eV
    loglinExtrap (bool): Set to true if the extrapolation derivations at the boundary should be set to mimic SOL-KiT's log-linear   
                         extrapolation instead of the default linear extrapolation. Defaults to False.
    numNeutrals (int): Number of neutral states tracked. Should be set to 1 if using AMJUEL rates. Defaults to 1.
    Tu (float): Initial upstream temperature in eV. Defaults to 20.
    Td (float): Initial downstream temperature in eV. Defaults to 5.
    nu (float): Initial upstream density in normalized units. Automatically determines the downstream density using a constant pressure 
                assumption. Defaults to 0.8. 
    useKineticCorrections (bool): If true will attempt to use kinetic corrections for the heat flux and sheath energy transmission 
                                  coefficient from a kinetic simulation as well as restart from a given (kinetic) run. Defaults to False.
    fixedKineticGamma (float): If present will be used instead of the loaded sheath energy transmission coefficient from the kinetic 
                               simulation. Defaults to 5.0 (but is overwritten unless specifically supplied)
    fixedQRatio (np.array): Spatial vector representing the ratio of the kinetic conductive heat flux to the Braginskii value. If
                            present will be used instead of the the loaded ratio from the kinetic simulation (useful if the kinetic simulation has negative ratios which could lead to instability). Defaults to all ones (but is overwritten unless specifically supplied)
    hdf5InputFile (str): Path of hdf5 input file (without the extension!). Defaults to "ReMKiT1DVarInput".
    Nh (int): Number of upstream cells affected by the heating operator. Defaults to 17.
    heatingPower (float): Effective upstream heating in MW/m^2. Defaults to 1.0.
    perturbationTimeSignal (sc.TimeSignal): Optional time signal for a periodic heating perturbation. Defaults to no perturbation.
    pertHeatingPower (float): The heating perturbation amplitude in MW/m^2. Defaults to 10. 
    amjuelCXRate (bool): If true uses the AMJUEL CX rate H.2 3.1.8 instead of the SOL-KiT model. This rate is then divided by 
                         sqrt(2kTe/mi) to get the cross-section for neutral diffusion. Defaults to False. 
    amjuelRates (bool): If true uses AMJUEL ionization and recombination particle and energy rates instead of the SOL-KiT atomic state 
                        resolved CRM. Defaults to False. 
    includedJanevTransitions: Set of included transitions in the SOL-KiT style CRM. The full list is ["ex","deex","ion","recomb3b"].
                              Defaults to ["ion"].
    includeSpontEmission: If true will include spontaneous emission in the SOL-KiT style CRM for up to n=20. Defaults to False.
    initialTimestep: Used timestep in shortest e-i collision times in the system. Defaults to 0.1.
    amjuelPath (str): Path to amjuel.tex file. Defaults to "../data/amjuel.tex".

    Returns:
        RKWrapper: Wrapper containing run information
    """
    
    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    amu = 1.6605390666e-27 #atomic mass unit
    ionMass = 2.014*amu # deuterium mass
    epsilon0 = 8.854188e-12 #vacuum permittivity 
    heavySpeciesMass = 2.014 #in amus

    rk = RKWrapper()

    rk.jsonFilepath = kwargs.get("jsonFilepath","./config.json")
    hdf5Filepath = kwargs.get("hdf5Filepath","./RMKOutput/RMK_SK_comp_staggered_thesis/") 
    rk.setHDF5Path(hdf5Filepath)

    
    numProcsX = kwargs.get("mpiProcs",8)
    numProcsH = 1 # Number of processes in harmonic 
    numProcs = numProcsX * numProcsH
    haloWidth = 1 # Halo width in cells

    rk.setMPIData(numProcsX,numProcsH,haloWidth)

    rk.setNormDensity(1.0e19)
    rk.setNormTemperature(10.0)
    rk.setNormRefZ(1.0)

    tempNorm = rk.normalization["eVTemperature"] 
    densNorm = rk.normalization["density"]
    skNorms = skn.calculateNorms(tempNorm,densNorm,1)
    
    timeNorm = skNorms["time"]
    lengthNorm = skNorms["length"]
    sigmaNorm = skNorms["crossSection"]

    dx0 = kwargs.get("dx0",0.27)
    dxN = kwargs.get("dxN",0.0125)
    Nx = kwargs.get("Nx",128) 
    xGridWidths = np.geomspace(dx0,dxN,Nx)
    L = sum(xGridWidths)
    dv0 = kwargs.get("dv0",0.05)
    dvN = kwargs.get("dvN",0.4) 
    Nv = kwargs.get("Nv",80)  
    vGridWidths = np.geomspace(dv0,dvN,Nv)
    lMax = kwargs.get("lmax",0)
    gridObj = Grid(xGridWidths,vGridWidths,lMax,interpretXGridAsWidths=True,interpretVGridAsWidths=True,isLengthInMeters=True)

    rk.grid = gridObj

    # Diffusion coefficient derivation in 1D with neutral temperature Tn and with the cross section used being the low energy charge-exchange cross-seciton
    # NOTE: SOL-KiT has a spurious sqrt(2) factor in the diffusion coefficient, so that is kept here for a consistent comparison
    Tn = kwargs.get("Tn",3.0)/tempNorm

    diffusionDeriv = sc.simpleDerivation(np.sqrt(Tn)/2,[-1.0])

    rk.addCustomDerivation("neutDiffD",diffusionDeriv)

    rk.addCustomDerivation("identityDeriv",sc.simpleDerivation(1.0,[1.0]))
    rk.addCustomDerivation("sqDivide",sc.simpleDerivation(1.0,[2.0,-1.0]))
    absDeriv = sc.multiplicativeDerivation("identityDeriv",[1],funcName="abs")
    rk.addCustomDerivation("absDeriv",absDeriv)
    rk.addCustomDerivation("square",sc.simpleDerivation(1.0,[2.0]))

    rk.addCustomDerivation("linExtrapRight",sc.boundedExtrapolationDerivation(sc.linExtrapolation(),ignoreUpperBound=True))

    loglinExtrap = kwargs.get("loglinExtrap",False)
    if loglinExtrap:
        rk.addCustomDerivation("logLinExtrapRight",sc.boundedExtrapolationDerivation(sc.linLogExtrapolation(),ignoreUpperBound=True))
    
        rk.addCustomDerivation("lastCellExt",sc.locValExtractorDerivation(len(xGridWidths)))

        rk.addCustomDerivation("secondToLastCellExt",sc.locValExtractorDerivation(len(xGridWidths)-1))

        rk.addCustomDerivation("uLastCell",sc.additiveDerivation(["lastCellExt","secondToLastCellExt"],1.0,[[1],[2]],[0.5,0.5]))
        dxNNorm = dxN/lengthNorm
        dxNStagNorm = dxNNorm + xGridWidths[-2]/(2*lengthNorm)

    else:
        
        rk.addCustomDerivation("linExtrapRightLB",sc.boundedExtrapolationDerivation(sc.linExtrapolation(),expectLowerBoundVar=True,ignoreUpperBound=True))

        rk.addCustomDerivation("boundaryFlux",sc.multiplicativeDerivation("linExtrapRight",innerDerivationIndices=[1],outerDerivation="linExtrapRightLB",outerDerivationIndices=[2,3]))

    rk.addSpecies("e",0,atomicA=elMass/amu,charge=-1.0,associatedVars=["ne","Ge","We"]) 
    rk.addSpecies("D+",-1,atomicA=2.014,charge=1.0,associatedVars=["ni","Gi"])

    # Set neutrals 
    numNeutrals=kwargs.get("numNeutrals",1)
    neutralDensList = ["n"+str(i) for i in range(1,numNeutrals+1)] # List of neutral density names

    for neutral in neutralDensList:
        rk.addSpecies("D"+neutral[1:],int(neutral[1:]),heavySpeciesMass,associatedVars=[neutral])

    electronSpecies = rk.getSpecies("e")
    ionSpecies = rk.getSpecies("D+")
    
    # Two-point model initialization 

    Tu = kwargs.get("Tu",20)/tempNorm #upstream temperature
    Td = kwargs.get("Td",5)/tempNorm #downstream temperature

    T = (Tu**(7/2) - (Tu**(7/2)-Td**(7/2))*gridObj.xGrid/L)**(2/7)

    nu = kwargs.get("nu",0.8) #upstream density

    n = nu*Tu/T 

    W = 3*n*T/2
    # Set conserved variables in container

    rk.addVarAndDual("ne",n,units='$10^{19} m^{-3}$',isCommunicated=True) #Units are not used by ReMKiT1D, but are useful to specify for later plotting
    rk.addVarAndDual("ni",n,units='$10^{19} m^{-3}$',isCommunicated=True)
    rk.addVarAndDual("Ge",primaryOnDualGrid=True,isCommunicated=True) # Ge_dual is evolved, and Ge is derived
    rk.addVarAndDual("Gi",primaryOnDualGrid=True,isCommunicated=True)
    rk.addVarAndDual("We",W,units='$10^{20} eV m^{-3}$',isCommunicated=True)

    # Temperatures
    rk.addVarAndDual("Te",T,isStationary=True,units='$10eV$',isCommunicated=True)

    # Set heat fluxes 

    rk.addVarAndDual("qe",isStationary=True,primaryOnDualGrid=True,isCommunicated=True)

    # Set E field

    rk.addVarAndDual("E",primaryOnDualGrid=True)

    # Set derived fluid quantities

    rk.addVarAndDual("ue",isDerived=True,primaryOnDualGrid=True,derivationRule=sc.derivationRule("flowSpeedFromFlux",["Ge_dual","ne_dual"]),isCommunicated=True)
    rk.addVarAndDual("ui",isDerived=True,primaryOnDualGrid=True,derivationRule=sc.derivationRule("flowSpeedFromFlux",["Gi_dual","ni_dual"]),isCommunicated=True)
    rk.addVar("cs",isDerived=True,derivationRule=sc.derivationRule("sonicSpeedD+",["Te","Te"]))

    if loglinExtrap:

        rk.addVar("cs_lc",isDerived=True,isScalar=True,derivationRule=sc.derivationRule("lastCellExt",["cs"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

        rk.addVar("nb",isDerived=True,isScalar=True,derivationRule=sc.derivationRule("logLinExtrapRight",["ni"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

        rk.addVar("ui_lc",isDerived=True,isScalar=True,derivationRule=sc.derivationRule("uLastCell",["cs","ui_dual"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

        rk.addVar("ue_lc",isDerived=True,isScalar=True,derivationRule=sc.derivationRule("uLastCell",["cs","ue_dual"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

    else:
        rk.addVar("cs_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
                ,derivationRule=sc.derivationRule("linExtrapRight",["cs"]))

        rk.addVar("n_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
                ,derivationRule=sc.derivationRule("linExtrapRight",["ne"]))

        rk.addVar("G_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
                ,derivationRule=sc.derivationRule("boundaryFlux",["ni","ui","cs_b"]))

        rk.addVar("u_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
                ,derivationRule=sc.derivationRule("flowSpeedFromFlux",["G_b","n_b"]))

    # Set scalar quantities 
    rk.addVar("time",isScalar=True,isDerived=True)

    if kwargs.get("useKineticCorrections",False):
        rk.addVar("gammaRight",kwargs.get("fixedKineticGamma",5.0)*np.ones(1),isScalar=True,isDerived=True)
        rk.addVar("qRatio",kwargs.get("fixedQRatio",np.ones(Nx)),isOnDualGrid=True,isDerived=True)
        loadVars=["ne","ni","Ge_dual","Gi_dual","We","Te","qe","E_dual"] + neutralDensList
        if "fixedQRatio" not in kwargs:
            loadVars.append("qRatio")
        if "fixedKineticGamma" not in kwargs:
            loadVars.append("gammaRight")
        rk.setHDF5FileInitialData(loadVars,filename=kwargs.get("hdf5InputFile","ReMKiT1DVarInput"))
            
    else:
        rk.addVar("gammaRight",isScalar=True,isDerived=True,derivationRule=sc.derivationRule("rightElectronGamma",["Te","Te"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

    # Set neutral densities

    for neut in neutralDensList:
            rk.addVarAndDual(neut,units='$10^{19} m^{-3}$',isCommunicated=True)

    # We need a distribution function to calculate rates from cross-sections built into the code
    f = np.zeros([gridObj.numX(),gridObj.numH(),gridObj.numV()])
    for i in range(gridObj.numX()):
        f[i,gridObj.getH(0)-1,:] = np.pi**(-1.5) * T[i] ** (-1.5) * n[i] * np.exp(-gridObj.vGrid**2/T[i])
    rk.addVar("f_unscaled",f,isDerived=True,isDistribution=True,derivationRule=sc.derivationRule("maxwellianDistribution",["Te","ne"]))
    rk.addVar("ne_numerical",n,isDerived=True,derivationRule=sc.derivationRule("densityMoment",["f_unscaled"]),units='$10^{19} m^{-3}$',isCommunicated=True) 
    rk.addVar("ne_rescaled",isDerived=True,derivationRule=sc.derivationRule("sqDivide",["ne","ne_numerical"]))
    rk.addVar("f",f,isDerived=True,isDistribution=True,derivationRule=sc.derivationRule("maxwellianDistribution",["Te","ne_rescaled"]))

    #Electron continuity advection

    #Adding the model tag to tag list
    modelTag = "continuity-ne"
    
    #Initializing model using common models 

    if loglinExtrap:
        electronContModel = cm.staggeredAdvection(modelTag=modelTag
                                           ,advectedVar="ne"
                                           ,fluxVar="Ge_dual")

        normBC = sc.CustomNormConst(multConst=-1/dxNNorm)
        vDataBC = sc.VarData(reqRowVars=["cs_lc","nb","ne"],reqRowPowers=[1.0,1.0,-1.0])
        evolvedVar = "ne"
        outflowTerm = sc.GeneralMatrixTerm(evolvedVar,customNormConst=normBC,varData=vDataBC,stencilData=sc.diagonalStencil(evolvedXCells=[len(xGridWidths)]),implicitGroups=[2])

        electronContModel.addTerm("bcTerm",outflowTerm)
    else:
        electronContModel = cm.staggeredAdvection(modelTag=modelTag, advectedVar="ne",
                                                fluxVar="Ge_dual", advectionSpeed="ue", lowerBoundVar="cs", rightOutflow=True)

    rk.addModel(electronContModel.dict())

    #Ion continuity advection

    #Adding the model tag to tag list
    modelTag = "continuity-ni"

    #Initializing model using common models
    if loglinExtrap:
        ionContModel = cm.staggeredAdvection(modelTag=modelTag
                                                ,advectedVar="ni"
                                                ,fluxVar="Gi_dual")

        normBC = sc.CustomNormConst(multConst=-1/dxNNorm)
        vDataBC = sc.VarData(reqRowVars=["cs_lc","nb","ni"],reqRowPowers=[1.0,1.0,-1.0])
        evolvedVar = "ni"
        outflowTerm = sc.GeneralMatrixTerm(evolvedVar,customNormConst=normBC,varData=vDataBC,stencilData=sc.diagonalStencil(evolvedXCells=[len(xGridWidths)]))
        ionContModel.addTerm("bcTerm",outflowTerm)

    else:
        ionContModel = cm.staggeredAdvection(modelTag=modelTag, advectedVar="ni",
                                        fluxVar="Gi_dual", advectionSpeed="ui", lowerBoundVar="cs", rightOutflow=True)

    rk.addModel(ionContModel.dict())

    #Electron pressure grad

    #Adding the model tag to tag list
    modelTag = "pressureGrad-Ge"

    #Initializing model
    electronPressureGradModel = cm.staggeredPressureGrad(modelTag=modelTag,fluxVar="Ge_dual",densityVar="ne",temperatureVar="Te",speciesMass=elMass)

    rk.addModel(electronPressureGradModel.dict())

    #Ion pressure grad

    #Adding the model tag to tag list
    modelTag = "pressureGrad-Gi"

    #Initializing model
    ionPressureGradModel = cm.staggeredPressureGrad(modelTag=modelTag,fluxVar="Gi_dual",densityVar="ni",temperatureVar="Te",speciesMass=ionMass)

    rk.addModel(ionPressureGradModel.dict())

    #Electron momentum advection

    #Adding the model tag to tag list
    modelTag = "advection-Ge"

    #Initializing model

    if loglinExtrap:
        electronMomAdvModel = cm.staggeredAdvection(modelTag=modelTag
                                        ,advectedVar="Ge_dual"
                                        ,fluxVar=""
                                        ,advectionSpeed="ue"
                                        ,staggeredAdvectionSpeed="ue_dual"
                                        ,lowerBoundVar="cs"
                                        ,rightOutflow=False,
                                        staggeredAdvectedVar=True)

        normBC = sc.CustomNormConst(multConst=-1/dxNStagNorm)
        vDataBC = sc.VarData(reqRowVars=["cs_lc","nb"],reqRowPowers=[2.0,1.0],reqColVars=["ne"],reqColPowers=[-1.0])
        evolvedVar = "Ge_dual"
        implicitVar = "ne"
        outflowTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,customNormConst=normBC,varData=vDataBC,stencilData=sc.diagonalStencil(evolvedXCells=[len(xGridWidths)-1]))

        electronMomAdvModel.addTerm("bcTerm",outflowTerm)
    else:
        electronMomAdvModel = cm.staggeredAdvection(modelTag=modelTag
                                                ,advectedVar="Ge_dual"
                                                ,fluxVar=""
                                                ,advectionSpeed="ue"
                                                ,staggeredAdvectionSpeed="ue_dual"
                                                ,lowerBoundVar="cs"
                                                ,rightOutflow=True,
                                                staggeredAdvectedVar=True)

    rk.addModel(electronMomAdvModel.dict())

    #Ion momentum advection

    #Adding the model tag to tag list
    modelTag = "advection-Gi"

    #Initializing model

    if loglinExtrap:
        ionMomAdvModel = cm.staggeredAdvection(modelTag=modelTag
                                                ,advectedVar="Gi_dual"
                                                ,fluxVar=""
                                                ,advectionSpeed="ui"
                                                ,staggeredAdvectionSpeed="ui_dual"
                                                ,lowerBoundVar="cs"
                                                ,rightOutflow=False,
                                                staggeredAdvectedVar=True)

        normBC = sc.CustomNormConst(multConst=-1/dxNStagNorm)
        vDataBC = sc.VarData(reqRowVars=["cs_lc","nb"],reqRowPowers=[2.0,1.0],reqColVars=["ni"],reqColPowers=[-1.0])
        evolvedVar = "Gi_dual"
        implicitVar = "ni"
        outflowTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,customNormConst=normBC,varData=vDataBC,stencilData=sc.diagonalStencil(evolvedXCells=[len(xGridWidths)-1]))

        ionMomAdvModel.addTerm("bcTerm",outflowTerm)
    else:
        ionMomAdvModel = cm.staggeredAdvection(modelTag=modelTag
                                            ,advectedVar="Gi_dual"
                                            ,fluxVar=""
                                            ,advectionSpeed="ui"
                                            ,staggeredAdvectionSpeed="ui_dual"
                                            ,lowerBoundVar="cs"
                                            ,rightOutflow=True,
                                            staggeredAdvectedVar=True)

    rk.addModel(ionMomAdvModel.dict())

    #Ampere-Maxwell E field equation 
    
    #Adding the model tag to tag list
    modelTag = "ampereMaxwell"

    #Initializing model
    ampereMawellModel = cm.ampereMaxwell(modelTag=modelTag,
                                        eFieldName="E_dual",
                                        speciesFluxes=["Ge_dual","Gi_dual"],
                                        species=[electronSpecies,ionSpecies])

    rk.addModel(ampereMawellModel.dict())

    #Lorentz force terms 
    
    #Adding the model tag to tag list
    modelTag = "lorentzForce"

    #Initializing model
    lorentzForceModel = cm.lorentzForces(modelTag=modelTag,
                                        eFieldName="E_dual",
                                        speciesFluxes=["Ge_dual","Gi_dual"],
                                        speciesDensities=["ne_dual","ni_dual"],
                                        species=[electronSpecies,ionSpecies])

    rk.addModel(lorentzForceModel.dict())

    # Implicit temperature equations

    #Adding the model tag to tag list
    modelTag = "implicitTemp"

    #Initializing model
    implicitTempModel = cm.implicitTemperatures(modelTag=modelTag,
                                                speciesFluxes=["Ge_dual"],
                                                speciesDensities=["ne"],
                                                speciesEnergies=["We"],
                                                speciesTemperatures=["Te"],
                                                species=[electronSpecies],
                                                speciesDensitiesDual=["ne_dual"]
                                                )
    rk.addModel(implicitTempModel.dict())

    #Electron energy advection

    #Adding the model tag to tag list
    modelTag = "advection-We"

    vData = sc.VarData(reqColVars=["We_dual","ne_dual"],reqColPowers=[1.0,-1.0])

    electronWAdvection = cm.staggeredAdvection(modelTag=modelTag
                                            ,advectedVar="We"
                                            ,fluxVar="Ge_dual",
                                            vData=vData)

    # No boundary terms means reflective boundaries => allows all outflow to be governed by sheath heat transmission coefficients 
    rk.addModel(electronWAdvection.dict())

    #Electron pressure advection

    #Adding the model tag to tag list
    modelTag = "advection-pe"

    vData = sc.VarData(reqColVars=["Te_dual"])

    #Initializing model
    electronPAdvection = cm.staggeredAdvection(modelTag=modelTag
                                            ,advectedVar="We"
                                            ,fluxVar="Ge_dual",
                                            vData=vData)

    # No boundary terms means reflective boundaries => allows all outflow to be governed by sheath heat transmission coefficients 

    rk.addModel(electronPAdvection.dict())

    # Lorentz force work terms

    #Adding the model tag to tag list
    modelTag = "lorentzForceWork"

    #Initializing model
    lorentzForceWorkModel = cm.lorentzForceWork(modelTag=modelTag,
                                        eFieldName="E_dual",
                                        speciesFluxes=["Ge_dual"],
                                        speciesEnergies=["We"],
                                        species=[electronSpecies])

    rk.addModel(lorentzForceWorkModel.dict())

    # Derived from expressions in [Makarov et al](https://doi.org/10.1063/5.0047618)

    ionZ = 1
    sqrt2 = np.sqrt(2)

    delta = 1 + (65*sqrt2/32 + 433*sqrt2/288 - 23*sqrt2/16)*ionZ + (5629/1152 - 529/128)*ionZ**2 #A30 in Makarov assuming single ion species and 0 mass ratio

    thermFrictionConst = 25*sqrt2*ionZ*(1+11*sqrt2*ionZ/30)/(16*delta) #A50

    elCondConst = 125*(1+433*sqrt2*ionZ/360)/(32*delta)

    # Braginskii heat fluxes

    #Adding the model tag to tag list
    modelTag = "braginskiiq"

    #Initializing model
    braginskiiHFModel = sc.CustomModel(modelTag=modelTag)

    mbData = sc.VarlikeModelboundData()

    mbData.addVariable("logLei",sc.derivationRule("logLeiD+",["Te_dual","ne_dual"]))

    braginskiiHFModel.setModelboundData(mbData.dict())

    # Setting normalization constant calculation 
    normConstI = sc.CustomNormConst(multConst=-1.0)

    nConstGradT = 12*np.pi**1.5*epsilon0**2/np.sqrt(elMass*elCharge) # Comes from e-i collision time

    normConstGradTEl = sc.CustomNormConst(multConst=-nConstGradT*elCondConst,normNames=["eVTemperature","length","heatFlux"],normPowers=[3.5,-1.0,-1.0])

    #Variable data 

    if kwargs.get("useKineticCorrections",False):
        gradDataEl = sc.VarData(reqRowVars=["Te_dual","qRatio"],reqRowPowers=[2.5,1.0],reqMBRowVars=["logLei"],reqMBRowPowers=[-1.0])
    else:
        gradDataEl = sc.VarData(reqRowVars=["Te_dual"],reqRowPowers=[2.5],reqMBRowVars=["logLei"],reqMBRowPowers=[-1.0])

    # Electrons 

    evolvedVar = "qe_dual"

    #Identity term

    identityTermEl = sc.GeneralMatrixTerm(evolvedVar,customNormConst=normConstI,stencilData=sc.diagonalStencil())

    braginskiiHFModel.addTerm("identityTerm_e",identityTermEl)

    #Gradient terms 

    implicitVar = "Te"

    gradTermEl = sc.GeneralMatrixTerm(evolvedVar,implicitVar=implicitVar,customNormConst=normConstGradTEl,stencilData=sc.staggeredGradStencil(),varData=gradDataEl)

    braginskiiHFModel.addTerm("bulkGrad_e",gradTermEl)

    rk.addModel(braginskiiHFModel.dict())

    # Electron heat flux divergence 

    #Adding the model tag to tag list
    modelTag = "divq_e"

    #Initializing model
    electronDivQModel = sc.CustomModel(modelTag=modelTag)

    #Setting normalization constants

    normFlux = sc.CustomNormConst(multConst=-1/elCharge,normNames=["heatFlux","time","length","density","eVTemperature"],normPowers=[1.0,1.0,-1.0,-1.0,-1.0])

    #Bulk flux divergence 

    evolvedVar = "We"
    implicitVar = "qe_dual"

    divFluxTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar=implicitVar,customNormConst=normFlux,stencilData=sc.staggeredDivStencil())

    electronDivQModel.addTerm("divFlux",divFluxTerm)
    
    if loglinExtrap:
        normBC = sc.CustomNormConst(multConst=-1.0/dxNNorm,normNames=["speed","time","length"],normPowers=[1.0,1.0,-1.0])
        
        # Add Right boundary term with Bohm condition to outflow

        implicitVar = "ne"
        vDataBC = sc.VarData(reqRowVars=["gammaRight","cs_lc","nb","ne"],reqRowPowers=[1.0,1.0,1.0,-1.0],reqColVars=["Te"])
        rightBCTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,customNormConst=normBC,varData=vDataBC,stencilData=sc.diagonalStencil(evolvedXCells=[len(xGridWidths)]))

        electronDivQModel.addTerm("rightBCT",rightBCTerm)

        # Add Right boundary term with Bohm condition to outflow (kinetic energy term)
        vDataBCKin = sc.VarData(reqRowVars=["cs_lc","nb","ne"],reqRowPowers=[3.0,1.0,-1.0])

        rightBCTerm2 = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar, customNormConst=normBC,
                                        varData=vDataBCKin, stencilData=sc.diagonalStencil(evolvedXCells=[len(xGridWidths)]))

        electronDivQModel.addTerm("rightBCU", rightBCTerm2)
    else:

        normBC = sc.CustomNormConst(multConst=-1.0, normNames=["speed", "time", "length"], normPowers=[1.0, 1.0, -1.0])
        
        # Add Right boundary term with Bohm condition to outflow (internal energy term)

        implicitVar = "ne"
        vDataBCRight = sc.VarData(reqRowVars=["gammaRight"], reqColVars=["Te"])

        rightBCTerm1 = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar, customNormConst=normBC,
                                        varData=vDataBCRight, stencilData=sc.boundaryStencilDiv("ue", "cs"))

        electronDivQModel.addTerm("rightBCT", rightBCTerm1)

        # Add Right boundary term with Bohm condition to outflow (kinetic energy term)

        vDataBCRightKin = sc.VarData(reqRowVars=["u_b"],reqRowPowers=[2.0])

        rightBCTerm2 = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar, customNormConst=normBC,
                                        varData=vDataBCRightKin, stencilData=sc.boundaryStencilDiv("ue", "cs"))

        electronDivQModel.addTerm("rightBCU", rightBCTerm2)

    rk.addModel(electronDivQModel.dict())

    Nh = kwargs.get("Nh",17)
    Lh = sum(xGridWidths[0:Nh])
    heatingPower = kwargs.get("heatingPower",1.0)/Lh #in MW/m^3
    energyInjectionRate = heatingPower *1e6 * timeNorm/(densNorm*elCharge*tempNorm)
    energyInjectionRate = energyInjectionRate
    xProfileEnergy = np.zeros(Nx)
    xProfileEnergy[0:Nh] = energyInjectionRate

    # Energy source model

    #Adding the model tag to tag list
    modelTag = "energySource"

    #Initializing model
    energySourceModel = sc.CustomModel(modelTag=modelTag)

    #Electrons

    evolvedVar = "We"
    energySourceTermEl = cm.simpleSourceTerm(evolvedVar=evolvedVar,sourceProfile=xProfileEnergy)

    energySourceModel.addTerm("electronSource",energySourceTermEl)

    #Perturbation 

    if "perturbationTimeSignal" in kwargs:
        assert isinstance(kwargs.get("perturbationTimeSignal"),sc.TimeSignalData), "perturbationTimeSignal not of correct type"
        heatingPower = kwargs.get("pertHeatingPower",10.0)/Lh #in MW/m^3
        energyInjectionRate = heatingPower *1e6 * timeNorm/(densNorm*elCharge*tempNorm)
        energyInjectionRate = energyInjectionRate
        xProfileEnergy[0:Nh] = energyInjectionRate
        heatTermPert = cm.simpleSourceTerm(evolvedVar=evolvedVar,sourceProfile=xProfileEnergy,timeSignal=kwargs.get("perturbationTimeSignal"))
        energySourceModel.addTerm("perturbationSource",heatTermPert)


    rk.addModel(energySourceModel.dict())

    #Electron-ion friction force terms 
    
    #Adding the model tag to tag list
    modelTag = "eiFriction"

    #Initializing model
    eiFrictionModel = sc.CustomModel(modelTag=modelTag)

    # Setting normalization constant calculation 
    normConstTel = sc.CustomNormConst(multConst=-elCharge*thermFrictionConst/elMass,normNames=["eVTemperature","time","speed","length"],normPowers=[1.0,1.0,-1.0,-1.0])
    normConstTion = sc.CustomNormConst(multConst=elCharge*thermFrictionConst/ionMass,normNames=["eVTemperature","time","speed","length"],normPowers=[1.0,1.0,-1.0,-1.0])

    # Creating modelbound data properties for e-i Coulomb log 

    mbData = sc.VarlikeModelboundData()
    mbData.addVariable("logLei",sc.derivationRule("logLeiD+",["Te_dual","ne_dual"]))

    eiFrictionModel.setModelboundData(mbData.dict())

    vDataGradT = sc.VarData(reqRowVars=["ne_dual"])

    #Grad T terms 
    implicitVar="Te"
    evolvedVar = "Ge_dual"

    electronGradTFriction = sc.GeneralMatrixTerm(evolvedVar,implicitVar=implicitVar,customNormConst=normConstTel,varData=vDataGradT,stencilData=sc.staggeredGradStencil())

    eiFrictionModel.addTerm("electronGradTFriction",electronGradTFriction)

    evolvedVar = "Gi_dual"

    ionGradTFriction = sc.GeneralMatrixTerm(evolvedVar,implicitVar=implicitVar,customNormConst=normConstTion,varData=vDataGradT,stencilData=sc.staggeredGradStencil(),implicitGroups=[2])

    eiFrictionModel.addTerm("ionGradTFriction",ionGradTFriction)

    rk.addModel(eiFrictionModel.dict())

    #Ion-neutral CX friction force terms 
    
    #Adding the model tag to tag list
    modelTag = "inFriction"

    mbData = sc.VarlikeModelboundData()

    #Initializing model
    inFrictionModel = sc.CustomModel(modelTag=modelTag)

    #Ion friction term 

    evolvedVar = "Gi_dual"

    implicitVar = "Gi_dual"

    amjuelPath = kwargs.get("amjuelPath","../data/amjuel.tex")
    if kwargs.get("amjuelCXRate",False):
        rk.addCustomDerivation("logTiDeriv",sc.generalizedIntPolyDerivation([[1]],[rk.normalization["eVTemperature"]/2],funcName="log"))
        rk.addVar("logTi",isDerived=True,derivationRule=sc.derivationRule("logTiDeriv",["Te_dual"]))
        rk.addCustomDerivation("cxRateDeriv",ams.AMJUELDeriv1D("3.1.8","H.2",timeNorm=timeNorm,densNorm=densNorm,amjuelFilename=amjuelPath))
        rk.addVar("cxRate",isDerived=True,derivationRule=sc.derivationRule("cxRateDeriv",["logTi"]))

        normConstCX = sc.CustomNormConst(multConst=-1) 
        vDataIonCX = sc.VarData(reqRowVars=["n1_dual","cxRate"])  
        ionCXFriction = sc.GeneralMatrixTerm(evolvedVar,implicitVar=implicitVar,customNormConst=normConstCX,varData=vDataIonCX,stencilData=sc.diagonalStencil())

        inFrictionModel.addTerm("iFriction_cx",ionCXFriction)
        
    else:
        mbData.addVariable("abs_ui",derivationRule=sc.derivationRule("absDeriv",["ui_dual"]))

        # Use constant low-energy CX cross-sections
        sigmaCx = [3.0e-19, 2**4 * 1.0e-19, 3**4 * 7.0e-20] + [i**4 * 6.0e-20 for i in range(4,numNeutrals+1)]

        # Setting normalization constant calculation 
        normConstCX = [sc.CustomNormConst(multConst=-sigmaCx[i]/sigmaNorm,normNames=["time","density","speed","crossSection"],normPowers=[1.0,1.0,1.0,1.0]) for i in range(numNeutrals)]

        vDataIonCX = [sc.VarData(reqRowVars=["n" + str(i+1)+"_dual"],reqMBRowVars=["abs_ui"])  for i in range(numNeutrals)]

        ionCXFriction = [sc.GeneralMatrixTerm(evolvedVar,implicitVar=implicitVar,customNormConst=normConstCX[i],varData=vDataIonCX[i],stencilData=sc.diagonalStencil()) for i in range(numNeutrals)]

        for i in range(numNeutrals):
            inFrictionModel.addTerm("iFriction_cx"+str(i+1),ionCXFriction[i])

    inFrictionModel.setModelboundData(mbData.dict())
    
    rk.addModel(inFrictionModel.dict())

    # Ground state diffusion and recyling

    #Adding the model tag to tag list
    modelTag = "neutDyn"

    #Initializing model
    neutDynModel = sc.CustomModel(modelTag=modelTag)

    recConst = 1.0 # Recycling coef
    normConstRec = sc.CustomNormConst(multConst=recConst,normNames=["speed","time","length"],normPowers=[1.0,1.0,-1.0])
    
    if kwargs.get("amjuelCXRate",False):
        
        rk.addCustomDerivation("amjDiff",sc.simpleDerivation(np.sqrt(Tn)/2,[-1.0,0.5,-1.0]))

        normConstDiff = sc.CustomNormConst(multConst = elMass/ionMass) 
        evolvedVar = "n1"
        implicitVar = "n1" 
        diffTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,customNormConst=normConstDiff,stencilData=sc.diffusionStencil("amjDiff",["ni_dual","Te_dual","cxRate"],doNotInterpolate=True))
        neutDynModel.addTerm("neutralDiff",diffTerm)
        
    else:
        sigmaCx = [3.0e-19, 2**4 * 1.0e-19, 3**4 * 7.0e-20] + [i**4 * 6.0e-20 for i in range(4,numNeutrals+1)]
        normConstDiff = [sc.CustomNormConst(multConst = np.sqrt(elMass/ionMass) / (sigmaCx[i] / sigmaNorm), normNames=["density","length","crossSection"],normPowers=[-1.0,-1.0,-1.0]) for i in range(numNeutrals)]

        # Diffusion term
        for i in range(numNeutrals):
            evolvedVar = "n" + str(i+1)
            implicitVar = "n" + str(i+1)
            diffTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,customNormConst=normConstDiff[i],stencilData=sc.diffusionStencil("neutDiffD",["ni_dual"],doNotInterpolate=True))
            neutDynModel.addTerm("neutralDiff"+str(i+1),diffTerm)

    #Recycling term 
    evolvedVar = "n1"
    implicitVar = "ni"

    if loglinExtrap:
        normBC = sc.CustomNormConst(multConst=recConst/dxNNorm)
        vDataBC = sc.VarData(reqRowVars=["cs_lc","nb","ni"],reqRowPowers=[1.0,1.0,-1.0])
        recTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,customNormConst=normBC,varData=vDataBC,stencilData=sc.diagonalStencil(evolvedXCells=[len(xGridWidths)]),implicitGroups=[2])
    else:
        recTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,customNormConst=normConstRec,stencilData=sc.boundaryStencilDiv("ui","cs"),implicitGroups=[2])
        
    neutDynModel.addTerm("recyclingTerm",recTerm)

    rk.addModel(neutDynModel.dict())

    # ### CRM density and energy evolution

    mbData = crm.ModelboundCRMData()

    if kwargs.get("amjuelRates",False):
        ams.addAMJUELDerivs(ams.amjuelHydrogenAtomDerivs(),rk,timeNorm,amjuelFilename=amjuelPath)
        rk.addCustomDerivation("recombPartEnergy",ams.AMJUELDeriv("2.1.8","H.4",timeNorm=timeNorm,densNorm=densNorm,tempNorm=-tempNorm/13.6,amjuelFilename=amjuelPath))
        rk.addCustomDerivation("recombEnergyTotal",sc.additiveDerivation(["recombEnergy","recombPartEnergy"],1,[[1,2],[1,2]],[1,1]))
        ams.addLogVars(rk,"ne","Te")

        mbData.addTransitionEnergy(1.0)
        ionizationTransition = crm.derivedTransition([0,1],[0,0,-1],transitionEnergy=1.0,ruleName="ionPart",requiredVars=["logne","logTe"],energyRuleName="ionEnergy",energyRequiredVars=["logne","logTe"])
        mbData.addTransition("ionAMJUEL",ionizationTransition)

        recombTransition = crm.derivedTransition([0,-1],[1],transitionEnergy=1.0,ruleName="recombPart",requiredVars=["logne","logTe"],energyRuleName="recombEnergyTotal",energyRequiredVars=["logne","logTe"])
        mbData.addTransition("recombAMJUEL",recombTransition)
        
        ionTransPrefix="ionAMJUEL"
    else:

        includedJanevTransitions = kwargs.get("includedJanevTransitions",["ion"]) 
        crm.addJanevTransitionsToCRMData(mbData,numNeutrals,tempNorm,"f","Te",detailedBalanceCSPriority=1,processes=includedJanevTransitions)

        if kwargs.get("includeSpontEmission",False):
            spontTransDict = crm.readNISTAkiCSV("Aki.csv")
            crm.addHSpontaneousEmissionToCRMData(mbData,spontTransDict,min(numNeutrals,20),min(numNeutrals,20),timeNorm,tempNorm) #NIST data only has a full transition list for n<=20

        ionTransPrefix = "JanevIon"
    #CRM model
    
    #Adding the model tag to tag list
    modelTag = "CRMmodel"

    #Initializing model
    crmModel = sc.CustomModel(modelTag=modelTag)

    crmModel.setModelboundData(mbData.dict())

    #Add ionization term generator for ions
    ionInds,ionEnergies = mbData.getTransitionIndicesAndEnergies(ionTransPrefix)
    crmTermGeneratorIon = crm.termGeneratorCRM(implicitTermGroups=[2],evolvedSpeciesIDs=[-1],includedTransitionIndices=ionInds)
    crmModel.addTermGenerator("crmTermGenIonIonization",crmTermGeneratorIon)

    otherIndices = list(range(1,len(mbData.transitionTags)+1))
    for ind in ionInds:
        otherIndices.remove(ind)
    if len(otherIndices) > 0:
        crmTermGeneratorOther = crm.termGeneratorCRM(implicitTermGroups=[4],evolvedSpeciesIDs=[-1],includedTransitionIndices=otherIndices)
        crmModel.addTermGenerator("crmTermGenIonOther",crmTermGeneratorOther)
    #Add all other terms for other particle species
    crmTermGenerator = crm.termGeneratorCRM(evolvedSpeciesIDs=[0]+[i+1 for i in range(numNeutrals)])

    crmModel.addTermGenerator("crmTermGen",crmTermGenerator)
    electronEnergyTermGenerator = crm.termGeneratorCRMElEnergy("We",implicitTermGroups=[3])

    crmModel.addTermGenerator("crmElEnergyGen",electronEnergyTermGenerator)

    rk.addModel(crmModel.dict())

    integrator = sc.picardBDEIntegrator(absTol=100.0,convergenceVars=["ne","ni","Ge_dual","Gi_dual","We","Te","n1"],nonlinTol=1e-10,internalStepControl=True) 

    rk.addIntegrator("BE",integrator)

    initialTimestep=kwargs.get("initialTimestep",0.1)
    rk.setIntegratorGlobalData(4,2,initialTimestep) 

    timestepControllerOptions = sc.scalingTimestepController(["ne","Te"],[-1.0,1.5])

    rk.setTimestepController(timestepControllerOptions)

    bdeStep = sc.IntegrationStep("BE",defaultEvaluateGroups=[1,2,3,4],defaultUpdateModelData=True,defaultUpdateGroups=[1,2,3,4])

    for tag in rk.modelTags():
        bdeStep.addModel(tag)

    rk.addIntegrationStep("StepBDE",bdeStep.dict())

    rk.setTimeTargetTimestepping(9000.0)
    rk.setMinimumIntervalOutput(1500.0)
    rk.setRestartOptions(True, False, 1000) 

    rk.addTermDiagnosisForVars(["We","ne","Ge_dual","Gi_dual","n1","E_dual","qe_dual"])

    rk.addVar("ionsource",isDerived=True)
    rk.addManipulator("ionsource",sc.groupEvaluatorManipulator("CRMmodel",2,"ionsource"))

    if kwargs.get("amjuelRates",False):
        rk.addVar("recombSource",isDerived=True)
        rk.addManipulator("recombSource",sc.groupEvaluatorManipulator("CRMmodel",4,"recombSource"))

    rk.addVar("elInelEnergyLossIon",isDerived=True)
    rk.addManipulator("elInelEnergyLossIon",sc.groupEvaluatorManipulator("CRMmodel",3,"elInelEnergyLossIon"))

    rk.setPETScOptions(cliOpts="-pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1",kspSolverType="gmres")

    return rk