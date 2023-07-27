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

def generatorSKThesisKin(**kwargs) -> RKWrapper:
    """Generate a wrapper containing a SOL-KiT run similar to the Mijin thesis kinetic electron version. Neutrals are diffusive, the heating profile is a step function upstream, recycling is at 100%, the neutrals have a fixed temperature (with a spurious sqrt(2) factor to match SOL-KiT's diffusion) and Ti=Te. The CRM can either use a SOL-KiT-like Janev cross-section calculation or AMJUEL rates (where the electrons have their energy reduced using a drag-like cooling term), and similarly for CX. Both the spatial and velocity grid widths are geometric. If not initializing from a previous run, the initial profiles are based on the 2-Point Model, with 100% ionization fraction.

    The following kwargs are available:

    jsonFilepath (str): Config filepath. Defaults to ".config.json".
    hdf5Filepath (str): Input and output hdf5 folder path. Defaults to "./RMKOutput/RMK_SK_comp_staggered_kin/".
    mpiProcsX (int): Number of MPI processes in the spatial direction. Defaults to 16.
    mpiProcsH (int): Number of MPI processes in the harmonic direction. Defaults to 1.
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
    initFromFluidRun: If true will initialize from given hdf5 file assuming it is a fluid run. Defaults to False.
    hdf5InputFile (str): Path of hdf5 input file (without the extension!). Defaults to "ReMKiT1DVarInput".
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
    hdf5Filepath = kwargs.get("hdf5Filepath","./RMKOutput/RMK_SK_comp_staggered_kin/" ) 
    rk.setHDF5Path(hdf5Filepath)

    numProcsX = kwargs.get("mpiProcsX",16) # Number of processes in x direction
    numProcsH = kwargs.get("mpiProcsH",1) # Number of processes in harmonic 
    numProcs = numProcsX * numProcsH
    haloWidth = 1 # Halo width in cells

    rk.setMPIData(numProcsX,numProcsH,haloWidth)

    # ### Normalization

    rk.setNormDensity(1.0e19)
    rk.setNormTemperature(10.0)
    rk.setNormRefZ(1.0)

    tempNorm = rk.normalization["eVTemperature"] 
    densNorm = rk.normalization["density"]
    skNorms = skn.calculateNorms(tempNorm,densNorm,1)
    
    timeNorm = skNorms["time"]
    lengthNorm = skNorms["length"]
    sigmaNorm = skNorms["crossSection"]

    # ### Grid initialization

    dx0 = kwargs.get("dx0",0.27)
    dxN = kwargs.get("dxN",0.0125)
    Nx = kwargs.get("Nx",128) 
    xGridWidths = np.geomspace(dx0,dxN,Nx)
    L = sum(xGridWidths)
    dv0 = kwargs.get("dv0",0.05)
    dvN = kwargs.get("dvN",0.4) 
    Nv = kwargs.get("Nv",80)  
    vGridWidths = np.geomspace(dv0,dvN,Nv)
    lMax = kwargs.get("lmax",1)
    gridObj = Grid(xGridWidths,vGridWidths,lMax,interpretXGridAsWidths=True,interpretVGridAsWidths=True,isLengthInMeters=True)

    # Add the grid to the config file
    rk.grid = gridObj

    # ### Custom derivations

    # Diffusion coefficient derivation in 1D with neutral temperature Tn and with the cross section used being the low energy charge-exchange cross-seciton
    # NOTE: SOL-KiT has a spurious sqrt(2) factor in the diffusion coefficient, so that is kept here for a consistent comparison
    Tn = kwargs.get("Tn",3.0)/tempNorm

    diffusionDeriv = sc.simpleDerivation(np.sqrt(Tn)/2,[-1.0])

    rk.addCustomDerivation("neutDiffD",diffusionDeriv)

    rk.addCustomDerivation("lbcRightExt",sc.distScalingExtrapolationDerivation(True,True))

    rk.addCustomDerivation("linExtrapRight",sc.boundedExtrapolationDerivation(sc.linExtrapolation(),ignoreUpperBound=True))

    loglinExtrap = kwargs.get("loglinExtrap",False)
    if loglinExtrap:
        rk.addCustomDerivation("logLinExtrapRight",sc.boundedExtrapolationDerivation(sc.linLogExtrapolation(),ignoreUpperBound=True))
    
        rk.addCustomDerivation("lastCellExt",sc.locValExtractorDerivation(len(xGridWidths)))

        rk.addCustomDerivation("secondToLastCellExt",sc.locValExtractorDerivation(len(xGridWidths)-1))

        rk.addCustomDerivation("uLastCell",sc.additiveDerivation(["lastCellExt","secondToLastCellExt"],1.0,[[1],[2]],[0.5,0.5]))

        rk.addCustomDerivation("fluxDeriv",sc.simpleDerivation(1.0,[1.0,1.0]))
        dxNNorm = dxN/lengthNorm
        dxNStagNorm = dxNNorm + xGridWidths[-2]/(2*lengthNorm)

    else:
        
        rk.addCustomDerivation("linExtrapRightLB",sc.boundedExtrapolationDerivation(sc.linExtrapolation(),expectLowerBoundVar=True,ignoreUpperBound=True))

        rk.addCustomDerivation("boundaryFlux",sc.multiplicativeDerivation("linExtrapRight",innerDerivationIndices=[1],outerDerivation="linExtrapRightLB",outerDerivationIndices=[2,3]))
    
    rk.addCustomDerivation("identityDeriv",sc.simpleDerivation(1.0,[1.0]))
    absDeriv = sc.multiplicativeDerivation("identityDeriv",[1],funcName="abs")
    rk.addCustomDerivation("absDeriv",absDeriv)
    rk.addCustomDerivation("square",sc.simpleDerivation(1.0,[2.0]))

    rk.addCustomDerivation("convHeatflux",sc.simpleDerivation(5/2,[1.0,1.0]))
    rk.addCustomDerivation("condHeatflux",sc.additiveDerivation(["identityDeriv","convHeatflux"],1.0,[[1],[2,3]],[1.0,-1.0]))


    nConstGradT = 12*np.pi**1.5*epsilon0**2/np.sqrt(elMass*elCharge)  # Comes from e-i collision time

    ionZ = 1
    sqrt2 = np.sqrt(2)

    delta = (1 + 65*sqrt2/32 + 433*sqrt2/288 - 23*sqrt2/16)*ionZ + (5629/1152 - 529/128) * ionZ**2  # A30 in Makarov assuming single ion species and 0 mass ratio
    elCondConst = 125*(1+433*sqrt2*ionZ/360)/(32*delta)

    #Taken from config after initialization
    qNorm = skNorms["heatFlux"]

    normalizationConst = tempNorm**3.5/(lengthNorm*qNorm)
    kappaDeriv = sc.simpleDerivation(-elCondConst*nConstGradT*normalizationConst, [2.5,-1.0])
    rk.addCustomDerivation("kappa", kappaDeriv)

    qDeriv = sc.multiplicativeDerivation(innerDerivation="kappa", innerDerivationIndices=[1,2], outerDerivation="gradDeriv", outerDerivationIndices=[1])
    rk.addCustomDerivation("qT", qDeriv)

    rk.setStandardTextbookOptions(tempDerivSpeciesIDs=[0])

    # ### Handling particle species data

    rk.addSpecies("e",0,atomicA=elMass/amu,charge=-1.0,associatedVars=["ne","ne_dual","Ge","We"]) 
    rk.addSpecies("D+",-1,atomicA=2.014,charge=1.0,associatedVars=["ni","ni_dual","Gi"])

    # Set neutrals 
    numNeutrals=kwargs.get("numNeutrals",1)
    neutralDensList = ["n"+str(i) for i in range(1,numNeutrals+1)] # List of neutral density names

    for neutral in neutralDensList:
        rk.addSpecies("D"+neutral[1:],int(neutral[1:]),heavySpeciesMass,associatedVars=[neutral,neutral+"_dual"])

    ionSpecies = rk.getSpecies("D+")


    # Two-point model initialization 

    Tu = 20/tempNorm #upstream temperature
    Td = 5/tempNorm #downstream temperature

    T = (Tu**(7/2) - (Tu**(7/2)-Td**(7/2))*gridObj.xGrid/L)**(2/7)

    nu = kwargs.get("nu",0.8) #upstream density

    n = nu*Tu/T 

    W = 3*n*T/2

    f = np.zeros([gridObj.numX(),gridObj.numH(),gridObj.numV()])
    for i in range(gridObj.numX()):
        f[i,gridObj.getH(0)-1,:] = np.pi**(-1.5) * T[i] ** (-1.5) * n[i] * np.exp(-gridObj.vGrid**2/T[i])

    # Rescale distribution function to ensure that the numerical density moment agrees with the initial values
    numerical_dens = gridObj.velocityMoment(f,0,1)
    for i in range(gridObj.numX()):
        f[i,gridObj.getH(0)-1,:] = n[i] *f[i,gridObj.getH(0)-1,:]/numerical_dens[i]
        
    rk.addVarAndDual("f",f,isDistribution=True,isCommunicated=True)

    rk.addVar("We",W,units='$10eV$',isDerived=True,derivationRule=sc.derivationRule("energyMoment",["f"]),isCommunicated=True)
    rk.addVarAndDual("ne",n,units='$10^{19} m^{-3}$',isDerived=True,derivationRule=sc.derivationRule("densityMoment",["f"]),isCommunicated=True)
    rk.addVarAndDual("Ge",isDerived=True,primaryOnDualGrid=True,derivationRule=sc.derivationRule("fluxMoment",["f"]),isCommunicated=True)
    rk.addVarAndDual("qe_tot",isDerived=True,primaryOnDualGrid=True,derivationRule=sc.derivationRule("heatFluxMoment",["f"]))
    rk.addVarAndDual("qe",isDerived=True,primaryOnDualGrid=True,derivationRule=sc.derivationRule("condHeatflux",["qe_tot_dual","Te_dual","ue_dual"]))
    rk.addVarAndDual("Te",T,isDerived=True,units='$10eV$',isCommunicated=True,derivationRule=sc.derivationRule("tempFromEnergye",["We","ne","Ge"]))
    rk.addVarAndDual("qT",isDerived=True,derivationRule=sc.derivationRule("qT",["Te","logLee"]),isCommunicated=True) # Braginskii heatflux variable for comparison

    rk.addVarAndDual("ni",n,isCommunicated=True)
    rk.addVarAndDual("Gi",primaryOnDualGrid=True,isCommunicated=True)

    rk.addVar("qRatio",isDerived=True,isOnDualGrid=True,derivationRule=sc.derivationRule("flowSpeedFromFlux",["qe_dual","qT_dual"]))

    rk.addVar("cs",isDerived=True,derivationRule=sc.derivationRule("sonicSpeedD+",["Te","Te"]))

    hostProc = numProcs-numProcsH
    if loglinExtrap:

        rk.addVar("cs_lc",isDerived=True,isScalar=True,derivationRule=sc.derivationRule("lastCellExt",["cs"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

        rk.addVar("ni_b",isDerived=True,isScalar=True,derivationRule=sc.derivationRule("logLinExtrapRight",["ni"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

        rk.addVar("ne_b",isDerived=True,isScalar=True,derivationRule=sc.derivationRule("logLinExtrapRight",["ne"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

        rk.addVar("ui_lc",isDerived=True,isScalar=True,derivationRule=sc.derivationRule("uLastCell",["cs","ui_dual"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

        rk.addVar("ue_lc",isDerived=True,isScalar=True,derivationRule=sc.derivationRule("uLastCell",["cs","ue_dual"]),isCommunicated=True,hostScalarProcess=numProcs-numProcsH)

        rk.addVar("G_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
                ,derivationRule=sc.derivationRule("fluxDeriv",["ni_b","cs_lc"]))

    else:
        rk.addVar("cs_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
                ,derivationRule=sc.derivationRule("linExtrapRight",["cs"]))

        rk.addVar("ne_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
                ,derivationRule=sc.derivationRule("linExtrapRight",["ne"]))

        rk.addVar("ni_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
                ,derivationRule=sc.derivationRule("linExtrapRight",["ni"]))

        rk.addVar("G_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
                ,derivationRule=sc.derivationRule("boundaryFlux",["ni","ui","cs_b"]))


    # Set E field

    rk.addVarAndDual("E",primaryOnDualGrid=True)

    # Set derived fluid quantities

    rk.addVarAndDual("ue",isDerived=True,primaryOnDualGrid=True,derivationRule=sc.derivationRule("flowSpeedFromFlux",["Ge_dual","ne_dual"]),isCommunicated=True)
    rk.addVarAndDual("ui",isDerived=True,primaryOnDualGrid=True,derivationRule=sc.derivationRule("flowSpeedFromFlux",["Gi_dual","ni_dual"]),isCommunicated=True)

    # Set scalar quantities 
    rk.addVar("time",isScalar=True,isDerived=True)

    # Set neutral densities

    for neut in neutralDensList:
            rk.addVarAndDual(neut,units='$10^{19} m^{-3}$',isCommunicated=True)


    # ### Models 

    # ### Electron models

    # ### Advection

    advModel = cm.kinAdvX(modelTag="advf", distFunName="f", gridObj=gridObj)
    rk.addModel(advModel.dict())


    # ### E-field advection

    cm.addExAdvectionModel(modelTag="E-adv", distFunName="f", eFieldName="E_dual", wrapper=rk, dualDistFun="f_dual")


    # ### e-e collisions for l=0

    cm.addEECollIsotropic(modelTag="e-e0", distFunName="f", elTempVar="Te", elDensVar="ne", wrapper=rk)


    # ### e-i collisions for odd l with flowing ions

    cm.addFlowingIonEIColl(modelTag="e-i_odd",
                            distFunName="f",
                            ionDensVar="ni",
                            ionFlowSpeedVar="ui_dual",
                            electronDensVar="ne_dual",
                            electronTempVar="Te_dual",
                            ionSpeciesName="D+",
                            evolvedHarmonics=list(range(2, gridObj.numH()+1, 2)),
                            wrapper=rk,
                            dualDistFun="f_dual",
                            ionFluxVar="Gi_dual")

    # ### e-e collisions for odd l

    cm.addEECollHigherL(modelTag="e-e_odd",
                        distFunName="f",
                        elTempVar="Te_dual",
                        elDensVar="ne_dual",
                        wrapper=rk,
                        evolvedHarmonics=list(range(2, gridObj.numH()+1, 2)),
                        dualDistFun="f_dual")

    if lMax > 1:
        cm.addFlowingIonEIColl(modelTag="e-i_even",
                            distFunName="f",
                            ionDensVar="ni",
                            ionFlowSpeedVar="ui",
                            electronDensVar="ne",
                            electronTempVar="Te",
                            ionSpeciesName="D+",
                            evolvedHarmonics=list(range(3, gridObj.numH()+1, 2)),
                            wrapper=rk)

        cm.addEECollHigherL(modelTag="e-e_even",
                        distFunName="f",
                        elTempVar="Te",
                        elDensVar="ne",
                        wrapper=rk,
                        evolvedHarmonics=list(range(3, gridObj.numH()+1, 2)))
        
    # ### Logical boundary condition

    cm.addLBCModel("lbc_right","f",rk,sc.derivationRule("lbcRightExt",["f","ne","ne_dual","ne_b"]),
                    "G_b",evolvedHarmonics=list(range(1, gridObj.numH()+1, 2)))

    # ### Heating

    Nh = kwargs.get("Nh",17)
    Lh = sum(xGridWidths[0:Nh])
    heatingPower = kwargs.get("heatingPower",1.0)/Lh #in MW/m^3
    energyInjectionRate = heatingPower *1e6 * timeNorm/(densNorm*elCharge*tempNorm)
    energyInjectionRate = energyInjectionRate
    xProfileEnergy = np.zeros(Nx)
    xProfileEnergy[0:Nh] = energyInjectionRate
    heatTerm = cm.diffusiveHeatingTerm("f","ne",heatingProfile=xProfileEnergy.tolist(),wrapper=rk)
    heatModel = sc.CustomModel("heating")
    heatModel.addTerm("diffHeating",heatTerm)

     # ### Heating perturbation

    if "perturbationTimeSignal" in kwargs:
        assert isinstance(kwargs.get("perturbationTimeSignal"),sc.TimeSignalData), "perturbationTimeSignal not of correct type"
        heatingPower = kwargs.get("pertHeatingPower",10.0)/Lh #in MW/m^3
        energyInjectionRate = heatingPower *1e6 * timeNorm/(densNorm*elCharge*tempNorm)
        energyInjectionRate = energyInjectionRate
        xProfileEnergy[0:Nh] = energyInjectionRate
        heatTermPert = cm.diffusiveHeatingTerm("f","ne",heatingProfile=xProfileEnergy.tolist(),wrapper=rk,timeSignal=kwargs.get("perturbationTimeSignal"))
        heatModel.addTerm("diffHeatingPert",heatTermPert)

    
    rk.addModel(heatModel.dict())

    # ### Ion continuity

    #Ion continuity advection

    #Adding the model tag to tag list
    modelTag = "continuity-ni"
    # Initializing model using common models

    if loglinExtrap:
        ionContModel = cm.staggeredAdvection(modelTag=modelTag
                                                ,advectedVar="ni"
                                                ,fluxVar="Gi_dual")

        normBC = sc.CustomNormConst(multConst=-1/dxNNorm)
        vDataBC = sc.VarData(reqRowVars=["cs_lc","ni_b","ni"],reqRowPowers=[1.0,1.0,-1.0])
        evolvedVar = "ni"
        outflowTerm = sc.GeneralMatrixTerm(evolvedVar,customNormConst=normBC,varData=vDataBC,stencilData=sc.diagonalStencil(evolvedXCells=[len(xGridWidths)]))
        ionContModel.addTerm("bcTerm",outflowTerm)

    else:
        ionContModel = cm.staggeredAdvection(modelTag=modelTag, advectedVar="ni",
                                        fluxVar="Gi_dual", advectionSpeed="ui", lowerBoundVar="cs", rightOutflow=True)

    rk.addModel(ionContModel.dict())

    # ### Pressure gradient forces

    #Ion pressure grad

    #Adding the model tag to tag list

    modelTag = "pressureGrad-Gi"

    #Initializing model
    ionPressureGradModel = cm.staggeredPressureGrad(modelTag=modelTag,fluxVar="Gi_dual",densityVar="ni",temperatureVar="Te",speciesMass=ionMass)

    rk.addModel(ionPressureGradModel.dict())

    # ### Momentum advection

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
        vDataBC = sc.VarData(reqRowVars=["cs_lc","ni_b"],reqRowPowers=[2.0,1.0],reqColVars=["ni"],reqColPowers=[-1.0])
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

    # ### Ampere-Maxwell term and Lorentz force

    #Ampere-Maxwell E field equation 
    
    #Adding the model tag to tag list
    modelTag = "ampereMaxwell"

    #Initializing model
    ampereMaxwellModel = cm.ampereMaxwell(modelTag=modelTag,
                                        eFieldName="E_dual",
                                        speciesFluxes=["Gi_dual"],
                                        species=[ionSpecies])

    kinAMTerm = cm.ampereMaxwellKineticElTerm("f","E_dual")

    ampereMaxwellModel.addTerm("kinElTerm",kinAMTerm)

    rk.addModel(ampereMaxwellModel.dict())

    #Lorentz force terms 
    
    #Adding the model tag to tag list
    modelTag = "lorentzForce"

    #Initializing model
    lorentzForceModel = cm.lorentzForces(modelTag=modelTag,
                                        eFieldName="E_dual",
                                        speciesFluxes=["Gi_dual"],
                                        speciesDensities=["ni_dual"],
                                        species=[ionSpecies])

    rk.addModel(lorentzForceModel.dict())

    # ### CX friction

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

    # ### Neutral diffusion and recycling

    # Ground state diffusion and recyling

    #Adding the model tag to tag list
    modelTag = "neutDyn"

    #Initializing model
    neutDynModel = sc.CustomModel(modelTag=modelTag)

    recConst = 1.0 # Recycling coef
    normConstRec = sc.CustomNormConst(multConst=recConst,normNames=["speed","time","length"],normPowers=[1.0,1.0,-1.0])
    
    if kwargs.get("amjuelCXRate",False):
        
        rk.addCustomDerivation("amjDiff",sc.simpleDerivation(np.sqrt(Tn)/2,[-1.0,0.5,-1]))

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
        vDataBC = sc.VarData(reqRowVars=["cs_lc","ni_b","ni"],reqRowPowers=[1.0,1.0,-1.0])
        recTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,customNormConst=normBC,varData=vDataBC,stencilData=sc.diagonalStencil(evolvedXCells=[len(xGridWidths)]),implicitGroups=[2])
    else:
        recTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,customNormConst=normConstRec,stencilData=sc.boundaryStencilDiv("ui","cs"),implicitGroups=[2])

    neutDynModel.addTerm("recyclingTerm",recTerm)

    rk.addModel(neutDynModel.dict())

    # ### CRM 

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
        
    else:

        includedJanevTransitions = kwargs.get("includedJanevTransitions",["ion"]) 
        crm.addJanevTransitionsToCRMData(mbData,numNeutrals,tempNorm,"f","Te",detailedBalanceCSPriority=1,processes=includedJanevTransitions)

        if kwargs.get("includeSpontEmission",False):
            spontTransDict = crm.readNISTAkiCSV("../data/Aki.csv")
            crm.addHSpontaneousEmissionToCRMData(mbData,spontTransDict,min(numNeutrals,20),min(numNeutrals,20),timeNorm,tempNorm) #NIST data only has a full transition list for n<=20
        
    #CRM model
    
    #Adding the model tag to tag list
    modelTag = "CRMmodel"

    #Initializing model
    crmModel = sc.CustomModel(modelTag=modelTag)

    crmModel.setModelboundData(mbData.dict())

    #Add term generator responsible for buildling CRM model for neutrals and ions
    crmTermGenerator = crm.termGeneratorCRM(evolvedSpeciesIDs=[-1]+[i+1 for i in range(numNeutrals)])

    crmModel.addTermGenerator("crmTermGen",crmTermGenerator)

    ionInds = []
    recomb3bInds = []
    if kwargs.get("amjuelRates",False):
        #Cooling terms due to radiation 
        ionInds,ionEnergies = mbData.getTransitionIndicesAndEnergies("ionAMJUEL")
        ionizationCoolingTerm = cm.dvEnergyTerm("f",sc.VarData(reqRowVars=["n1"], reqMBRowVars=["rate2index"+str(ionInds[0])]),rk,multConst=1.0)

        crmModel.addTerm("ionizationCooling",ionizationCoolingTerm)

        recomb3bInds,recombEnergies = mbData.getTransitionIndicesAndEnergies("recombAMJUEL")
        recombCoolingTerm = cm.dvEnergyTerm("f",sc.VarData(reqRowVars=["ni"],reqMBRowVars=["rate2index"+str(recomb3bInds[0])]),rk,multConst=1.0)

        crmModel.addTerm("recombCooling",recombCoolingTerm)
    else:
        
        if "ex" in includedJanevTransitions:
            # Add Boltzmann term generators for excitation
            exInds,exEnergies = mbData.getTransitionIndicesAndEnergies("JanevEx")

            crmBoltzTermGenExE = crm.termGeneratorCRMBoltz("f",1,exInds,exEnergies,implicitTermGroups=[1]) #Emission terms

            crmModel.addTermGenerator("exCRME",crmBoltzTermGenExE)

            crmBoltzTermGenExE2 = crm.termGeneratorCRMBoltz("f",2,exInds,exEnergies,implicitTermGroups=[1],associatedVarIndex=2) #Emission terms for l=1

            crmModel.addTermGenerator("exCRME2",crmBoltzTermGenExE2)

            crmBoltzTermGenExA = crm.termGeneratorCRMBoltz("f",1,exInds,exEnergies,absorptionTerms=True,implicitTermGroups=[1]) #Absorption terms

            crmModel.addTermGenerator("exCRMA",crmBoltzTermGenExA)

        #Add Boltzmann term generators for ionization

        if "ion" in includedJanevTransitions:
            
            ionInds,ionEnergies = mbData.getTransitionIndicesAndEnergies("JanevIon")

            crmBoltzTermGenIonE = crm.termGeneratorCRMBoltz("f",1,ionInds,ionEnergies,implicitTermGroups=[2]) #Emission terms

            crmModel.addTermGenerator("ionCRME",crmBoltzTermGenIonE)

            crmBoltzTermGenIonA = crm.termGeneratorCRMBoltz("f",1,ionInds,ionEnergies,absorptionTerms=True,implicitTermGroups=[2]) #Absorption terms

            crmModel.addTermGenerator("ionCRMA",crmBoltzTermGenIonA)

            crmBoltzTermGenIonE2 = crm.termGeneratorCRMBoltz("f",2,ionInds,ionEnergies,associatedVarIndex=2) #Emission terms for l=1

            crmModel.addTermGenerator("ionCRME2",crmBoltzTermGenIonE2)

        if "deex" in includedJanevTransitions:

            #Add Boltzmann term generators for deexcitation

            deexInds,deexEnergies = mbData.getTransitionIndicesAndEnergies("JanevDeex")

            crmBoltzTermGenDeexE = crm.termGeneratorCRMBoltz("f",1,deexInds,deexEnergies,detailedBalanceTerms=True,implicitTermGroups=[2]) #Emission terms

            crmModel.addTermGenerator("deexCRME",crmBoltzTermGenDeexE)

            crmBoltzTermGenDeexE2 = crm.termGeneratorCRMBoltz("f",2,deexInds,deexEnergies,detailedBalanceTerms=True,implicitTermGroups=[2],associatedVarIndex=2) #Emission terms for l=1

            crmModel.addTermGenerator("deexCRME2",crmBoltzTermGenDeexE2)

            crmBoltzTermGenDeexA = crm.termGeneratorCRMBoltz("f",1,deexInds,deexEnergies,absorptionTerms=True,detailedBalanceTerms=True,implicitTermGroups=[2]) #Absorption terms

            crmModel.addTermGenerator("deexCRMA",crmBoltzTermGenDeexA)

        if "recomb3b" in includedJanevTransitions:

            # #Add Boltzmann term generators for 3b recombination

            recomb3bInds,recomb3bEnergies = mbData.getTransitionIndicesAndEnergies("JanevRecomb3b")

            crmBoltzTermGen3bRecombE = crm.termGeneratorCRMBoltz("f",1,recomb3bInds,recomb3bEnergies,detailedBalanceTerms=True) #Emission terms

            crmModel.addTermGenerator("recomb3bCRME",crmBoltzTermGen3bRecombE)

            crmBoltzTermGen3bRecombE2 = crm.termGeneratorCRMBoltz("f",2,recomb3bInds,recomb3bEnergies,detailedBalanceTerms=True,associatedVarIndex=2) #Emission terms for l=1

            crmModel.addTermGenerator("recomb3bCRME2",crmBoltzTermGen3bRecombE2)

            crmBoltzTermGen3bRecombA = crm.termGeneratorCRMBoltz("f",1,recomb3bInds,recomb3bEnergies,absorptionTerms=True,detailedBalanceTerms=True) #Absorption terms

            crmModel.addTermGenerator("recomb3bCRMA",crmBoltzTermGen3bRecombA)

    #Add secondary electron sources/sinks due to ionization and recombination

    secElInds = ionInds + recomb3bInds 
    
    crmSecElTermGen = crm.termGeneratorCRMSecEl("f",secElInds)

    crmModel.addTermGenerator("secElCRM",crmSecElTermGen)

    #Add model to wrapper

    rk.addModel(crmModel.dict())

    # ### Integrator options

    integrator = sc.picardBDEIntegrator(absTol=100.0,convergenceVars=["ne","ni","Ge_dual","Gi_dual","We","n1"],associatedPETScGroup=1,maxNonlinIters=50,nonlinTol=1e-10,internalStepControl=True) 

    rk.addIntegrator("BE",integrator)


    # ### Timestep control
    # 
    # Here the timestep is rescaled based on collisionality.

    initialTimestep=kwargs.get("initialTimestep",0.1)
    rk.setIntegratorGlobalData(4,2,initialTimestep) 

    timestepControllerOptions = sc.scalingTimestepController(["ne","Te"],[-1.0,1.5])

    rk.setTimestepController(timestepControllerOptions)

    # ### Controlling integration steps

    bdeStep = sc.IntegrationStep("BE",defaultEvaluateGroups=[1,2,3,4],defaultUpdateModelData=True,defaultUpdateGroups=[1,2,3,4],globalStepFraction=1.0)
    for tag in rk.modelTags():
            bdeStep.addModel(tag)

    rk.addIntegrationStep("StepBDE",bdeStep.dict())

    # ### Time loop options
    # 
    # The main part of ReMKiT1D is the time loop, where the variables are advanced through time by repeatedly calling the integrators defined above. The following shows a way to set timeloop options based on a time target:

    rk.setTimeTargetTimestepping(50.0)
    rk.setMinimumIntervalOutput(10.0)
    rk.setRestartOptions(True, False, 100) #Change to True when restarting

    if kwargs.get("initFromFluidRun",False):
        rk.setHDF5FileInitialData(["f","ni","Gi_dual","E_dual"]+neutralDensList,filename=kwargs.get("hdf5InputFile","ReMKiT1DVarInput"))

    rk.addTermDiagnosisForVars(["Gi_dual","ni","E_dual","f"])

    rk.addVar("gammaRight",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=hostProc)

    rk.addManipulator("gammaExtRight",sc.extractorManipulator("lbc_right","gamma","gammaRight"))

    rk.addVar("logLee",isDerived=True)
    rk.addManipulator("logLeeExtract",sc.extractorManipulator("e-e0","logLee","logLee"))

    rk.setPETScOptions(cliOpts="-pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1",kspSolverType="gmres")

    return rk


