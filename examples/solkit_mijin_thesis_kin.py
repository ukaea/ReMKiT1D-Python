import numpy as np

import RMK_support as rmk 
import RMK_support.common_models as cm
import RMK_support.crm_support as crm
import RMK_support.amjuel_support as ams
from RMK_support import node, varFromNode, log
import RMK_support.stencils as stencils
from RMK_support.derivations import NodeDerivation, DerivationClosure,SimpleDerivation,BoundedExtrapolationDerivation


def generatorSKThesisKin(**kwargs) -> rmk.RMKContext:
    """Generate a context containing a SOL-KiT run similar to the Mijin thesis kinetic electron version. Neutrals are diffusive, the heating profile is a step function upstream, recycling is at 100%, the neutrals have a fixed temperature (with a spurious sqrt(2) factor to match SOL-KiT's diffusion) and Ti=Te. The CRM can either use a SOL-KiT-like Janev cross-section calculation or AMJUEL rates (where the electrons have their energy reduced using a drag-like cooling term), and similarly for CX. Both the spatial and velocity grid widths are geometric. If not initializing from a previous run, the initial profiles are based on the 2-Point Model, with 100% ionization fraction.

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
    numNeutrals (int): Number of neutral states tracked. Should be set to 1 if using AMJUEL rates. Defaults to 1.
    Tu (float): Initial upstream temperature in eV. Defaults to 20.
    Td (float): Initial downstream temperature in eV. Defaults to 5.
    nu (float): Initial upstream density in normalized units. Automatically determines the downstream density using a constant pressure assumption. Defaults to 0.8.
    Nh (int): Number of upstream cells affected by the heating operator. Defaults to 17.
    heatingPower (float): Effective upstream heating in MW/m^2. Defaults to 1.0.
    perturbationTimeSignal (sc.TimeSignal): Optional time signal for a periodic heating perturbation. Defaults to no perturbation.
    pertHeatingPower (float): The heating perturbation amplitude in MW/m^2. Defaults to 10.
    amjuelCXRate (bool): If true uses the AMJUEL CX rate H.2 3.1.8 instead of the SOL-KiT model. This rate is then divided by sqrt(2kTe/mi) to get the cross-section for neutral diffusion. Defaults to False.
    amjuelRates (bool): If true uses AMJUEL ionization and recombination particle and energy rates instead of the SOL-KiT atomic state resolved CRM. Defaults to False.
    includedJanevTransitions: Set of included transitions in the SOL-KiT style CRM. The full list is ["ex","deex","ion","recomb3b"]. Defaults to ["ion"].
    includeSpontEmission: If true will include spontaneous emission in the SOL-KiT style CRM for up to n=20. Defaults to False.
    initialTimestep: Used timestep in shortest e-i collision times in the system. Defaults to 0.1.
    initFromFluidRun: If true will initialize from given hdf5 file assuming it is a fluid run. Defaults to False.
    hdf5InputFile (str): Path of hdf5 input file (without the extension!). Defaults to "ReMKiT1DVarInput".
    amjuelPath (str): Path to amjuel.tex file. Defaults to "../data/amjuel.tex".

    Returns:
        RMKContext: Context containing run information
    """

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    amu = 1.6605390666e-27  # atomic mass unit
    ionMass = 2.014 * amu  # deuterium mass
    epsilon0 = 8.854188e-12  # vacuum permittivity
    heavySpeciesMass = 2.014  # in amus

    rk = rmk.RMKContext()
    hdf5Filepath = kwargs.get(
        "hdf5Filepath", "./RMKOutput/RMK_SK_comp_staggered_kin/"
    )
    rk.IOContext = rmk.IOContext(jsonFilepath=kwargs.get("jsonFilepath", "./config.json"),HDF5Dir=hdf5Filepath)

    rk.mpiContext = rmk.MPIContext(kwargs.get("mpiProcsX", 8),kwargs.get("mpiProcsH", 1))

    dx0 = kwargs.get("dx0", 0.27)
    dxN = kwargs.get("dxN", 0.0125)
    Nx = kwargs.get("Nx", 128)
    xGridWidths = np.geomspace(dx0, dxN, Nx)
    L = sum(xGridWidths)
    dv0 = kwargs.get("dv0", 0.05)
    dvN = kwargs.get("dvN", 0.4)
    Nv = kwargs.get("Nv", 80)
    vGridWidths = np.geomspace(dv0, dvN, Nv)
    lMax = kwargs.get("lmax", 1)
    rk.grid = rmk.Grid(
        xGridWidths,
        vGridWidths,
        lMax,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
        isLengthInMeters=True,
    )

    tempNorm = rk.norms["eVTemperature"]
    densNorm = rk.norms["density"]

    timeNorm = rk.norms["time"]

    lengthNorm = rk.norms["length"]
    qNorm = rk.norms["heatFlux"]

    Tn = kwargs.get("Tn", 3.0) / tempNorm

    rk.species.add(rmk.Species("e", 0, atomicA=elMass / amu, charge=-1.0))
    rk.species.add(rmk.Species("D+", -1, atomicA=heavySpeciesMass, charge=1.0))

    # Set neutrals
    numNeutrals = kwargs.get("numNeutrals", 1)
    neutralDensList = [
        "n" + str(i) for i in range(1, numNeutrals + 1)
    ]  # List of neutral density names

    neutralSpecies = []
    for neutral in neutralDensList:
        neutralSpecies.append(rmk.Species("D" + neutral[1:],
            int(neutral[1:]),
            heavySpeciesMass))
        rk.species.add(neutralSpecies[-1])

    electronSpecies = rk.species["e"]
    ionSpecies = rk.species["D+"]

    rk.textbook = rmk.Textbook(rk.grid,tempDerivSpeciesIDs=[electronSpecies.speciesID])

    # Two-point model initialization

    Tu = 20 / tempNorm  # upstream temperature
    Td = 5 / tempNorm  # downstream temperature

    TInit = (Tu ** (7 / 2) - (Tu ** (7 / 2) - Td ** (7 / 2)) * rk.grid.xGrid / L) ** (2 / 7)

    nu = kwargs.get("nu", 0.8)  # upstream density

    nInit = nu * Tu / TInit

    WInit = 3 * nInit * TInit / 2

    fInit = np.zeros([rk.grid.numX,rk.grid.numH,rk.grid.numV])
    for i in range(rk.grid.numX):
        fInit[i,rk.grid.getH(0)-1,:] = (np.pi*TInit[i])**(-1.5) * nInit[i]* np.exp(-rk.grid.vGrid**2/TInit[i])

    # Rescale distribution function to ensure that the numerical density moment agrees with the initial values
    numerical_dens = rk.grid.velocityMoment(fInit,0,1)
    for i in range(rk.grid.numX):
        fInit[i,rk.grid.getH(0)-1,:] = nInit[i] *fInit[i,rk.grid.getH(0)-1,:]/numerical_dens[i]
        
    f,f_dual = rmk.varAndDual("f",rk.grid,isDistribution=True,data=fInit)
    We = rmk.Variable("W",rk.grid,data=WInit,derivation=rk.textbook["energyMoment"],derivationArgs=[f.name],units="$10^{20} eVm^{-3}$")
    ne,ne_dual = rmk.varAndDual("n",rk.grid,derivation=rk.textbook["densityMoment"],derivationArgs=[f.name],units="$10^{19} m^{-3}$",data=nInit)

    Ge_dual,Ge = rmk.varAndDual("Ge",rk.grid,primaryOnDualGrid=True,derivation=rk.textbook["fluxMoment"],derivationArgs=[f.name])

    ue_dual,ue = rmk.varAndDual("ue",rk.grid,primaryOnDualGrid=True,derivation=rk.textbook["flowSpeedFromFlux"],derivationArgs=[Ge_dual.name,ne_dual.name])

    Te,Te_dual = rmk.varAndDual("Te",rk.grid,derivation=rk.textbook["tempFromEnergye"],derivationArgs=[We.name,ne.name,Ge.name],data=TInit)
    qe_tot_dual,qe_tot = rmk.varAndDual("qe_tot",rk.grid,primaryOnDualGrid=True,derivation=rk.textbook["heatFluxMoment"],derivationArgs=[f.name])

    qe_dual,qe = rmk.varAndDual("qe",rk.grid,primaryOnDualGrid=True,derivation = NodeDerivation("qe",node(qe_tot_dual) - 2.5 * node(ue_dual)*node(Te_dual) ))

    ni, ni_dual = rmk.varAndDual("ni",rk.grid,data=nInit)
    Gi_dual,Gi = rmk.varAndDual("Gi",rk.grid,primaryOnDualGrid=True)
    ui_dual,ui = rmk.varAndDual("ui",rk.grid,primaryOnDualGrid=True,derivation=rk.textbook["flowSpeedFromFlux"],derivationArgs=[Gi_dual.name,ni_dual.name])

    electronSpecies.associateVar(ne,ne_dual)
    ionSpecies.associateVar(ni,ni_dual)

    rk.variables.add(f,f_dual,We,ne,ne_dual,Ge_dual,Ge,qe_tot_dual,qe_tot,Te,Te_dual,qe_dual,qe,ue_dual,ue,ni,ni_dual,Gi,Gi_dual,ui,ui_dual)

    E_dual, E = rmk.varAndDual("E",rk.grid,primaryOnDualGrid=True)

    rk.variables.add(E_dual,E)

    cs = rmk.Variable("cs",rk.grid,derivation=rk.textbook["sonicSpeedD+"],derivationArgs=[Te.name,Te.name])

    cs_b = BoundedExtrapolationDerivation("cs_b")(cs)
    cs_b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]

    n_b = BoundedExtrapolationDerivation("n_b")(ni)
    n_b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]

    extrap_LB_ui = DerivationClosure(BoundedExtrapolationDerivation("extrap_LB_ui",lowerBound=cs_b),ui)
    extrap_n = DerivationClosure(BoundedExtrapolationDerivation("extrap_n"),ni)

    boundaryFluxDeriv = extrap_LB_ui*extrap_n

    G_b = rmk.Variable("G_b",rk.grid,derivation=boundaryFluxDeriv,isScalar=True)
    G_b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]
    u_b = rmk.Variable("u_b",rk.grid,derivation=rk.textbook["flowSpeedFromFlux"],derivationArgs=[G_b.name,n_b.name],isScalar=True)
    u_b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]

    rk.variables.add(cs,cs_b,n_b,G_b,u_b)

    #Braginskii heatflux for comparison 

    logLee = rmk.Variable("logLee",rk.grid,isDerived=True) # To be written to by a manipulator later

    nConstGradT = 12*np.pi**1.5*epsilon0**2/np.sqrt(elMass*elCharge)  # Comes from e-i collision time

    sqrt2 = np.sqrt(2)
    ionZ = rk.norms["referenceIonZ"]
    delta = (1 + 65*sqrt2/32 + 433*sqrt2/288 - 23*sqrt2/16)*ionZ + (5629/1152 - 529/128) * ionZ**2  # A30 in Makarov assuming single ion species and 0 mass ratio
    elCondConst = 125*(1+433*sqrt2*ionZ/360)/(32*delta)

    normalizationConst = rk.normTemperature**3.5/(lengthNorm*qNorm)
    kappaClosure = DerivationClosure(SimpleDerivation("kappa",-elCondConst*nConstGradT*normalizationConst,[2.5,-1.0]),Te,logLee)
    gradTClosure = DerivationClosure(rk.textbook["gradDeriv"],Te)
    qt = (kappaClosure*gradTClosure)(Te).rename("qT")
    qt_dual = qt.makeDual()

    q_ratio = varFromNode("qRatio",rk.grid,node(qe_dual)/node(qt_dual),isOnDualGrid=True)
    rk.variables.add(qt,qt_dual,q_ratio,logLee)

    # Set neutral densities

    nn = []
    nn_dual = []
    for i,neut in enumerate(neutralDensList):
        n,n_dual = rmk.varAndDual(neut,rk.grid,units="$10^{19} m^{-3}$")
        nn.append(n)
        neutralSpecies[i].associateVar(n,n_dual)
        nn_dual.append(n_dual)
        rk.variables.add(n,n_dual)
    
    # ### Models

    # ### Electron models

    # ### Advection

    rk.models.add(cm.kinAdvX(f,rk.grid).rename("advf"))

    # ### E-field advection

    rk.models.add(cm.advectionEx(f,E_dual,rk.grid,rk.norms).rename("E-adv"))

    # ### e-e collisions for l=0

    rk.models.add(cm.eeCollIsotropic(f,Te,ne,rk.norms,rk.grid,rk.textbook).rename("e-e0"))

    # ### e-i collisions for odd l with flowing ions

    rk.models.add(cm.flowingIonEIColl(rk.grid,rk.textbook,rk.norms,f,ni,ui_dual,ne_dual,Te_dual,ionSpecies,list(range(2,rk.grid.numH+1,2)),Gi_dual).rename("e-i_odd"))

    # ### e-e collisions for odd l

    rk.models.add(cm.eeCollHigherL(rk.grid,rk.textbook,rk.norms,f,Te_dual,ne_dual,list(range(2,rk.grid.numH+1,2))).rename("e-e_odd"))

    if lMax > 1:
        
        rk.models.add(cm.flowingIonEIColl(rk.grid,rk.textbook,rk.norms,f,ni,ui,ne,Te,ionSpecies,list(range(3,rk.grid.numH+1,2)),Gi).rename("e-i_even"))

        rk.models.add(cm.eeCollHigherL(rk.grid,rk.textbook,rk.norms,f,Te,ne,list(range(3,rk.grid.numH+1,2))).rename("e-e_even"))

    # ### Logical boundary condition

    rk.models.add(cm.logicalBCModel(rk.grid,f,G_b,ne,ne_dual,n_b,evolvedHarmonics=list(range(1, rk.grid.numH + 1, 2))).rename("lbc_right"))

    # ### Heating

    Nh = kwargs.get("Nh", 17)
    Lh = sum(xGridWidths[0:Nh])
    heatingPower = kwargs.get("heatingPower", 1.0) / Lh  # in MW/m^3
    energyInjectionRate = (
        heatingPower * 1e6 * timeNorm / (densNorm * elCharge * tempNorm)
    )
    xProfileEnergy = np.zeros(Nx)
    xProfileEnergy[0:Nh] = energyInjectionRate

    heatingModel = rmk.Model("heating")

    heatingModel.ddt[f] += cm.diffusiveHeatingTerm(rk.grid,rk.norms,f,ne,rk.grid.profile(xProfileEnergy)).rename("diffHeating")

    # ### Heating perturbation

    if "perturbationTimeSignal" in kwargs:
        assert isinstance(
            kwargs.get("perturbationTimeSignal"), rmk.TimeSignalData
        ), "perturbationTimeSignal not of correct type"
        heatingPower = kwargs.get("pertHeatingPower", 10.0) / Lh  # in MW/m^3
        energyInjectionRate = (
            heatingPower * 1e6 * timeNorm / (densNorm * elCharge * tempNorm)
        )
        xProfileEnergy[0:Nh] = energyInjectionRate
        heatingModel.ddt[f] += cm.diffusiveHeatingTerm(rk.grid,rk.norms,f,ne,rk.grid.profile(xProfileEnergy),kwargs.get("perturbationTimeSignal")).rename("diffHeatingPert")

    rk.models.add(heatingModel)

    rk.models.add(cm.standardBaseFluid(ionSpecies,ni,Gi,ui,Te,E))
    rk.models.add(cm.bohmBoundaryModel(ionSpecies,ni,Gi,ui,Te,cs))

    amModel = cm.ampereMaxwell(E_dual,[Gi_dual],[ionSpecies],rk.norms).rename("ampereMaxwell")
    amModel.ddt[E_dual] += cm.ampereMaxwellKineticElTerm(f,rk.norms).rename("kinElTerm")
    rk.models.add(amModel)

    # ### CX friction

    # Ion-neutral CX friction force terms

    inFrictionModel = rmk.Model("inFriction")

    amjuelPath = kwargs.get("amjuelPath", "../data/amjuel.tex")
    if kwargs.get("amjuelCXRate", False):
        logTi = rmk.varFromNode("logTi",rk.grid,isOnDualGrid=True,node=log(rk.norms["eVTemperature"]*node(Te_dual)/2)).onDualGrid()
        
        cxRate = rmk.Variable("cxRate",rk.grid,derivation=ams.AMJUELDeriv1D("cxRateDeriv","3.1.8","H.2",timeNorm=timeNorm,densNorm=densNorm,amjuelFilename=amjuelPath),derivationArgs = [logTi.name],isOnDualGrid=True)

        rk.variables.add(logTi,cxRate)

        inFrictionModel.ddt[Gi_dual] += - nn_dual[0] * cxRate * rmk.DiagonalStencil()(Gi_dual).rename("iFriction_cx")

    else:
        abs_ui = rmk.varFromNode("abs_ui",rk.grid,isOnDualGrid=True,node=rmk.abs(node(ui_dual)))
        rk.variables.add(abs_ui)
        # Use constant low-energy CX cross-sections
        sigmaCx = [3.0e-19, 2**4 * 1.0e-19, 3**4 * 7.0e-20] + [
            i**4 * 6.0e-20 for i in range(4, numNeutrals + 1)
        ]

        normConst = rk.norms["time"] * rk.norms["speed"] * rk.norms["density"]
        for i in range(numNeutrals):

            inFrictionModel.ddt[Gi_dual] += -sigmaCx[i] * normConst * nn_dual[i] * abs_ui * rmk.DiagonalStencil()(Gi_dual).rename("iFriction_cx" + str(i + 1))


    rk.models.add(inFrictionModel)

    # Ground state diffusion and recyling

    neutDynModel = rmk.Model("neutDyn")

    if kwargs.get("amjuelCXRate", False):
        diffCoeff = DerivationClosure(NodeDerivation("amjDiff",np.sqrt(Tn)/2 * node(Te_dual)**(0.5) /(node(ni_dual)*node(cxRate))))
        
        neutDynModel.ddt[nn[0]] += elMass/ionMass * stencils.DiffusionStencil(diffCoeff,diffCoeffOnDualGrid=True)(nn[0]).rename("neutralDiff")

    else:
        sigmaCx = [3.0e-19, 2**4 * 1.0e-19, 3**4 * 7.0e-20] + [
            i**4 * 6.0e-20 for i in range(4, numNeutrals + 1)
        ]

        normConstDiff = np.sqrt(elMass / ionMass) /(rk.norms["density"]*rk.norms["length"])

        # Diffusion coefficient derivation in 1D with neutral temperature Tn and with the cross section used being the low energy charge-exchange cross-seciton
        # NOTE: SOL-KiT has a spurious sqrt(2) factor in the diffusion coefficient, so that is kept here for a consistent comparison
        diffusionDeriv = DerivationClosure(SimpleDerivation("neutDiffD",np.sqrt(Tn) / 2, [-1.0]),ni_dual)
        # Diffusion term
        for i,neut in enumerate(nn):
            neutDynModel.ddt[neut] += normConstDiff/sigmaCx[i] * stencils.DiffusionStencil(diffusionDeriv,diffCoeffOnDualGrid=True)(neut).rename("neutralDiff"+str(i+1))

    neutDynModel.ddt[nn[0]] += stencils.BCDivStencil(ui,cs)(ni).rename("recyclingTerms")

    rk.models.add(neutDynModel)

    # ### CRM

    mbData = crm.CRMModelboundData(rk.grid)

    if kwargs.get("amjuelRates", False):
        amjuelDerivs = ams.generateAMJUELDerivs(ams.amjuelHydrogenAtomDerivs(),rk.norms,amjuelFilename=amjuelPath)

        logne,logTe = ams.AMJUELLogVars(rk.norms,ne,Te)
        rk.variables.add(logne,logTe)
        recombPartEnergy = DerivationClosure(ams.AMJUELDeriv("recombPartEnergy","2.1.8","H.4",timeNorm=timeNorm,densNorm=densNorm,tempNorm=-tempNorm/13.6,amjuelFilename=amjuelPath),logne,logTe)
        recombEnergy = DerivationClosure(amjuelDerivs["recombEnergy"],logne,logTe)
        recombEnergyTotal = recombEnergy + recombPartEnergy
        

        ionizationTransition = crm.DerivedTransition("ionAMJUEL",[electronSpecies,neutralSpecies[0]],[electronSpecies,electronSpecies,ionSpecies],rateDeriv=DerivationClosure(amjuelDerivs["ionPart"],logne,logTe),energyRateDeriv=DerivationClosure(amjuelDerivs["ionEnergy"],logne,logTe))

        mbData.addTransition(ionizationTransition)
        recombTransition = crm.DerivedTransition("recombAMJUEL",[electronSpecies,ionSpecies],[neutralSpecies[0]],rateDeriv = DerivationClosure(amjuelDerivs["recombPart"],logne,logTe),energyRateDeriv=recombEnergyTotal)
        mbData.addTransition(recombTransition)
    else:
        includedJanevTransitions = kwargs.get("includedJanevTransitions", ["ion"])
        crm.addJanevTransitionsToCRMData(
            mbData,
            numNeutrals,
            tempNorm,
            f,
            Te,
            detailedBalanceCSPriority=1,
            processes=includedJanevTransitions
        )

        if kwargs.get("includeSpontEmission", False):
            spontTransDict = crm.readNISTAkiCSV("../data/Aki.csv")
            crm.addHSpontaneousEmissionToCRMData(
                mbData,
                spontTransDict,
                min(numNeutrals, 20),
                min(numNeutrals, 20),
                timeNorm,
                tempNorm,
            )  # NIST data only has a full transition list for n<=20

    # CRM model

    crmModel = rmk.Model("CRM")

    crmModel.setModelboundData(mbData)
    
    # Add term generator responsible for buildling CRM model for neutrals and ions

    crmModel.addTermGenerator(crm.CRMTermGenerator(
        "crmTermGen",
        evolvedSpecies=[ionSpecies] + neutralSpecies
    ))

    ionInds = []
    recomb3bInds = []
    if kwargs.get("amjuelRates", False):
        # Cooling terms due to radiation
        ionInds = mbData.getTransitionIndices("ionAMJUEL")

        crmModel.ddt[f] += nn[0] * (mbData["rate2index"+str(ionInds[0])] @  cm.dvEnergyTerm(rk.grid,f).rename("ionizationCooling"))

        recomb3bInds = mbData.getTransitionIndices(
            "recombAMJUEL"
        )

        crmModel.ddt[f] += ni * (mbData["rate2index"+str(recomb3bInds[0])] @  cm.dvEnergyTerm(rk.grid,f).rename("recombCooling"))

    else:

        janevMap = {"ex":"JanevEx",
                    "ion":"JanevIon",
                    "deex":"JanevDeex",
                    "recomb3b":"JanevRecomb3b"}
            
        for key in janevMap:
            
            if key in includedJanevTransitions:
                inds = mbData.getTransitionIndices(janevMap[key])

                if key == "ion":
                    ionInds+=inds 
                if key == "recomb3b":
                    recomb3bInds+=inds 

                crmModel.addTermGenerator(crm.CRMBoltzTermGenerator(key+"CRME",f,1,inds,mbData))

                crmModel.addTermGenerator(crm.CRMBoltzTermGenerator(key+"CRME2",f,2,inds,mbData,associatedVarIndex=2))

                crmModel.addTermGenerator(crm.CRMBoltzTermGenerator(key+"CRMA",f,1,inds,mbData,absorptionTerms=True))

    # Add secondary electron sources/sinks due to ionization and recombination

    secElInds = ionInds + recomb3bInds

    crmModel.addTermGenerator(crm.CRMSecElTermGenerator("secElCRM",f,secElInds))

    rk.models.add(crmModel)

    # ### Integrator options

    integrator = rmk.BDEIntegrator("BDE",absTol=100.0,convergenceVars=[ne,ni,Ge_dual,Gi_dual,We,nn[0]],maxNonlinIters=50,internalStepControl=True,nonlinTol=1e-10)
    integrationStep = rmk.IntegrationStep("BE",integrator)
    integrationStep.add(rk.models) 
    rk.integrationScheme = rmk.IntegrationScheme(dt=rmk.Timestep(kwargs.get("initialTimestep",0.1)*Te**(1.5)/ne),steps=integrationStep)

    rk.integrationScheme.setOutputPoints(kwargs.get("outputPoints",[10.0,20.0,30.0,40.0,50.0]))
    rk.IOContext.setRestartOptions(save=True) 

    if kwargs.get("initFromFluidRun", False):
        rk.IOContext.setHDF5InputOptions(inputFile=kwargs.get("hdf5InputFile", "ReMKiT1DVarInput"),inputVars=[f,ni,Gi_dual,E_dual]+nn)

    rk.addTermDiagnostics(Gi_dual,ni,E_dual,f)

    # rk.variables.add(rmk.Variable("gammaRight",rk.grid,isScalar=True,scalarHostProcess=rk.mpiContext.fluidProcs[-1]))
    # rk.manipulators.add(rmk.MBDataExtractor("gammaExtRight",rk.models["lbc_right"],rk.models["lbc_right"].mbData["gamma"],rk.variables["gammaRight"]))

    # rk.variables.add(rmk.Variable("TRight",rk.grid,isScalar=True,scalarHostProcess=rk.mpiContext.fluidProcs[-1]))
    # rk.manipulators.add(rmk.MBDataExtractor("tempExtRight",rk.models["lbc_right"],rk.models["lbc_right"].mbData["shTemp"],rk.variables["TRight"]))

    rk.manipulators.add(rmk.MBDataExtractor("logLeeExtract",rk.models["e-e0"],logLee))

    rk.setPETScOptions(
        cliOpts="-pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1",
        kspSolverType="gmres",
    )

    return rk
