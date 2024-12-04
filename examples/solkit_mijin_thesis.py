import numpy as np

import RMK_support as rmk 
import RMK_support.common_models as cm
import RMK_support.crm_support as crm
import RMK_support.amjuel_support as ams
from RMK_support import node, varFromNode, log
import RMK_support.stencils as stencils
from RMK_support.derivations import NodeDerivation, DerivationClosure,SimpleDerivation,BoundedExtrapolationDerivation

def generatorSKThesis(**kwargs) -> rmk.RMKContext:
    """Generate a context containing a SOL-KiT run similar to the Mijin thesis fluid version. Neutrals are diffusive, the heating profile is a step function upstream, recycling is at 100%, the neutrals have a fixed temperature (with a spurious sqrt(2) factor to match SOL-KiT's diffusion) and Ti=Te. The CRM can either use a SOL-KiT-like Janev cross-section calculation or AMJUEL rates, and similarly for CX. Both the spatial and velocity grid widths are geometric. If not initializing from a previous run, the initial profiles are based on the 2-Point Model, with 100% ionization fraction.

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
    outputPoints (List[float]): List of used output points. Defaults to [1500,3000,4500,6000,7500,9000].

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
        "hdf5Filepath", "./RMKOutput/RMK_SK_comp_staggered_thesis/"
    )
    rk.IOContext = rmk.IOContext(jsonFilepath=kwargs.get("jsonFilepath", "./config.json"),HDF5Dir=hdf5Filepath)

    rk.mpiContext = rmk.MPIContext(kwargs.get("mpiProcs", 8))

    tempNorm = rk.norms["eVTemperature"]
    densNorm = rk.norms["density"]

    timeNorm = rk.norms["time"]

    dx0 = kwargs.get("dx0", 0.27)
    dxN = kwargs.get("dxN", 0.0125)
    Nx = kwargs.get("Nx", 128)
    xGridWidths = np.geomspace(dx0, dxN, Nx)
    L = sum(xGridWidths)
    dv0 = kwargs.get("dv0", 0.05)
    dvN = kwargs.get("dvN", 0.4)
    Nv = kwargs.get("Nv", 80)
    vGridWidths = np.geomspace(dv0, dvN, Nv)
    lMax = kwargs.get("lmax", 0)
    rk.grid = rmk.Grid(
        xGridWidths,
        vGridWidths,
        lMax,
        interpretXGridAsWidths=True,
        interpretVGridAsWidths=True,
        isLengthInMeters=True,
    )

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

    # Two-point model initialization

    Tu = kwargs.get("Tu", 20) / tempNorm  # upstream temperature
    Td = kwargs.get("Td", 5) / tempNorm  # downstream temperature

    TInit = (Tu ** (7 / 2) - (Tu ** (7 / 2) - Td ** (7 / 2)) * rk.grid.xGrid / L) ** (2 / 7)

    nu = kwargs.get("nu", 0.8)  # upstream density

    nInit = nu * Tu / TInit

    WInit = 3 * nInit * TInit / 2
    # Set conserved variables in container

    ne,ne_dual = rmk.varAndDual("ne",rk.grid,data=nInit,units="$10^{19} m^{-3}$")
    electronSpecies.associateVar(ne)
    ni,ni_dual = rmk.varAndDual("ni",rk.grid,data=nInit,units="$10^{19} m^{-3}$")
    ionSpecies.associateVar(ni)

    rk.variables.add(ne,ne_dual,ni,ni_dual)
    Ge_dual,Ge = rmk.varAndDual("Ge",rk.grid,primaryOnDualGrid=True)
    Gi_dual,Gi = rmk.varAndDual("Gi",rk.grid,primaryOnDualGrid=True)

    rk.variables.add(Ge_dual,Ge,Gi_dual,Gi)

    We, We_dual = rmk.varAndDual("We",rk.grid,units="$10^{20} eV m^{-3}$",data=WInit)
    Te,Te_dual = rmk.varAndDual("Te",rk.grid,units="$10eV$",isStationary=True,data=TInit)

    rk.variables.add(We,We_dual,Te,Te_dual)

    # Set heat fluxes

    qe_dual,qe = rmk.varAndDual("qe",rk.grid,primaryOnDualGrid=True,isStationary=True)

    rk.variables.add(qe_dual,qe)

    # Set E field

    E_dual, E = rmk.varAndDual("E",rk.grid,primaryOnDualGrid=True)

    rk.variables.add(E_dual,E)
    # Set derived fluid quantities

    rk.textbook = rmk.Textbook(rk.grid,tempDerivSpeciesIDs=[0])

    ue_dual,ue = rmk.varAndDual("ue",rk.grid,primaryOnDualGrid=True,derivation=rk.textbook["flowSpeedFromFlux"],derivationArgs=[Ge_dual.name,ne_dual.name])
    ui_dual,ui = rmk.varAndDual("ui",rk.grid,primaryOnDualGrid=True,derivation=rk.textbook["flowSpeedFromFlux"],derivationArgs=[Gi_dual.name,ni_dual.name])

    rk.variables.add(ue_dual,ue,ui_dual,ui)

    cs = rmk.Variable("cs",rk.grid,derivation=rk.textbook["sonicSpeedD+"],derivationArgs=[Te.name,Te.name])

    cs_b = BoundedExtrapolationDerivation("cs_b")(cs)
    cs_b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]

    n_b = BoundedExtrapolationDerivation("n_b")(ne)
    n_b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]

    extrap_LB_ui = DerivationClosure(BoundedExtrapolationDerivation("extrap_LB_ui",lowerBound=cs_b),ui)
    extrap_n = DerivationClosure(BoundedExtrapolationDerivation("extrap_n"),ni)

    boundaryFluxDeriv = extrap_LB_ui*extrap_n

    G_b = rmk.Variable("G_b",rk.grid,derivation=boundaryFluxDeriv,isScalar=True)
    G_b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]
    u_b = rmk.Variable("u_b",rk.grid,derivation=rk.textbook["flowSpeedFromFlux"],derivationArgs=[G_b.name,n_b.name],isScalar=True)
    u_b.scalarHostProcess = rk.mpiContext.fluidProcs[-1]

    rk.variables.add(cs,cs_b,n_b,G_b,u_b)

    if kwargs.get("useKineticCorrections", False):

        gamma = rmk.Variable("gammaRight",rk.grid,isScalar=True,isDerived=True,data=kwargs.get("fixedKineticGamma", 5.0) * np.ones(1),scalarHostProcess=rk.mpiContext.fluidProcs[-1])

        qRatio = rmk.Variable("qRatio",rk.grid,isDerived=True,isOnDualGrid=True,data=kwargs.get("fixedQRatio", np.ones(Nx)))

        rk.variables.add(gamma,qRatio)

        loadVars = [
            ne,
            ni,
            Ge_dual,
            Gi_dual,
            We,
            Te,
            qe,
            E_dual,
        ] + neutralDensList
        if "fixedQRatio" not in kwargs:
            loadVars.append(qRatio)
        if "fixedKineticGamma" not in kwargs:
            loadVars.append(gamma)
        rk.IOContext.setHDF5InputOptions(inputFile=kwargs.get("hdf5InputFile", "ReMKiT1DVarInput"),
                                         inputVars=loadVars)

    else:
        gamma = rmk.Variable("gammaRight",rk.grid,isScalar=True,derivation=rk.textbook["rightElectronGamma"],derivationArgs=[Te.name,Te.name],scalarHostProcess=rk.mpiContext.fluidProcs[-1])
        rk.variables.add(gamma)

    # Set neutral densities

    nn = []
    nn_dual = []
    for i,neut in enumerate(neutralDensList):
        n,n_dual = rmk.varAndDual(neut,rk.grid,units="$10^{19} m^{-3}$")
        nn.append(n)
        neutralSpecies[i].associateVar(n)
        nn_dual.append(n_dual)
        rk.variables.add(n,n_dual)

    # We need a distribution function to calculate rates from cross-sections built into the code
    fInit = np.zeros([rk.grid.numX(), rk.grid.numH(), rk.grid.numV()])
    for i in range(rk.grid.numX()):
        fInit[i, rk.grid.getH(0) - 1, :] = (
            np.pi ** (-1.5) * TInit[i] ** (-1.5) * nInit[i] * np.exp(-rk.grid.vGrid**2 / TInit[i])
        )

    f_unscaled = rmk.Variable("f_unscaled",rk.grid,isDistribution=True,derivation=rk.textbook["maxwellianDistribution"],derivationArgs=[Te.name,ne.name])
    ne_numerical = rmk.Variable("ne_numerical",rk.grid,units="$10^{19} m^{-3}$",derivation=rk.textbook["densityMoment"],derivationArgs=[f_unscaled.name])

    ne_rescaled = varFromNode("ne_rescaled",rk.grid,node=node(ne)**2/node(ne_numerical))
    f = rmk.Variable("f",rk.grid,isDistribution=True,derivation=rk.textbook["maxwellianDistribution"],derivationArgs=[Te.name,ne_rescaled.name])

    rk.variables.add(f_unscaled,ne_numerical,ne_rescaled,f)

    #using base models to add the bulk of the equations

    rk.models.add(cm.standardBaseFluid(electronSpecies,ne,Ge,ue,Te,E,We,qe))
    rk.models.add(cm.standardBaseFluid(ionSpecies,ni,Gi,ui,Te,E))
    rk.models.add(cm.bohmBoundaryModel(electronSpecies,ne,Ge,ue,Te,cs,We,gamma))
    rk.models.add(cm.bohmBoundaryModel(ionSpecies,ni,Gi,ui,Te,cs))
    
    rk.models.add(cm.ampereMaxwell(E_dual,[Ge_dual,Gi_dual],[electronSpecies,ionSpecies],rk.norms))

    # Derived from expressions in [Makarov et al](https://doi.org/10.1063/5.0047618)

    ionZ = 1
    sqrt2 = np.sqrt(2)

    delta = (
        1
        + (65 * sqrt2 / 32 + 433 * sqrt2 / 288 - 23 * sqrt2 / 16) * ionZ
        + (5629 / 1152 - 529 / 128) * ionZ**2
    )  # A30 in Makarov assuming single ion species and 0 mass ratio

    thermFrictionConst = (
        25 * sqrt2 * ionZ * (1 + 11 * sqrt2 * ionZ / 30) / (16 * delta)
    )  # A50

    elCondConst = 125 * (1 + 433 * sqrt2 * ionZ / 360) / (32 * delta)

    # Braginskii heat fluxes

    braginskii = rmk.Model("braginskii")

    mbData = rmk.VarlikeModelboundData()
    logLei = rmk.Variable("logLei",rk.grid,derivation=rk.textbook["logLeiD+"],derivationArgs=[Te_dual.name,ne_dual.name],isOnDualGrid=True)
    mbData.addVar(logLei)
    braginskii.setModelboundData(mbData)


    nConstGradT = - 12 * np.pi**1.5 * epsilon0**2 / np.sqrt(elMass * elCharge) * elCondConst  *rk.norms["eVTemperature"]**3.5 / (rk.norms["length"]*rk.norms["heatFlux"])# Comes from e-i collision time

    # Variable data

    if kwargs.get("useKineticCorrections", False):

        braginskii.ddt[qe_dual] += nConstGradT * Te_dual**(2.5) * qRatio * (logLei**(-1) @ stencils.StaggeredGradStencil()(Te)).rename("qGradT")

    else:
        braginskii.ddt[qe_dual] += nConstGradT * Te_dual**(2.5) * (logLei**(-1) @ stencils.StaggeredGradStencil()(Te)).rename("qGradT")


    # thermoforce friction 
    normConst = elCharge * thermFrictionConst * rk.norms["eVTemperature"] / (rk.norms["speed"]**2)
    braginskii.ddt[Ge_dual] += -normConst/elMass * ne_dual *stencils.StaggeredGradStencil()(Te).rename("electronGradTFriction")
    braginskii.ddt[Gi_dual] += normConst/ionMass * ne_dual *stencils.StaggeredGradStencil()(Te).rename("ionGradTFriction")

    rk.models.add(braginskii)

    Nh = kwargs.get("Nh", 17)
    Lh = sum(xGridWidths[0:Nh])
    heatingPower = kwargs.get("heatingPower", 1.0) / Lh  # in MW/m^3
    energyInjectionRate = (
        heatingPower * 1e6 * timeNorm / (densNorm * elCharge * tempNorm)
    )
    xProfileEnergy = np.zeros(Nx)
    xProfileEnergy[0:Nh] = energyInjectionRate

    # Energy source model

    energySourceModel = rmk.Model("energySource")

    energySourceModel.ddt[We] += cm.simpleSourceTerm(We,rk.grid.profile(xProfileEnergy))

    # Perturbation

    if "perturbationTimeSignal" in kwargs:
        assert isinstance(
            kwargs.get("perturbationTimeSignal"),rmk.TimeSignalData
        ), "perturbationTimeSignal not of correct type"
        heatingPower = kwargs.get("pertHeatingPower", 10.0) / Lh  # in MW/m^3
        energyInjectionRate = (
            heatingPower * 1e6 * timeNorm / (densNorm * elCharge * tempNorm)
        )
        xProfileEnergy[0:Nh] = energyInjectionRate
        
        energySourceModel.ddt[We] += cm.simpleSourceTerm(We,rk.grid.profile(xProfileEnergy),timeSignal=kwargs.get("perturbationTimeSignal"))

    rk.models.add(energySourceModel)

    inFrictionModel = rmk.Model("inFriction")

    amjuelPath = kwargs.get("amjuelPath", "../data/amjuel.tex")
    if kwargs.get("amjuelCXRate", False):
        logTi = rmk.varFromNode("logTi",rk.grid,isOnDualGrid=True,node=log(rk.norms["eVTemperature"]*node(Te_dual)/2))
        
        cxRate = rmk.Variable("cxRate",rk.grid,derivation=ams.AMJUELDeriv1D("cxRateDeriv","3.1.8","H.2",timeNorm=timeNorm,densNorm=densNorm,amjuelFilename=amjuelPath),derivationArgs = [logTi.name])

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
        diffCoeff = DerivationClosure(NodeDerivation("amjDiff",np.sqrt(Tn)/2 * node(Te_dual**(0.5)) /(node(ni_dual)*node(cxRate))),ni_dual,Te_dual,cxRate)
        
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
    # ### CRM density and energy evolution

    mbData = crm.CRMModelboundData(rk.grid)

    if kwargs.get("amjuelRates", False):
        amjuelDerivs = ams.generateAMJUELDerivs(ams.amjuelHydrogenAtomDerivs(),rk.norms,amjuelFilename=amjuelPath)

        logne,logTe = ams.AMJUELLogVars(rk.norms,ne,Te)
        rk.variables.add(logne,logTe)
        recombPartEnergy = DerivationClosure(ams.AMJUELDeriv("recombPartEnergy","2.1.8","H.4",timeNorm=timeNorm,densNorm=densNorm,tempNorm=-tempNorm/13.6,amjuelFilename=amjuelPath),logne,logTe)
        recombPart = DerivationClosure(amjuelDerivs["recombEnergy"],logne,logTe)
        recombEnergyTotal = recombPart + recombPartEnergy
        

        ionizationTransition = crm.DerivedTransition("ionAMJUEL",[electronSpecies,neutralSpecies[0]],[electronSpecies,electronSpecies,ionSpecies],rateDeriv=DerivationClosure(amjuelDerivs["ionPart"],logne,logTe),energyRateDeriv=DerivationClosure(amjuelDerivs["ionEnergy"],logne,logTe))

        mbData.addTransition(ionizationTransition)
        recombTransition = crm.DerivedTransition("recombAMJUEL",[electronSpecies,ionSpecies],[neutralSpecies[0]],rateDeriv = DerivationClosure(amjuelDerivs["recombPart"],logne,logTe),energyRateDeriv=recombEnergyTotal)
        mbData.addTransition(recombTransition)
        ionTransPrefix = "ionAMJUEL"
    else:
        includedJanevTransitions = kwargs.get("includedJanevTransitions", ["ion"])
        crm.addJanevTransitionsToCRMData(
            mbData,
            numNeutrals,
            tempNorm,
            f,
            Te,
            detailedBalanceCSPriority=1,
            processes=includedJanevTransitions,
            lowestCellEnergy=rk.grid.vGrid[0]**2
        )

        if kwargs.get("includeSpontEmission", False):
            spontTransDict = crm.readNISTAkiCSV("Aki.csv")
            crm.addHSpontaneousEmissionToCRMData(
                mbData,
                spontTransDict,
                min(numNeutrals, 20),
                min(numNeutrals, 20),
                timeNorm,
                tempNorm,
            )  # NIST data only has a full transition list for n<=20

        ionTransPrefix = "JanevIon"
    # CRM model

    crmModel = rmk.Model("CRM")

    crmModel.setModelboundData(mbData)

    # Add ionization term generator for ions
    ionInds = mbData.getTransitionIndices(ionTransPrefix)
    crmModel.addTermGenerator(crm.CRMTermGenerator(
        "crmTermGenIonIonization",
        implicitGroups=[2],
        evolvedSpecies=[ionSpecies],
        includedTransitionIndices=ionInds,
    ))

    otherIndices = list(range(1, len(mbData.transitionTags) + 1))
    for ind in ionInds:
        otherIndices.remove(ind)
    if len(otherIndices) > 0:
        crmModel.addTermGenerator(crm.CRMTermGenerator(
            "crmTermGenIonOther",
            implicitGroups=[4],
            evolvedSpecies=[ionSpecies],
            includedTransitionIndices=otherIndices,
        ))

    # Add all other terms for other particle species
    crmModel.addTermGenerator(crm.CRMTermGenerator(
        "crmTermGen",
        evolvedSpecies=[electronSpecies] + neutralSpecies
    ))

    crmModel.addTermGenerator(crm.CRMElEnergyTermGenerator(
        "crmElEnergyGen",
        We, implicitGroups=[3]
    ))

    rk.models.add(crmModel)

    integrator = rmk.BDEIntegrator("BDE",absTol=100.0,convergenceVars=[ne,ni,Ge_dual,Gi_dual,We,Te,nn[0]],internalStepControl=True)
    integrationStep = rmk.IntegrationStep("BE",integrator)
    integrationStep.add(rk.models) 
    rk.integrationScheme = rmk.IntegrationScheme(dt=rmk.Timestep(kwargs.get("initialTimestep",0.1)*Te**(1.5)/ne),steps=integrationStep)

    rk.integrationScheme.setOutputPoints(kwargs.get("outputPoints",[1500,3000,4500,6000,7500,9000]))
    rk.IOContext.setRestartOptions(save=True) 

    rk.addTermDiagnostics(We,ne,Ge_dual,Gi_dual,nn[0],E_dual,qe_dual)

    
    rk.variables.add(rmk.Variable("ionsource",rk.grid, isDerived=True))
    rk.manipulators.add(rmk.GroupEvaluator("ionsource",crmModel,2,rk.variables["ionsource"]))

    if kwargs.get("amjuelRates", False):
        rk.variables.add(rmk.Variable("recombSource",rk.grid, isDerived=True))
        rk.manipulators.add(rmk.GroupEvaluator("recombSource",crmModel,4,rk.variables["recombSource"]))

    rk.variables.add(rmk.Variable("elInelEnergyLossIon",rk.grid, isDerived=True))
    rk.manipulators.add(rmk.GroupEvaluator("elInelEnergyLossIon",crmModel,3,rk.variables["elInelEnergyLossIon"]))

    rk.setPETScOptions(
        cliOpts="-pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1",
        kspSolverType="gmres",
    )

    return rk
