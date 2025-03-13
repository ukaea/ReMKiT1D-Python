import sys

sys.path.append("./example_sources")
import epperlein_short_test  # type: ignore
import solkit_mijin_thesis  # type: ignore
import solkit_mijin_thesis_kin  # type: ignore
import os
import numpy as np
import pytest


@pytest.mark.latex
def test_es():

    kLambda = 2.0  # Braginskii k * mfp - To reproduce points in Figure 17 in the paper use values from np.geomspace(0.5e-2,2,8)
    k = kLambda / (3 * np.sqrt(np.pi) / (4 * np.sqrt(2)))
    Nx = 128  # number of spatal grids
    dx = 2 * np.pi / (k * Nx)
    dt = 0.01  # time step in e-i collional times
    ionZ = 1.0  # ion charge
    Nt = 3000  # number of timesteps
    lmax = 1  # highest resolved harmonic - change to reproduce points in Figure 17 in the paper

    # these tests only check for latex construction errors
    with pytest.warns(UserWarning):
        rk = epperlein_short_test.esTestGenerator(
            dx=dx,
            Nx=Nx,
            lmax=lmax,
            ionZ=ionZ,
            mpiProcsX=16,
            mpiProcsH=1,
            hdf5Filepath="./RMKOutput/RMK_ES_test/",
            initialTimestep=dt,
            Nt=Nt,
        )

        rk.setPETScOptions(
            cliOpts="-pc_type bjacobi -sub_pc_factor_shift_type nonzero",
            kspSolverType="gmres",
        )

        rk.generatePDF("Epperlein-Short Test")

        rk.writeConfigFile()

    os.remove(os.path.curdir + "/Epperlein-Short_Test.pdf")
    os.remove(os.path.curdir + "/config.json")


@pytest.mark.latex
def test_solkit_fluid_janev():

    dx0 = 0.13542325129584085e0 * 8.5 / 2.5 * 10.18 / 9.881556569543156
    dxN = 0.13542325129584085e0 * 0.5 / 2.5 * 10.18 / 9.881556569543156
    heatingPower = 3.5464790894703255

    # these tests only check for latex construction errors

    rk = solkit_mijin_thesis.generatorSKThesis(
        dx0=dx0 / 4,
        dxN=dxN / 4,
        Nx=256,
        Nh=17,
        lmax=1,
        mpiProcs=16,
        initialTimestep=2.0,
        nu=0.8 / 1.09345676,
        heatingPower=heatingPower,
        includedJanevTransitions=["ex", "deex", "ion", "recomb3b"],
        numNeutrals=10,
    )

    rk.setPETScOptions(
        cliOpts="-pc_type bjacobi -sub_pc_factor_shift_type nonzero",
        kspSolverType="gmres",
    )

    rk.generatePDF("SOL-KiT Fluid Test")

    rk.writeConfigFile()

    os.remove(os.path.curdir + "/SOL-KiT_Fluid_Test.pdf")
    os.remove(os.path.curdir + "/config.json")


@pytest.mark.latex
def test_solkit_kinetic_janev():

    dx0 = 0.13542325129584085e0 * 8.5 / 2.5 * 10.18 / 9.881556569543156
    dxN = 0.13542325129584085e0 * 0.5 / 2.5 * 10.18 / 9.881556569543156
    heatingPower = 3.5464790894703255

    with pytest.warns(UserWarning):

        rk = solkit_mijin_thesis_kin.generatorSKThesisKin(
            dx0=dx0,
            mpiProcsX=16,
            dxN=dxN,
            Nx=64,
            Nh=9,
            lmax=1,
            numNeutrals=5,
            initialTimestep=0.5,
            heatingPower=heatingPower,
            includedJanevTransitions=["ex", "deex", "ion", "recomb3b"],
            nu=0.8 / 1.09345676,
        )

        rk.generatePDF("SOL-KiT Kinetic Test")

        rk.writeConfigFile()

    os.remove(os.path.curdir + "/SOL-KiT_Kinetic_Test.pdf")
    os.remove(os.path.curdir + "/config.json")
