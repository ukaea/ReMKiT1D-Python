import numpy as np
import pytest

import epperlein_short_test  # type: ignore
import solkit_mijin_thesis  # type: ignore
import solkit_mijin_thesis_kin  # type: ignore


@pytest.mark.latex
def test_es(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

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

        rk.generatePDF("Epperlein-Short Test")

        rk.writeConfigFile()

    assert (tmp_path / "Epperlein-Short_Test.pdf").is_file()
    assert (tmp_path / "config.json").is_file()


@pytest.mark.latex
def test_solkit_fluid_janev(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dx0 = 0.13542325129584085e0 * 8.5 / 2.5 * 10.18 / 9.881556569543156
    dxN = 0.13542325129584085e0 * 0.5 / 2.5 * 10.18 / 9.881556569543156
    heatingPower = 3.5464790894703255

    # these tests only check for latex construction errors
    with pytest.warns(UserWarning):

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

    rk.generatePDF("SOL-KiT Fluid Test")

    rk.writeConfigFile()

    assert (tmp_path / "SOL-KiT_Fluid_Test.pdf").is_file()
    assert (tmp_path / "config.json").is_file()


@pytest.mark.latex
def test_solkit_kinetic_janev(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

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

    assert (tmp_path / "SOL-KiT_Kinetic_Test.pdf").is_file()
    assert (tmp_path / "config.json").is_file()


def test_example_generator_regressions():
    with pytest.warns(UserWarning):
        es = epperlein_short_test.esTestGenerator(Nx=8, Nv=10, Nt=300, mpiProcsX=2)
        es_cfg = es.dict()
    assert isinstance(es_cfg["timeloop"]["fixedSaveInterval"], int)
    assert es_cfg["timeloop"]["fixedSaveInterval"] == 10

    with pytest.warns(UserWarning):
        sk_fluid = solkit_mijin_thesis.generatorSKThesis(
            Nx=8, Nv=10, Nh=2, mpiProcs=2, numNeutrals=2, useKineticCorrections=True
        )
        fluid_cfg = sk_fluid.dict()
    assert "n1" in fluid_cfg["HDF5"]["inputVars"]
    assert "n2" in fluid_cfg["HDF5"]["inputVars"]

    with pytest.warns(UserWarning):
        sk_kin_default = solkit_mijin_thesis_kin.generatorSKThesisKin(
            Nx=8, Nv=10, Nh=2, mpiProcsX=2
        )
        sk_kin_hot = solkit_mijin_thesis_kin.generatorSKThesisKin(
            Nx=8, Nv=10, Nh=2, mpiProcsX=2, Tu=40.0, Td=10.0
        )
    assert np.all(
        sk_kin_hot.variables["Te"].values > sk_kin_default.variables["Te"].values
    )
