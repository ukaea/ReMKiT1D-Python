import pytest
import sys

sys.path.append('../')
import anisotropic_plasma.anisotropy as aniso

def test_phi():
    assert aniso.phi(0.5) == pytest.approx(0.87041975)

def testX():
    assert aniso.X(1.5) == pytest.approx(0.5)
    assert aniso.X(1.001) == pytest.approx(0.001)

def test_K_LMN():
    # Tests for X ~ 1:
    assert aniso.K_LMN(1.5,"200") == pytest.approx(0.61125925)
    assert aniso.K_LMN(1.5,"002") == pytest.approx(0.51832099)
    assert aniso.K_LMN(1.5,"220") == pytest.approx(0.11796297)
    assert aniso.K_LMN(1.5,"202") == pytest.approx(0.09293826)
    assert aniso.K_LMN(1.5,"004") == pytest.approx(0.22162965)
    assert aniso.K_LMN(1.5,"420") == pytest.approx(0.09919444)
    assert aniso.K_LMN(1.5,"222") == pytest.approx(0.02502471)
    assert aniso.K_LMN(1.5,"204") == pytest.approx(0.05718513)
    assert aniso.K_LMN(1.5,"006") == pytest.approx(0.21925936)

    # Tests for X << 1:
    assert aniso.K_LMN(1.001,"200") == pytest.approx(0.66653333,rel=1e-2)
    assert aniso.K_LMN(1.001,"002") == pytest.approx(0.66626667,rel=1e-2)
    assert aniso.K_LMN(1.001,"220") == pytest.approx(0.13329524,rel=1e-2)
    assert aniso.K_LMN(1.001,"202") == pytest.approx(0.13321905,rel=1e-2)
    assert aniso.K_LMN(1.001,"004") == pytest.approx(0.39942857,rel=1e-2)
    assert aniso.K_LMN(1.001,"420") == pytest.approx(0.11424761,rel=1e-2)
    assert aniso.K_LMN(1.001,"222") == pytest.approx(0.03805714,rel=1e-2)
    assert aniso.K_LMN(1.001,"204") == pytest.approx(0.11409524,rel=1e-2)
    assert aniso.K_LMN(1.001,"006") == pytest.approx(0.57009524,rel=1e-2)

    with pytest.raises(ValueError) as exception:
        aniso.K_LMN(1.5,"000")

def test_psi():
    assert aniso.psi(1.5) == pytest.approx(0.35925932)
    assert aniso.psi(1.001) == pytest.approx(0.26687567,rel=1e-2)

def test_xi():
    assert aniso.xi(1.5) == pytest.approx(0.1930371)
    assert aniso.psi(1.001) == pytest.approx(0.2664762,rel=1e-2)

def test_cPerp():
    assert aniso.cPerp(1,1,1,1,1.5,1.5,True) == pytest.approx(-7.3573389)
    assert aniso.cPerp(1,1,1,1,1.5,1.5,False) == pytest.approx(-11.7366083)

def test_cPar():
    assert aniso.cPar(1,1,1,1.5,1.5,True) == pytest.approx(0.64918768)
    assert aniso.cPar(1,1,1,1.5,1.5,False) == pytest.approx(0.71851864)

def test_ePerp():
    assert aniso.ePerp(1,1,1,1.5,1.5,True) == pytest.approx(2.70401136)
    assert aniso.ePerp(1,1,1,1.5,1.5,False) == pytest.approx(2.3164452)

def test_ePar():
    assert aniso.ePar(1,1,1,1,1.5,1.5,True) == pytest.approx(-22.95346389)
    assert aniso.ePar(1,1,1,1,1.5,1.5,False) == pytest.approx(-13.64088966)
