from RMK_support.grid import Grid
import RMK_support.crm_support as crm
import RMK_support.sk_normalization as skn
import numpy as np
import pytest


def test_normalization():

    elCharge = skn.elCharge
    elMass = skn.elMass
    epsilon0 = skn.epsilon0

    Te = 2.0
    ne = 0.8
    Z = 1.0
    removeLogLeiDiscontinuity = True

    assert skn.velNorm(Te) == np.sqrt(2 * elCharge * Te / elMass)
    assert skn.velNorm(0.0) == 0.0

    assert skn.heatFluxNorm(Te, ne) == elMass * ne * skn.velNorm(Te) ** 3 / 2
    assert skn.heatFluxNorm(0.0, ne) == 0.0

    expectedOutput = {
        "eVTemperature": Te,
        "density": ne,
        "referenceIonZ": Z,
        "time": skn.collTimeei(Te, ne, Z, removeLogLeiDiscontinuity),
        "velGrid": skn.velNorm(Te),
        "speed": skn.velNorm(Te),
        "EField": skn.eFieldNorm(Te, ne, Z, removeLogLeiDiscontinuity),
        "heatFlux": skn.heatFluxNorm(Te, ne),
        "crossSection": skn.crossSectionNorm(Te, ne, Z, removeLogLeiDiscontinuity),
        "length": skn.lenNorm(Te, ne, Z, removeLogLeiDiscontinuity),
    }

    assert skn.calculateNorms(Te, ne, Z, removeLogLeiDiscontinuity) == expectedOutput


def test_logLei():

    ne = 1.0
    Z = 1.0

    # NRL Plasma Formulary definition of logLei has jump discontinuity at Te = 10 * (Z^2) eV:
    T_jump = Z**2 * 10  # [eV]

    x = np.linspace(0.5 * T_jump, 1.5 * T_jump, 100)

    # If removeDiscontinuity=True, logLei increases monotonically across this range in Te
    removelogLeiDiscontinuity = True

    assert np.all(
        np.diff(
            np.array([skn.logLei(Te, ne, Z, removelogLeiDiscontinuity) for Te in x])
        )
        > 0
    )

    # ...otherwise there is a jump discontinuity
    removelogLeiDiscontinuity = False

    assert (
        np.all(
            np.diff(
                np.array([skn.logLei(Te, ne, Z, removelogLeiDiscontinuity) for Te in x])
            )
            > 0
        )
    ) == False
