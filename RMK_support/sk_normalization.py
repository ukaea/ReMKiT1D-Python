import numpy as np

elCharge = 1.60218e-19
elMass = 9.10938e-31
epsilon0 = 8.854188e-12  # vacuum permittivity


def velNorm(Te: float) -> float:
    """Electron thermal velocity as used for SOL-KiT-like normalization.

    Args:
        Te (float): Electron temperature in eVs.

    Returns:
        float: sqrt(2*e*Te/me)
    """

    return np.sqrt(2 * elCharge * Te / elMass)


def logLei(Te: float, ne: float, Z: float, removeDiscontinuity=False) -> float:
    """Calculate Coulomb logarithm for electron-ion collisions (NRL Formulary 2013 page 34 equation b). Assumes Te > TiZm_e/m_i

    Args:
        Te (float): Electron temperature in eV
        ne (float): Electron density in m^{-3}
        Z (float): Ion charge
        removeDiscontinuity (bool, optional): If true will remove the discontinuity at 10eV by shifting the cut-off to e**2 eV. Defaults to False.

    Returns:
        float: e-i Coulomb logarithm
    """
    cutoff = 10
    if removeDiscontinuity:
        cutoff = np.exp(2)
    if Te < cutoff * Z**2:
        return 23.0 - np.log(np.sqrt(ne * 1e-6) * Z * Te ** (-3 / 2))

    return 24.0 - np.log(np.sqrt(ne * 1e-6) / Te)


def collTimeei(Te: float, ne: float, Z: float, removeDiscontinuity=False) -> float:
    """Calculate electron-ion collision time in seconds. This is not the Braginskii time, which can be calculated as this value multiplied by 3 sqrt(pi)/4 .

    Args:
        Te (float): Electron temperature in eV
        ne (float): Electron density in m^{-3}
        Z (float): Ion charge
        removeDiscontinuity (bool, optional): If true will remove the LogLei discontinuity at 10eV by shifting the cut-off to e**2 eV. Defaults to False.

    Returns:
        float: e-i collision time in s
    """

    return (
        4
        * np.pi
        * epsilon0**2
        * np.sqrt(elMass / elCharge)
        * (2 * Te) ** (3 / 2)
        / (Z * logLei(Te, ne, Z, removeDiscontinuity) * ne * elCharge**2)
    )


def lenNorm(Te: float, ne: float, Z: float, removeDiscontinuity=False) -> float:
    """Calculate length norm in meters as t_ei * v_th. Convert to Braginskii by multiplying with 3 sqrt(pi/2)/4

    Args:
        Te (float): Electron temperature in eV
        ne (float): Electron density in m^{-3}
        Z (float): Ion charge
        removeDiscontinuity (bool, optional): If true will remove the LogLei discontinuity at 10eV by shifting the cut-off to e**2 eV. Defaults to False.

    Returns:
        float: Length norm in e-i collision
    """

    return collTimeei(Te, ne, Z, removeDiscontinuity) * velNorm(Te)


def heatFluxNorm(Te: float, ne: float) -> float:
    """Calculate heat flux norm as me * ne * v_th**3 /2. Note: this is 2 x the SOL-KiT normalization.

    Args:
        Te (float): Electron temperature in eV
        ne (float): Electron density in m^{-3}

    Returns:
        float: Heat flux normalization
    """

    return elMass * ne * velNorm(Te) ** 3 / 2


def crossSectionNorm(
    Te: float, ne: float, Z: float, removeDiscontinuity=False
) -> float:
    """Calculate cross section normalization as 1/(ne*lenNorm). Note: this is NOT the SOL-KiT normalization.

    Args:
        Te (float): Electron temperature in eV
        ne (float): Electron density in m^{-3}
        Z (float): Ion charge
        removeDiscontinuity (bool, optional): If true will remove the LogLei discontinuity at 10eV by shifting the cut-off to e**2 eV. Defaults to False.

    Returns:
        float: Cross section normalization in m^2
    """

    return 1 / (ne * lenNorm(Te, ne, Z, removeDiscontinuity))


def eFieldNorm(Te: float, ne: float, Z: float, removeDiscontinuity=False) -> float:
    """Electric field normalization given by me * v_th / (e*t_ei)

    Args:
        Te (float): Electron temperature in eV
        ne (float): Electron density in m^{-3}
        Z (float): Ion charge
        removeDiscontinuity (bool, optional): If true will remove the LogLei discontinuity at 10eV by shifting the cut-off to e**2 eV. Defaults to False.

    Returns:
        float: E-field normalization
    """

    return (
        elMass * velNorm(Te) / (elCharge * collTimeei(Te, ne, Z, removeDiscontinuity))
    )


def calculateNorms(
    Te: float, ne: float, Z: float, removeLogLeiDiscontinuity=False
) -> dict:
    """Calculates all norms in ReMKiT1D-compatible dictionary form

    Args:
        Te (float): Electron temperature in eV
        ne (float): Electron density in m^{-3}
        Z (float): Ion charge
        removeLogLeiDiscontinuity (bool, optional): If true will remove the LogLei discontinuity at 10eV by shifting the cut-off to e**2 eV. Defaults to False.

    Returns:
        dict: Dictionary containing all norms with corresponding ReMKiT1D keys
    """

    return {
        "eVTemperature": Te,
        "density": ne,
        "referenceIonZ": Z,
        "time": collTimeei(Te, ne, Z, removeLogLeiDiscontinuity),
        "velGrid": velNorm(Te),
        "speed": velNorm(Te),
        "EField": eFieldNorm(Te, ne, Z, removeLogLeiDiscontinuity),
        "heatFlux": heatFluxNorm(Te, ne),
        "crossSection": crossSectionNorm(Te, ne, Z, removeLogLeiDiscontinuity),
        "length": lenNorm(Te, ne, Z, removeLogLeiDiscontinuity),
    }
