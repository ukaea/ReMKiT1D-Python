from typing import Optional
import numpy as np

from .variable_container import Variable, node
from .derivations import Species
from . import derivations
from .remkit_context import RMKContext
from abc import ABC, abstractmethod


def density(name: str, context: RMKContext, **kwargs) -> Variable:
    """Generate a density variable based on normalisation data within a RMKContext. Accepts regular Variable kwargs, except for those having to do with units, which are set automatically.

    Args:
        name (str): Variable name
        context (RMKContext): Context containing the normalisation data and the grid on which the variable is defined

    Returns:
        Variable: Variable with density units
    """
    for key in ["units", "unitSI", "normSI"]:
        assert key not in kwargs, key + " not allowed in density call"

    if "subtype" not in kwargs:
        kwargs["subtype"] = "density"
    var = Variable(
        name,
        context.grid,
        **kwargs,
        units="norm. density",
        unitSI="$m^{-3}$",
        normSI=context.normDensity
    )

    return var


def temperature(name: str, context: RMKContext, **kwargs) -> Variable:
    """Generate a temperature variable based on normalisation data within a RMKContext. Accepts regular Variable kwargs, except for those having to do with units, which are set automatically.

    Args:
        name (str): Variable name
        context (RMKContext): Context containing the normalisation data and the grid on which the variable is defined

    Returns:
        Variable: Variable with temperature units
    """
    for key in ["units", "unitSI", "normSI"]:
        assert key not in kwargs, key + " not allowed in temperature call"

    if "subtype" not in kwargs:
        kwargs["subtype"] = "temperature"

    var = Variable(
        name,
        context.grid,
        **kwargs,
        units="norm. temperature",
        unitSI="eV",
        normSI=context.normTemperature
    )

    return var


def flux(name: str, context: RMKContext, **kwargs) -> Variable:
    """Generate a flux variable based on normalisation data within a RMKContext. Accepts regular Variable kwargs, except for those having to do with units, which are set automatically.

    Args:
        name (str): Variable name
        context (RMKContext): Context containing the normalisation data and the grid on which the variable is defined

    Returns:
        Variable: Variable with flux units
    """
    for key in ["units", "unitSI", "normSI"]:
        assert key not in kwargs, key + " not allowed in flux call"

    if "subtype" not in kwargs:
        kwargs["subtype"] = "flux"

    var = Variable(
        name,
        context.grid,
        **kwargs,
        units="norm. flux",
        unitSI="$m^{-2}s{-1}$",
        normSI=context.normDensity * context.norms["speed"]
    )

    return var


def speed(name: str, context: RMKContext, **kwargs) -> Variable:
    """Generate a speed variable based on normalisation data within a RMKContext. Accepts regular Variable kwargs, except for those having to do with units, which are set automatically.

    Args:
        name (str): Variable name
        context (RMKContext): Context containing the normalisation data and the grid on which the variable is defined

    Returns:
        Variable: Variable with speed units
    """

    for key in ["units", "unitSI", "normSI"]:
        assert key not in kwargs, key + " not allowed in speed call"

    if "subtype" not in kwargs:
        kwargs["subtype"] = "speed"

    var = Variable(
        name,
        context.grid,
        **kwargs,
        units="norm. speed",
        unitSI="$ms^{-1}$",
        normSI=context.norms["speed"]
    )

    return var


def energyDensity(name: str, context: RMKContext, **kwargs) -> Variable:
    """Generate an energy density variable based on normalisation data within a RMKContext. Accepts regular Variable kwargs, except for those having to do with units, which are set automatically.

    Args:
        name (str): Variable name
        context (RMKContext): Context containing the normalisation data and the grid on which the variable is defined

    Returns:
        Variable: Variable with energy density units
    """

    for key in ["units", "unitSI", "normSI"]:
        assert key not in kwargs, key + " not allowed in energy density call"

    if "subtype" not in kwargs:
        kwargs["subtype"] = "energyDensity"

    var = Variable(
        name,
        context.grid,
        **kwargs,
        units="norm. en. density",
        unitSI="$eV m^{-3}$",
        normSI=context.normDensity * context.normTemperature
    )

    return var


def energyFlux(name: str, context: RMKContext, **kwargs) -> Variable:
    """Generate an energy flux variable based on normalisation data within a RMKContext. Accepts regular Variable kwargs, except for those having to do with units, which are set automatically.

    Args:
        name (str): Variable name
        context (RMKContext): Context containing the normalisation data and the grid on which the variable is defined

    Returns:
        Variable: Variable with energy flux units
    """

    for key in ["units", "unitSI", "normSI"]:
        assert key not in kwargs, key + " not allowed in energy flux call"

    if "subtype" not in kwargs:
        kwargs["subtype"] = "energyFlux"

    var = Variable(
        name,
        context.grid,
        **kwargs,
        units="norm. en. flux",
        unitSI="$eV m^{-2} s^{-1}$",
        normSI=context.norms["heatFlux"]
    )

    return var


def electricField(name: str, context: RMKContext, **kwargs) -> Variable:
    """Generate an electric field variable based on normalisation data within a RMKContext. Accepts regular Variable kwargs, except for those having to do with units, which are set automatically.

    Args:
        name (str): Variable name
        context (RMKContext): Context containing the normalisation data and the grid on which the variable is defined

    Returns:
        Variable: Variable with electric field units
    """

    for key in ["units", "unitSI", "normSI"]:
        assert key not in kwargs, key + " not allowed in electric field call"

    if "subtype" not in kwargs:
        kwargs["subtype"] = "eField"

    var = Variable(
        name,
        context.grid,
        **kwargs,
        units="norm. el. field",
        unitSI="$Vm{-1}$",
        normSI=context.norms["EField"]
    )

    return var


class VariableFactory(ABC):
    """Abstract variable factory base class"""

    @abstractmethod
    def density(self, initVals: Optional[np.ndarray] = None) -> Variable:
        pass

    @abstractmethod
    def temperature(self, initVals: Optional[np.ndarray] = None) -> Variable:
        pass

    @abstractmethod
    def flux(self, initVals: Optional[np.ndarray] = None) -> Variable:
        pass

    @abstractmethod
    def flowSpeed(self, initVals: Optional[np.ndarray] = None) -> Variable:
        pass

    @abstractmethod
    def energyDensity(self, initVals: Optional[np.ndarray] = None) -> Variable:
        pass

    @abstractmethod
    def pressure(self, initVals: Optional[np.ndarray] = None) -> Variable:
        pass

    @abstractmethod
    def heatflux(self, initVals: Optional[np.ndarray] = None) -> Variable:
        pass

    @abstractmethod
    def viscosity(self, initVals: Optional[np.ndarray] = None) -> Variable:
        pass


class StandardFluidVariables(VariableFactory):
    """Standard fluid variable factory class, providing fluid variables as expected by the majority of common models"""

    def __init__(
        self,
        context: RMKContext,
        species: Species,
        associateOnCreation=True,
        addOnCreation=True,
    ):
        """Standard fluid variable factory class, providing fluid variables as expected by the majority of common models. The variables are made for a given species, and are by default associated to the species and added to the context. All variables have a dual associated with them.

        Args:
            context (RMKContext): Context containing the grid on which the variables will be defined, as well as normalisation data
            species (Species): Species with which the given variable is associated
            associateOnCreation (bool, optional): If true will associate the variables and their duals to the species object on creation. Defaults to True.
            addOnCreation (bool, optional): If true will add the variables and their duals to the context on variable creation. Defaults to True.
        """
        self.__context__ = context
        self.__species__ = species
        self.__associateOnCreation__ = associateOnCreation
        self.__addOnCreation__ = addOnCreation

    @property
    def species(self):
        """Species for which the variable factory will produce variables"""
        return self.__species__

    @species.setter
    def species(self, species: Species):
        self.__species__ = species

    def density(self, initVals: Optional[np.ndarray] = None) -> Variable:
        """Standard density variable. Implicit and primary lives on the regular grid.

        Args:
            initVals (Optional[np.ndarray], optional): Initial values of the density variable. Defaults to None.

        Returns:
            Variable: Standard density variable
        """
        n = density("n" + self.species.name, self.__context__, data=initVals).withDual()
        if self.__associateOnCreation__:
            if n.name not in self.species.associatedVarNames:
                self.species.associateVar(n, n.dual)
            self.species[n.subtype] = n
        if self.__addOnCreation__:
            if n.name not in self.__context__.variables.varNames:
                self.__context__.variables.add(n)
        return n

    def flux(self, initVals: Optional[np.ndarray] = None) -> Variable:
        """Standard particle flux variable. Implicit and primary lives on the dual grid.

        Args:
            initVals (Optional[np.ndarray], optional): Initial values of the flux variable (on dual grid). Defaults to None.

        Returns:
            Variable: Standard particle flux variable
        """
        G = flux(
            "G" + self.species.name + "_dual",
            self.__context__,
            isOnDualGrid=True,
            data=initVals,
        ).withDual("G" + self.species.name)
        if self.__associateOnCreation__:
            if G.name not in self.species.associatedVarNames:
                self.species.associateVar(G, G.dual)
            self.species[G.subtype] = G.dual
        if self.__addOnCreation__:
            if G.name not in self.__context__.variables.varNames:
                self.__context__.variables.add(G)
        return G

    def energyDensity(self, initVals: Optional[np.ndarray] = None) -> Variable:
        """Standard energy density variable. Implicit and primary lives on the regular grid.

        Args:
            initVals (Optional[np.ndarray], optional): Initial values of the energy density variable. Defaults to None.

        Returns:
            Variable: Standard energy density variable
        """
        W = energyDensity(
            "W" + self.species.name, self.__context__, data=initVals
        ).withDual()
        if self.__associateOnCreation__:
            if W.name not in self.species.associatedVarNames:
                self.species.associateVar(W, W.dual)
            self.species[W.subtype] = W
        if self.__addOnCreation__:
            if W.name not in self.__context__.variables.varNames:
                self.__context__.variables.add(W)
        return W

    def temperature(self, initVals: Optional[np.ndarray] = None) -> Variable:
        """Standard temperature variable. Implicit, stationary, and lives on the regular grid. Expected to be "evolved" using the implicit temperature derivation model from common models.

        Args:
            initVals (Optional[np.ndarray], optional): Initial values of the temperature variable. Defaults to None.

        Returns:
            Variable: Standard temperature variable
        """
        T = temperature(
            "T" + self.species.name, self.__context__, data=initVals, isStationary=True
        ).withDual()
        if self.__associateOnCreation__:
            if T.name not in self.species.associatedVarNames:
                self.species.associateVar(T, T.dual)
            self.species[T.subtype] = T
        if self.__addOnCreation__:
            if T.name not in self.__context__.variables.varNames:
                self.__context__.variables.add(T)
        return T

    def flowSpeed(self, initVals: Optional[np.ndarray] = None) -> Variable:
        """Standard flow speed variable. Derived as flux/density, and primary lives on the dual grid.

        Args:
            initVals (Optional[np.ndarray], optional): Initial value of the flow speed variable. Defaults to None.

        Returns:
            Variable: Standard flow speed variable
        """
        u = speed(
            "u" + self.species.name + "_dual",
            self.__context__,
            data=initVals,
            derivation=self.__context__.textbook["flowSpeedFromFlux"],
            derivationArgs=[
                "G" + self.species.name + "_dual",
                "n" + self.species.name + "_dual",
            ],
            isOnDualGrid=True,
        ).withDual("u" + self.species.name)
        if self.__associateOnCreation__:
            if u.name not in self.species.associatedVarNames:
                self.species.associateVar(u, u.dual)
            self.species[u.subtype] = u.dual
        if self.__addOnCreation__:
            if u.name not in self.__context__.variables.varNames:
                self.__context__.variables.add(u)
        return u

    def heatflux(self, initVals: Optional[np.ndarray] = None) -> Variable:
        """Standard heat flux variable. Implicit, stationary, and primary lives on dual grid.

        Args:
            initVals (Optional[np.ndarray], optional): Initial value of the heat flux variable (on dual grid). Defaults to None.

        Returns:
            Variable: Standard heat flux variable
        """
        q = energyFlux(
            "q" + self.species.name + "_dual",
            self.__context__,
            data=initVals,
            isStationary=True,
            isOnDualGrid=True,
            subtype="heatflux",
        ).withDual("q" + self.species.name)
        if self.__associateOnCreation__:
            if q.name not in self.species.associatedVarNames:
                self.species.associateVar(q, q.dual)
            self.species[q.subtype] = q.dual
        if self.__addOnCreation__:
            if q.name not in self.__context__.variables.varNames:
                self.__context__.variables.add(q)
        return q

    def pressure(self, initVals: Optional[np.ndarray] = None) -> Variable:
        """Standard pressure variable. Derived as density*temperature, and with primary on regular grid.

        Args:
            initVals (Optional[np.ndarray], optional): Initial values of the pressure variable. Defaults to None.

        Returns:
            Variable: Standard pressure variable
        """
        p = energyDensity(
            "p" + self.species.name,
            self.__context__,
            data=initVals,
            derivation=derivations.NodeDerivation(
                "p" + self.species.name,
                node=node(self.density()) * node(self.temperature()),
            ),
            subtype="pressure",
        ).withDual()
        if self.__associateOnCreation__:
            if p.name not in self.species.associatedVarNames:
                self.species.associateVar(p, p.dual)
            self.species[p.subtype] = p
        if self.__addOnCreation__:
            if p.name not in self.__context__.variables.varNames:
                self.__context__.variables.add(p)
        return p

    def viscosity(self, initVals: Optional[np.ndarray] = None) -> Variable:
        """Standard viscosity variable. Implicit, stationary, and with primary on regular grid.

        Args:
            initVals (Optional[np.ndarray], optional): Initial values of the viscosity variable. Defaults to None.

        Returns:
            Variable: Standard viscosity variable
        """
        pi = energyDensity(
            "pi" + self.species.name,
            self.__context__,
            data=initVals,
            isStationary=True,
            subtype="viscosity",
        ).withDual()
        if self.__associateOnCreation__:
            if pi.name not in self.species.associatedVarNames:
                self.species.associateVar(pi, pi.dual)
            self.species[pi.subtype] = pi
        if self.__addOnCreation__:
            if pi.name not in self.__context__.variables.varNames:
                self.__context__.variables.add(pi)
        return pi
