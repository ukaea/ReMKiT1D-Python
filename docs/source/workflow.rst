================================
Workflow and main object classes
================================

A general workflow for the construction of ReMKiT1D run config.json files can be divided into 5 general steps:

#. Setting up global external library options (MPI, PETSc, HDF5)
#. Initializing the basic objects used by the code. These include normalization, the grid, a standard textbook object and optional custom derivations, as well as species data. Most of these have default options that can be used to simplify the initialization procedure.
#. Setting up the variable container by adding variables and optionally initializing them, as well as associating derived variables in the variable container with derivations by supplying a derivation name and a number of required variables used in the calculation. At this stage the interaction between the MPI and HDF5 libraries and the variables can be specified through options to communicate or output specific variables.
#. Defining the models used to evolve/calculate variables by specifying term options and modelbound data. This is the main part of the configuration file and the most complex. See different notebooks for examples of how to initialize models. Optional data manipulator objects can be specified here as well.
#. The last step is specifying time integration options. Here the integrator structure and timestep control can be specified, as well as the structure of the main timeloop (timestep number, output frequency etc.)

The above steps are shown in the figure below, with some of the interdependencies shown in UML style. 

.. image:: ../ReMKiT1D_setup_steps.png
    :width: 600

Short descriptions of main objects in the Python interface used to construct valid ReMKiT1D configuration files are given here. However, the user is directed to the example notebooks for a more complete account of features, together with the individual function documentation. 

-------------
Grid
-------------

Grid options, including both the spatial and velocity grids, are set using a Grid object. 

It is possible to specify the 1D x and v grids both with cell centre coordinates as well as with individual cell widths. The spatial grid also supports varying face Jacobians (representing flux tube width variation).

ReMKiT1D generates both the spatial grid corresponding to the cell centres as well a dual/staggered grid, corresponding to the right cell faces.

See :obj:`RMK_support.grid` for docstrings, as well as the many examples for how grids are set in practice.

-----------------
VariableContainer
-----------------

The VariableContainer contains all globally available variables. ReMKiT1D variables are broadly categorized as either implicit, meaning they can be evolved using implicit methods built with the PETSc library, or derived, meaning they can have derivation rules associated with them. For more details about the implicit solve and derivation rules see the ReMKiT1D code paper. 

Variables can have varying dimensionality:

* Fluid variables live only on the x grid
* Distribution variables live on the x and velocity grids (the velocity grid being composed of the Legendre harmonic index and the velocity magnitude grids)
* Scalar variables 

The dimensionality of variables will impact how they are communicated using MPI as well as which derivation rules can be used with them. 

Finally, variables can be marked as stationary, which forces their time derivative to 0, allowing for implicit solution of equations such as the Poisson equation.

See :obj:`RMK_support.variable_container` for docstrings.

----------
RKWrapper
----------

The user is generally not going to be building individual ReMKiT1D objects by hand, even though this is possible by knowing the corresponding JSON keys. The RKWrapper object provided by RMK_support acts as a centralized configuration object for runs, where all settings can be specified. 

The wrappers allows users to:

#. Add variables directly to the wrapper, without constructing a VariableContainer
#. Add custom derivation objects (see examples for this use case)
#. Add Models and Manipulators 
#. Set external library options (MPI, PETSc, HDF5)
#. Set time integration options 
#. Set species data 
#. Access already added components (e.g. Models and their constituent Terms)

See :obj:`RMK_support.rk_wrapper` for docstrings.

-------------
Other objects
-------------

Other objects, in particular custom Term and Model objects and their components are available in :obj:`RMK_support.simple_containers`. Some more specialized objects are given in :obj:`RMK_support.crm_support`, while :obj:`RMK_support.common_models` contains many prebuilt models. 

For examples of how many of these objects are used see the provided notebooks, in particular ReMKiT1D_advection_test.ipynb and ReMKiT1D_custom_fluid.ipynb.


