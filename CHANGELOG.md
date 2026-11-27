# CHANGELOG

## v2.1.0, xxxx-xx-xx

- Support for ReMKiT1D v1.4.0 solver and performance features, updated PETSc/BDE defaults, and JSON startup reduction
- Python 3.11+ packaging, documentation, and CI modernization, plus bug fixes across I/O, dashboard, models, LaTeX rendering, and examples
- Subsumes skipped v2.0.1 bugfixes

### Breaking Changes

- Raised minimum Python version to `>= 3.11` (from `>= 3.8`) and build requirement to `setuptools >= 77.0`, removed `setup.py`, and set explicit minimum dependency versions (including `param >= 2.0`)
- Updated default `BDEIntegrator` and `RMKContext` PETSc solver options: `reusePreconditioner=True`, `kspSolverType="gmres"`, `maxIters=500`, auto-clamped `relTol` (1% of BDE `nonlinTol` within `[1e-15, 1e-12]`, fallback `1e-14`), and auto `PETScCommandLineOpts` (`bjacobi` + `lu` with RCM ordering and `-ksp_gmres_restart 100` for pure fluid contexts; `bjacobi` + `ilu(0)` when distributions are present)

### Deprecations

- Deprecated `skipPattern` and `MatrixTerm.withSkippingPattern()` (now emitting `DeprecationWarning` and ignored in PETSc COO-based ReMKiT1D v1.4.0 builds); removed internal `skipPattern=True` usage in `flowingIonEIColl` and `eeCollHigherL`

### New Features

- Added `reusePreconditioner` and `andersonDepth` options to `BDEIntegrator`
- Added opt-in `lagIJTerms` mode to `eeCollHigherL` and the kinetic SOL-KiT example (`examples/solkit_mijin_thesis_kin.py`) to lag higher-harmonic $I$/$J$ electron-electron collision integrals as modelbound data
- Exposed and validated `energyInterpolationGrid` in `CRMModelboundData` for ReMKiT1D v1.4.0 variable-energy inelastic transitions (and replaced mutable default `fixedTransitionEnergies=np.array([])` with `None`)
- Added and exported `filterWithin` in `calculation_tree_support` / `RMK_support`
- Omitted all-zero non-scalar derived `initVals` from generated config JSON by default (`VariableContainer` / `RMKContext`) to reduce config size and startup parsing time
- Documented `optionsPETSc` property as read-only and added guidance on opt-in `bcgsl` and `asm` PETSc options for low-core-count runs
- Added `RMK_support.__version__`, explicit `__all__`, PEP 561 `py.typed`, `CITATION.cff`, and `[dev]`/`[docs]` optional extras; modernized Sphinx/ReadTheDocs config, pytest/coverage configuration, and GitHub Actions workflows

### Bug Fixes

- Fixed bug in common variables which lead to rebuilding of density and temperature variables
- Fixed `writeDictToJSON` not truncating trailing bytes when overwriting longer files, and fixed NumPy 2.x scalar HDF5 loading in `loadVariableFromHDF5` and `loadDummyVarFromHDF5`
- Fixed `Variable.dataShape` for time-series scalar variables, `varAndDual` reading `defaultRemap` instead of `defaultLatex` (and missing `\right)`), and `MPIContext.addToComm` type annotation
- Fixed `ScalarMultiVariablePlot` / `ScalarMultiRunPlot` x-axis sliders using spatial `xGridDual` bounds instead of simulation time bounds, and updated `ReMKiT1DDashboard` for `param >= 2` (`param.Selector`) and `xarray` (`.dims`)
- Fixed `eeCollHigherL` passing bound method `elDensity.onDualGrid` instead of `elDensity.isOnDualGrid`, `timeDerivative` `isSingleHarmonics` keyword typo, `NDInterpolationDerivation` missing `bounds_error=False`, `TermGenerator` missing `super().__init__()` call, and `Timestep.check` / `IntegrationScheme.timestep` assertion formatting and setter annotation
- Fixed LaTeX rendering bugs in `UnaryTransform`, `PolynomialDerivation`, `VariableECSTransition`, `CRMElEnergyTermGenerator`, `IntegrationRule`, `LBCModelboundData`, `LBCStencil`, and `flux`/`electricField` SI units
- Fixed Bohm boundary condition ion speed (`u_i` -> `c_s`) in `solkit_mijin_thesis.py` and `solkit_mijin_thesis_kin.py`, fixed `epperlein_short_test.py` (`Nt // 30`), `solkit_mijin_thesis.py` (`loadVars` ordering and `akiPath`), and `solkit_mijin_thesis_kin.py` (`Tu`/`Td`/`akiPath`), removed obsolete manual PETSc overrides from examples/notebooks, and stripped saved outputs from `gaussian_advection_outflow.ipynb`

### New Notebooks

- N/A

## v2.0.0, 2025-08-29

- See beta.1 notes
- Support for v1.3 BDE integrator rewrite 
- Support for v1.3 distribution divergence stencil
- Support for v1.3 diagonal kinetic stencil upgrade

### Breaking Changes

- See beta.1 notes

### Deprecations

- See beta.1 notes

### New Features

- See beta.1 notes
- Updated BDE integrator options 
- New DistDivStencil class that includes the Jacobian in d/dx operator for distributions
- New harmonicOffset argument for diagonal stencil to allow for direct harmonic coupling
- Added option to have the first moment modelbound variables (rate1*) in CRMModelboundData living on the dual grid

### Bug Fixes

- N/A

### New Notebooks 

- See beta.1 notes

## v2.0.0-beta.1, 2025-03-13

- Complete rewrite of the Python interface 

### Breaking Changes

- Due to complete rewrite assume no backwards compatibility unless pointed out

### Deprecations

- The old pre-v2.0.0 dashboard is still included for some of the old examples, but has been deprecated

### New Features

- New OO hierarchy covering the complete workflow, starting from Variables to the IntegrationScheme 
- DSL-like term and model construction through stencils
- LaTeX rendering support for all objects in the new context class 
- Extension of unit support for variables as well as introduction of common variables
- Improved support for dual variables 
- Improved I/O support 
- Non-linear workflow support through lazy evaluation of all component json representations
- New and improved modular and extensible dashboard 
- Improved test coverage and introduction integrated testing for LaTeX summary construction

### Bug Fixes

- N/A

### New Notebooks 

- Old example notebooks have been rewritten for v2.0.0 
- New tutorial notebooks available in separate directory

## v1.2.1, 2025-02-27

- New standard textbook option for logLei discontinuity fix
- Support for changing BDE integrator consolidation interval
- Support for ignoring Jacobians in the diffusion stencil
- Begun refactor of CustomModel (now renamed Model, with CustomModel alias still available)

### Breaking Changes

- N/A

### Deprecations

- N/A

### New Features

- New standard textbook option to remove the discontinuity in the NRL formulary version of e-i Coulomb Log
- The BDE integrator consolidation interval can now be set
- ignoreJacobian option in diffusion stencil

### Bug Fixes

- e-i collision operator ion density dependence fixed

### New Notebooks 

- N/A

## v1.2.0, 2024-09-19

- Added CVODE integrator as an option
- Added support for new manipulator features
- Wrapper and model utilities
- New abstract Term and TermGenerator classes
- Support for DerivationTerms 
- New examples with CVODE and other v1.2.0 features
- Support for new timeloop options
- Support for setting relaxation weight in BDE integrator
- Improvements to Grid class
- Metadata improvements
- Added New Notebooks section to the changelog
- Bug fixes

### Breaking Changes

- ReMKiT1D v1.2.0 will no longer work with old config files, but RMK_Support v1.2.0 should still work with the scripts that generated those files, and will generate v1.2.0-compatible configs.

### Deprecations

- Adding models using just the dictionary form will now raise a deprecation warning.
- Adding term generators using just the dictionary form will now raise a deprecation warning.
- Explicitly setting the global number of implicit and general groups will now raise a deprecation warning.

### New Features

- New CVODE integrator
- Support for new DerivationTerms in ReMKiT1D v1.2.0
- Support for new manipulator features for stationary equations
- New helper functions in wrapper and variable container
- It is now possible to query active term groups in models through the wrapper to avoid mixed term group errors due to empty groups. 
- Terms now inherit from abstract class that enables term group tracking and validity checks
- Term generators are now a class that enables term group tracking
- Setting global integrator data can now automatically detect the correct number of implicit and general groups to request from ReMKiT1D
- The "time" variable is now added automatically by the wrapper unless instructed otherwise
- Can now directly control maximum number of BDE integrator restarts (still hard-capped to 10 in the Fortran code)
- New timeloop option for output-driven timesteps
- New timeloop restart option for setting initial output index
- Over- and under-relaxation now supported in BDE integrator 
- Added functions to the Grid class for calculating the dual cell widths and cell volumes
- Added spatial integral functions to the Grid class and analysis helper routines relating to spatial integrals in analysis_support
- Derivation rules are now stored as xarray attributes, so VariableContainers can be built directly from datasets
- Variables can now have normalisation constants and related unnormalised units associated with them as xarray attributes

### Bug Fixes

- Fixed bug where addTermDiagnosis calls in the wrapper always displayed a warning
- Fixed bug in customFluid1DStencil where the required variables were not correctly set to "none" when not passed
- Fixed bug in VarData validity checking where the wrong warning message was shown

### New Notebooks 

- ReMKiT1D_advection_matrix_free
- ReMKiT1D_limiters_CVODE
- ReMKiT1D_pred_prey_CVODE

## v1.1.0, 2024-02-02

- Support for features in ReMKiT1D v1.1.0
- Calculation tree improvements
- Common model additions
- QoL features
- \_\_rtruediv\_\_ implemented for nodes
- Bug fixes
- Documentation improvements

### Breaking Changes

- N/A

### Deprecations

- N/A

### New Features

- Generalized dvEnergyTerm
- Added automated distribution variable diagnostic manipulator function to wrapper
- Added n-dimensional linear interpolation derivation to simple_containers
- Log scale option for standard dashboard
- Support for non-default electron species ID in ModelboundCRMData
- Added leaf finding feature for calculation trees
- Added base fluid models per species to common_models (these add the standard continuity, momentum, and energy equations)
- Added model allowing for quick construction of calculation tree terms
- Added example notebook for the dvEnergyTerm and the calculation tree model terms
- Models can now be added to wrappers by passing the CustomModel object instead of its dictionary representation. This will trigger checks on the model (currently only checking whether the evolved and implicit variables are registered and whether the required row/column variables live on the same grids as the evolved/implicit variables)
- Calculation trees can now be evaluated by passing a dictionary with keys as the leaf variable names and numpy arrays as entries. User-defined UnaryTransformations need to have a Callable component in order to be used in evaluation of calculation trees
- It is now possible to specify the number of degrees of freedom that an implicit temperature derivation has. This can be useful when a species has temperature anisotropy.
- Can now pass the derivOptions to the variable as it is declared such that it can automatically add the custom derivation used in the variable's derivation rule.
- Added \_\_rtruediv\_\_ operator for node calculations.
- Added option to support resetting the time variable on restart.
- velocityMoment on Grid now handles 1D vectors of the length of the velocity grid and has better assertions
- addTermDiagnosisForVars and addTermDiagnosisForDistVars now raise warnings when no terms are added for any of the passed variables

### Bug Fixes

- Fixed a bug in io when no "time" variable is present in the wrapper
- Fixed communication in amjuel support routines when adding logs of variables
- Fixed bug asserting that len(derivTags) == len(linCoeffs) even if no linCoeffs provided in additiveDerivation
- Fixed a bug where addModel did not show a correct error message when the implicit variable wasn't found for a matrix term

## v1.0.3, 2023-07-27

- Minor bug fixes and code cleanup

### Bug Fixes

- Fixed divide-by-zero issue in common_models.py

## v1.0.2, 2023-06-28

- Minor calculation tree improvements and bug fixes

### New Features

- Added negation operator and right subtraction for calculation trees

### Bug Fixes

- Fixed subtraction bug in calculation tree
- Fixed bug in calculation tree where constants were modified by addition/multiplication even if the node had a unary transform

## v1.0.1, 2023-06-27

- Removed release - see v1.0.2

## v1.0.0, 2023-06-21

- Initial release

### Breaking Changes

- N/A

### Deprecations

- N/A

### New Features

- N/A

### Bug Fixes

- N/A
