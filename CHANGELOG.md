# CHANGELOG

## v1.2.0, 2024-03-26

- Bug fixes
- Added CVODE integrator as an option

### Breaking Changes

- N/A

### Deprecations

- N/A

### New Features

- New CVODE integrator

### Bug Fixes

- Fixed bug where addTermDiagnosis calls in the wrapper always displayed a warning
- Fixed bug in customFluid1DStencil where the required variables were not correctly set to "none" when not passed

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