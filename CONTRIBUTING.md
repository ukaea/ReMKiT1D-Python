# Contributing

This file is under active development. Please send any suggestions to stefan.mijin@ukaea.uk.

- All feature development should be done in separate branches. Please use branch names of the format dev-\${userID}-\${featureName}, where \${userID} is best set to the main feature developer's GitHub username and \${featureName} is a short name for the feature.
- Raise a relevant issue (if it doesn't exist) before creating your feature branch, and refer to the issue in the PR
- All new features should include reasonable unit test coverage, especially if the feature is low level. Pull requests which fail CI tests will be automatically rejected. 
- Every PR must include corresponding changes in the CHANGELOG.md file
- All contributions must pass mypy linting 

- Coding style points:
    - Use type hints always 
    - Use black formatting 
    - All variables and functions should be camelCase, except where this conflicts with standard Python practice 
    - Names of classes/types should be PascalCase
    - Names of files should be snake_case
    - Document all new features with google format docstrings
