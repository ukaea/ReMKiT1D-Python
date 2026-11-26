# Contributing

Thank you for your interest in contributing to `RMK_support`. Please direct questions or suggestions to stefan.mijin@ukaea.uk or open an issue on GitHub.

## Workflow

- Perform feature development on separate branches named `dev-${userID}-${featureName}`, where `${userID}` is the primary developer's GitHub username and `${featureName}` is a short descriptive name.
- Open or reference a relevant GitHub issue before creating a feature branch, and link the issue in the pull request.
- Include unit tests (`pytest`) for all new functionality, especially low-level components. Pull requests that fail CI checks will not be merged.
- Add an entry to `CHANGELOG.md` for every user-facing change.
- All contributions must pass `mypy` type checking and `black` formatting checks.

## Local setup and checks before opening a PR

```bash
# Install in editable mode with dev and docs extras (requires Python >= 3.11)
pip install -e ".[dev,docs]"

# Run type checking and formatting checks
mypy ./RMK_support
black ./RMK_support --check

# Run unit tests (add --exclude_latex_tests if latexmk is unavailable)
pytest

# Verify Sphinx documentation builds cleanly without warnings
sphinx-build -W -b html docs/source docs/build/html
```

## Coding style

- Always add type hints (package ships `py.typed` and targets Python >= 3.11).
- Format code with `black`.
- Variables and functions: `camelCase`, except where this conflicts with standard Python conventions (dunder methods, pytest fixtures, etc.).
- Classes and types: `PascalCase`.
- Module file names: `snake_case`.
- Document all new public classes and functions using Google-style docstrings.
