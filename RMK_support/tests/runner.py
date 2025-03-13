import pathlib, sys
import pytest


def main():
    cwd = pathlib.Path.cwd()

    # Add the project's root directory to the system path
    sys.path.append(str(cwd.parent))

    # This is optional, but you can add a lib directory
    # To the system path for tests to be able to use
    sys.path.append(str(cwd / "examples"))

    return pytest.main()


if __name__ == "__main__":
    raise SystemExit(main())
