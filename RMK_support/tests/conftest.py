import pytest


def pytest_addoption(parser):
    parser.addoption("--exclude_latex_tests", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "latex: mark test as containing calls to latex features"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--exclude_latex_tests"):

        skip_latex = pytest.mark.skip(reason="--exclude_latex_tests set")
        for item in items:
            if "latex" in item.keywords:
                item.add_marker(skip_latex)
        return

    return
