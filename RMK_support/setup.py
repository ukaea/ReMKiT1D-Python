from setuptools import setup, find_packages  # type: ignore

setup(
    name="RMK_support",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "holoviews",
        "panel",
        "matplotlib",
        "h5py",
        "scipy",
    ],
    author="UK Atomic Energy Authority",
    maintainer="Stefan Mijin",
    maintainer_email="stefan.mijin@ukaea.uk",
)
