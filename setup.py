from setuptools import setup, find_packages  # type: ignore

setup(
    name="RMK_support",
    version="1.2.0",
    packages=["RMK_support"],
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
    description="Python modules and notebooks used to initialize and analyze ReMKiT1D runs ",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="GPLv3",
)
