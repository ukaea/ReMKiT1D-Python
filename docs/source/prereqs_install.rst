==============================
Prerequisites and installation
==============================

For the prerequisites and installation instructions for the Fortran component of ReMKiT1D please see that repository. Here only the Python package will be covered.

-------------
Prerequisites
-------------

The following libraries are required by the package and will be installed automatically if installing using pip

#. numpy
#. xarray
#. holoviews
#. panel 
#. matplotlib
#. h5py
#. scipy
#. pylatex

For version compatibility with the Fortran codebase, check README.

-------------
Installation
-------------

RMK_support is installable through pip trivially by simply calling 

.. code-block:: console
    
    pip install RMK_support

Alternatively, and especially if working with non-release builds, it can be installed directly from the repo root

.. code-block:: console
    
    pip install .