==============================
Prerequisites and installation
==============================

For the prerequisites and installation instructions for the Fortran component of ReMKiT1D please see the `ReMKiT1D repository <https://github.com/ukaea/ReMKiT1D>`_. Here only the Python package is covered.

-------------
Prerequisites
-------------

``RMK_support`` requires **Python >= 3.11**. The following runtime libraries are required and installed automatically by ``pip``:

#. ``numpy``
#. ``xarray``
#. ``holoviews``
#. ``panel``
#. ``param``
#. ``matplotlib``
#. ``h5py``
#. ``scipy``
#. ``pylatex``

To use the PyLaTeX PDF generation utilities, a local LaTeX distribution with ``latexmk`` is also required (see the `PyLaTeX documentation <https://jeltef.github.io/PyLaTeX/current>`_).

For version compatibility with the Fortran codebase (``RMK_support v2.1.0`` ↔ ``ReMKiT1D v1.4.0``), see the repository ``README.md``.

------------
Installation
------------

``RMK_support`` is installable from PyPI via ``pip``:

.. code-block:: console

    pip install RMK_support

Alternatively, especially when working with development builds or contributing, install directly from the repository root:

.. code-block:: console

    pip install .
    # Editable install with development and documentation extras:
    pip install -e ".[dev,docs]"