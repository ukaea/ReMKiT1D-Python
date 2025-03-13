# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ReMKiT1D Python Support'
copyright = '2023, Stefan Mijin'
author = 'Stefan Mijin'
release = '2.0.0'

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','sphinx.ext.napoleon','sphinx.ext.mathjax',"nbsphinx","nbsphinx_link"]
napoleon_google_docstring = True
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = []

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_align": "left",
    "primary_sidebar_end": [],
    "navigation_depth": 0,
    "show_nav_level": 3
}

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)