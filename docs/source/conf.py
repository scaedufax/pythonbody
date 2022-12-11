# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pythonbody'
copyright = '2022, Uli Roth'
author = 'Uli Roth'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for autodoc -----------------------------------------------------
autoclass_content = "class"
autodoc_member_order = "groupwise"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
#html_theme = 'classic'
html_static_path = ['_static']


import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.path.insert(0, pathlib.Path(__file__).parents[3].resolve().as_posix())
