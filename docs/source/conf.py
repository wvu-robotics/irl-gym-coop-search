# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'irl_gym'
copyright = '2022, Jared J. Beard, R. Michael Butts, Yu Gu'
author = 'Jared J. Beard, R. Michael Butts, Yu Gu'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx_math_dollar', 'sphinx.ext.mathjax']

mathjax2_config = {
    'extensions': ['tex2jax.js'],
    'jax': ['input/TeX', 'output/HTML-CSS'],
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
        'packages': ['base', 'require']
    },
}

mathjax3_config = {
  'extensions': ['tex2jax.js'],
  'jax': ['input/TeX', 'output/HTML-CSS'],
  "tex2jax": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
    'packages': ['base', 'require']
  }
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
