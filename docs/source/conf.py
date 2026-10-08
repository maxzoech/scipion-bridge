# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Scipion Bridge"
copyright = "2025-2026, Maximilian Zoech"
author = "Maximilian Zoech"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "**.ipynb_checkpoints"]

# The library uses Google style docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

autoapi_dirs = ["../../src/scipion_bridge"]
autoapi_type = "python"
autoapi_root = "autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

# Do not execute notebooks while building the documentation; they download
# data and call external programs.
nbsphinx_execute = "never"

# Render ``code`` as inline Python literals.
default_role = "code"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = "Scipion Bridge"
html_static_path = []
html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
