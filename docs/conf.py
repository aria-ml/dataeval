# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

project = "DAML"
copyright = "2023, ARiA"
author = "ARiA"

site_url = "https://jatic.pages.jatic.net/aria/daml/"
repo_url = "https://gitlab.jatic.net/jatic/aria/daml"
repo_name = "DAML"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx_immaterial",
    "sphinx_immaterial.graphviz",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = False

# html_static_path = ['_static']
html_theme = "sphinx_immaterial"
html_show_sourcelink = False
html_theme_options = {
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to light mode",
            },
        },
    ]
}
