# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from os import getenv

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

project = "DataEval"
copyright = "2025, ARiA"  # noqa: A001
author = "ARiA"

site_url = "https://github.com/aria-ml/dataeval/"
repo_url = "https://github.com/aria-ml/dataeval/"
repo_name = "DataEval"

root_doc = "index"
language = "en"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

extensions = [
    # Internal Sphinx extensions
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.mathjax",
    # External extensions
    "sphinx_design",
    "myst_nb",
    "enum_tools.autoenum",
]

# Coverage show missing items
coverage_show_missing_items = True

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# List of items that shouldn't be included in the build.
exclude_patterns = [
    # Patterns
    "Thumbs.db",
    ".DS_Store",
    # Specific directories -- relative to source directories
    ".jupyter_cache",
    "build",
    # Specific documents
    "how_to/notebooks/ODLearningCurvesTutorial.ipynb",
]

# -----------------------------------------------------------------------------
# Extension configurations
# -----------------------------------------------------------------------------

# autodoc_type_aliases = {"ArrayLike": "ArrayLike"}
# autosummary_generate = False

#---------------------------------------------------------------------------------
# Autoapi settings including templates
#---------------------------------------------------------------------------------

autoapi_dirs = ["../../src/dataeval/"]
autoapi_type = "python"
autoapi_root = "reference"
autoapi_file_pattern = "*.py"
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "show-module-summary",
    "imported-members",
]
autoapi_generate_api_docs = True
autoapi_keep_files = True
autodoc_typehints = "description"
autoapi_own_page_level = "function"
autoapi_member_order = "groupwise"

needs_title_optional = True

# need this for autoapi templates.  Need both of the next two commands
# for different parts of the process. 
autoapi_template_dir = "./_templates/autoapi"
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", "_templates/autoapi"]

napoleon_use_ivar = True  # to correctly handle Attributes header in various classes

# define contains method for auto api template marcros
def contains(seq, item):
    return item in seq

# add the method to the jinja environment
def prepare_jinja_env(jinja_env) -> None:
    jinja_env.tests["contains"] = contains

autoapi_prepare_jinja_env = prepare_jinja_env

# -----------------------------------------------------------------------------
# MyST-NB settings
# -----------------------------------------------------------------------------

EXECUTION_MODE = getenv("NB_EXECUTION_MODE_OVERRIDE")
nb_execution_allow_errors = False
nb_execution_cache_path = ".jupyter_cache"
nb_execution_mode = "cache" if EXECUTION_MODE is None else EXECUTION_MODE
nb_execution_raise_on_error = True
nb_execution_timeout = -1

myst_enable_extensions = ["attrs_inline", "colon_fence", "dollarmath", "html_image"]
myst_heading_anchors = 4

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/DataEval_Logo.png"
html_favicon = "_static/DataEval_Favicon.png"

html_show_sourcelink = True
html_static_path = ["_static"]
html_css_files = ["ARiA.css"]
html_theme_options = {
    "navigation_depth": 2,
    "logo": {"text": "DataEval"},
}


def autoapi_skip_member(app, what, name, obj, skip, options):
    # skip undocumented attributes
    if what == "attribute" and obj.docstring == "":
        skip = True
    # skip modules with undefined or empty __all__
    if (what == "module" or what == "package") and (obj.all is None or len(obj.all) == 0):
        skip = True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", autoapi_skip_member)

    # pre-download data used in notebooks
    import warnings

    warnings.filterwarnings("ignore")

    if nb_execution_mode != "off":
        import os
        import sys

        sys.path.append(os.path.dirname(__file__))
        import data

        data.download()
