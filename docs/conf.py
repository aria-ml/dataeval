# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from os import getenv

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

project = "DataEval"
copyright = "2024, ARiA"  # noqa: A001
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
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.mathjax",
    # External extensions
    "sphinx_design",
    "myst_nb",
    "enum_tools.autoenum",
]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of items that shouldn't be included in the build.
exclude_patterns = [
    # Patterns
    "Thumbs.db",
    ".DS_Store",
    # Specific directories -- relative to source directories
    ".jupyter_cache",
    "_build",
    # Specific documents
    "how_to/notebooks/ODLearningCurvesTutorial.ipynb",
]

# -----------------------------------------------------------------------------
# Extension configurations
# -----------------------------------------------------------------------------

autodoc_type_aliases = {"ArrayLike": "ArrayLike"}
autosummary_generate = False

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


# because we expose private modules in public namespaces
# and rename some classes, documentation recognizes these
# public classes as aliases, which we don't want
def normalize_module(mod_names):
    import importlib

    for mod_name in mod_names:
        mod = importlib.import_module(mod_name)
        for cls_name in mod.__all__:
            cls = getattr(mod, cls_name)
            cls.__name__ = cls_name
            cls.__module__ = mod_name


def setup(app):
    # pre-download data used in notebooks
    import warnings
    from os import getcwd
    from sys import path

    warnings.filterwarnings("ignore")

    path.append(getcwd())
    import data

    if nb_execution_mode != "off":
        data.download()

    normalize_module(
        [
            "dataeval.detectors.drift",
            "dataeval.detectors.linters",
            "dataeval.detectors.ood",
            "dataeval.metrics.bias",
            "dataeval.metrics.estimators",
            "dataeval.metrics.stats",
            "dataeval.workflows",
            "dataeval.utils.tensorflow",
            "dataeval.utils.torch",
        ]
    )
