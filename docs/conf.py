# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from os import getenv

import numpy as np

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

# bittex config
bibtex_bibfiles = ["refs.bib"]
# Coverage show missing items
coverage_show_missing_items = True

# autoapi directories (where to look for files)
autoapi_dirs = ["../src/dataeval"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    #'private-members',
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

autoapi_ignore = [
    "*/utils/tensorflow/internal/utils.py",
    "*/utils/tensorflow/internal/autoencoder.py",
    "*/utils/tensorflow/internal/loss.py",
    "*/utils/tensorflow/internal/trainer.py",
    "*/utils/tensorflow/internal/pixelcnn.py",
]

autodoc_mock_imports = [
    "gmm",
    "PixelCNN",
    "trainer",
    "predict_batch",
    "tensorflow",
    "tf-keras",
    # "./src/dataeval/utils/tensorflow/_internal/gmm",
    # "./src/dataeval/utils/tensorflow/internal/trainer",
    # "./src/dataeval/utils/tensorflow/internal/utils",
    # "./src/dataeval/utils/tensorflow/internal/autoencoder",
    # "./src/dataeval/utils/tensorflow/internal/loss",
    # "./src/dataeval/utils/tensorflow/internal/pixelcnn",
    "./src/dataeval/utils/tensorflow/internal/trainer",
]

autoapi_keep_files = True
napoleon_use_ivar = True  # to correctly handle Attributes header in various classes
# Fixes duplicate documentation warning

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

# Set numpy print option to legacy 1.25 so native numpy types
# are not printed with dtype information.

# WITHOUT LEGACY=1.25
# >>> np.int32(16)
# np.int32(16)

# WITH LEGACY=1.25
# >>> np.int32(16)
# 16

if np.__version__[0] == "2":
    np.set_printoptions(legacy="1.25")


# because we expose private modules in public namespaces
# and rename some classes, documentation recognizes these
# public classes as aliases, which we don't want
# def normalize_module(mod_names):
#    import importlib
#
#    for mod_name in mod_names:
#        mod = importlib.import_module(mod_name)
#        for cls_name in mod.__all__:
#            cls = getattr(mod, cls_name)
#            cls.__name__ = cls_name
#            cls.__module__ = mod_name


# did not work. Warning was not fixed
# def maybe_skip_member(app, what, name, obj, skip, options):
#    # print app, what, name, obj, skip, options
#    skip = None
#    if what == "function" and "gmm" in name or what == "class" and "GaussianMixtureModelParams" in name:
#        skip = True
#    return skip


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


# did not work. Warning still issued.
#   app.connect("autoapi-skip-member", maybe_skip_member)


# did not work to skip documentation
# def setup(sphinx):
# didn't work. Member was not skipped
#        sphinx.connect('autoapi-skip-member', maybe_skip_member)

# ----------------------------------------------------------------------
# Mock Import from StackOverflow
# ---------------------------------------------------------------------
