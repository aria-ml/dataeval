# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from contextlib import contextmanager
from os import chdir, getcwd, getenv

# -- Project information ---------------------------------------------------
project = "DAML"
copyright = "2024, ARiA"  # noqa: A001
author = "ARiA"

site_url = "https://github.com/aria-ml/daml/"
repo_url = "https://github.com/aria-ml/daml/"
repo_name = "DAML"

root_doc = "index"
language = "en"

# -- General configuration -------------------------------------------------

extensions = [
    # Internal Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    # External extensions
    "sphinx_rtd_size",
    "sphinx_design",
    # "sphinx_tabs.tabs",
    # "sphinx_inline_tabs",
    "myst_nb",
    "enum_tools.autoenum",
]

source_suffix = [".rst", ".md"]

templates_path = ["_templates"]
exclude_patterns = [
    ".jupyter_cache",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "tutorials/notebooks/ODLearningCurvesTutorial.ipynb",
]

# -- Extension configurations ----------------------------------------------

autosummary_generate = False
sphinx_rtd_size_width = "80%"

# -- MyST-NB settings ------------------------------------------------------

EXECUTION_MODE = getenv("NB_EXECUTION_MODE_OVERRIDE")
nb_execution_allow_errors = False
nb_execution_cache_path = ".jupyter_cache"
nb_execution_mode = "cache" if EXECUTION_MODE is None else EXECUTION_MODE
nb_execution_raise_on_error = True
nb_execution_timeout = -1

myst_enable_extensions = ["colon_fence", "html_image"]


# -- HTML output ----------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_logo = "_static/DAML_Logo_Name.png"
# html_logo = "_static/DAML_LogoOnly.png"
html_favicon = "_static/DAML_Favicon.png"


html_show_sourcelink = False
html_theme_options = {
    "navigation_depth": 3,
    "logo_only": True,
}
html_static_path = ["_static"]

doctest_test_doctest_blocks = ""


@contextmanager
def cwd(path):
    old_path = getcwd()
    chdir(path)
    try:
        yield
    finally:
        chdir(old_path)


def predownload_data():
    import keras.api._v2.keras.datasets as kds
    import tensorflow_datasets as tfds
    from torchvision.datasets import MNIST

    # Assume we are running in the docs directory with notebooks in tutorials/notebooks
    with cwd("./tutorials/notebooks"):
        # AETrainerTutorial.ipynb
        # ClassLearningCurvesTutorial.ipynb
        # ClassLabelAnalysisTutorial.ipynb
        MNIST(root="./data/", train=True, download=True)
        MNIST(root="./data/", train=False, download=True)

        # BayesErrorRateEstimationTutorial.ipynb
        kds.mnist.load_data()

        # DriftDetectionTutorial.ipynb
        # HPDivergenceTutorial.ipynb
        # OODDetectionTutorial.ipynb
        tfds.load("mnist", split="train")
        tfds.load("mnist_corrupted/translate", split="train")


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
    predownload_data()
    normalize_module(
        [
            "daml.detectors",
            "daml.metrics",
            "daml.workflows",
            "daml.flags",
            "daml.models.tensorflow",
            "daml.models.torch",
        ]
    )
