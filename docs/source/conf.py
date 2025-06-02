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
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.mathjax",
    "sphinx_tabs.tabs",
    # External extensions
    "autoapi.extension",
    "myst_nb",
    "sphinx_design",
    "sphinx_immaterial",
    "sphinx_immaterial.graphviz",
    "sphinx_new_tab_link",
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
]

# Add any paths that contain templates here, relative to this directory.
# Default autoapi templates are at {pyenv}/lib/python{ver}/site-packages/autoapi/templates
templates_path = ["_templates", "_templates/autoapi"]

suppress_warnings = ["ref.python"]

# ---------------------------------------------------------------------------------
# Autoapi settings including templates
# ---------------------------------------------------------------------------------

autoapi_dirs = ["../../src/dataeval/"]
autoapi_type = "python"
autoapi_root = "reference/autoapi"
autoapi_file_pattern = "*.py"
autoapi_python_class_content = "class"
autoapi_options = [
    "members",
    "show-module-summary",
    "imported-members",
    "inherited-members",
]
autoapi_generate_api_docs = True
# uncomment to review or debug generated content
# autoapi_keep_files = True
autodoc_typehints = "description"
autoapi_own_page_level = "function"
autoapi_member_order = "groupwise"
autoapi_add_toctree_entry = False

# need this for autoapi templates
autoapi_template_dir = "./_templates/autoapi"

# ---------------------------------------------------------------------------------
# Autoapi jinja environment prep
# ---------------------------------------------------------------------------------


# define contains method for auto api template macros
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
myst_footnote_transition = False

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = "sphinx_immaterial"
html_logo = "_static/images/DataEval_Logo.png"
html_favicon = "_static/images/DataEval_Favicon.png"

html_show_sourcelink = True
html_static_path = ["_static"]
html_theme_options = {
    "repo_url": "https://github.com/aria-ml/dataeval/",
    "icon": {"repo": "fontawesome/brands/github"},
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "white",
            "accent": "indigo",
            "toggle": {
                "icon": "material/toggle-switch-off-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "black",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/toggle-switch",
                "name": "Switch to light mode",
            },
        },
    ],
    "features": [
        "navigation.expand",
        "navigation.instant",
        "navigation.sections",
        "navigation.tabs",
        "navigation.tabs.sticky",
    ],
}


def inherits_from(obj, full_name: str) -> bool:
    parent = obj.obj.get("inherited_from")
    return bool(parent and parent.get("full_name") == full_name)


def autoapi_skip_member(app, what, name, obj, skip, options):
    # skip undocumented attributes
    if what == "attribute" and obj.docstring == "":
        skip = True
    # skip modules with undefined or empty __all__
    if (what == "module" or what == "package") and (obj.all is None or len(obj.all) == 0):
        skip = True
    # selectively skip inherited members
    if inherits_from(obj, "torch.nn.modules.module.Module"):
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

        data.download()  # type: ignore
