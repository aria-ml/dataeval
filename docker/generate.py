"""Generate Dockerfile.<variant> files from docker/Dockerfile.j2 template."""

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

root = Path(__file__).resolve().parent.parent
config = yaml.safe_load((root / "docker" / "variants.yaml").read_text())

# Fallback for the `DATAEVAL_VERSION` build arg, used only by a local
# `docker build` that omits `--build-arg`. CI always passes an explicit version
# from docker/resolve-version.sh, and that is what a published image carries.
#
# This is a fixed string rather than the last git tag, so that rendering is a
# pure function of Dockerfile.j2 and variants.yaml. Deriving it from
# `git describe` made the output depend on the repository state around it, which
# broke in three separate ways:
#
#   - CI clones shallow and without tags, so `git describe` found nothing and
#     rendered "unknown" -- which is not valid PEP 440, so the Dockerfile it
#     produced could not even build locally (SETUPTOOLS_SCM_PRETEND_VERSION
#     rejects it). `nox -s docker_check` failed on every pipeline.
#   - The value differed per branch (1.1.0 on main, 1.1.2 on release/v1.1), so
#     cherry-picking container changes between branches always needed a
#     regeneration step to avoid a spurious diff.
#   - Every release tag dirtied all three generated files, so the check would
#     fail until somebody regenerated and committed the churn.
#
# A constant has none of those failure modes, and the value it replaces was
# never load-bearing.
PLACEHOLDER_VERSION = "0.0.0.dev0"

# autoescape is off: the output is a Dockerfile, not markup. Leaving Jinja's
# HTML escaping on would silently mangle any label or patch command containing
# an apostrophe or ampersand into a character entity.
env = Environment(
    loader=FileSystemLoader(root / "docker"),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)
template = env.get_template("Dockerfile.j2")

uv_version = config["uv_version"]
python_version = config["python_version"]

# Pre-render the base_image value (it may reference uv_version/python_version)
base_env = Environment(autoescape=False)

for name, variant in config["variants"].items():
    base_image = base_env.from_string(variant["base_image"]).render(
        uv_version=uv_version,
        python_version=python_version,
    )
    extras_flags = " ".join(f"--extra {e}" for e in variant["extras"])

    rendered = template.render(
        variant_name=name,
        base_image=base_image,
        uv_version=uv_version,
        python_version=python_version,
        extras_flags=extras_flags,
        label_title=variant["label_title"],
        label_description=variant["label_description"],
        version=PLACEHOLDER_VERSION,
        security_patches=variant.get("security_patches", []),
    )

    out = root / "docker" / f"Dockerfile.{name}"
    out.write_text(rendered)
    print(f"Generated {out}")
