"""Generate Dockerfile.<variant> files from docker/Dockerfile.j2 template."""

import re
import subprocess
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

root = Path(__file__).resolve().parent.parent
config = yaml.safe_load((root / "docker" / "variants.yaml").read_text())


def _default_version() -> str:
    """Resolve the last published tag for the `DATAEVAL_VERSION` build-arg default.

    `--abbrev=0` returns the most recent tag *without* the `-N-g<sha>[-dirty]`
    suffix that `git describe` normally appends. Using the bare tag keeps the
    committed Dockerfile.<variant> defaults stable across regenerations — the
    rendered ARG only churns when an actual release tag lands, not when a
    contributor regenerates from a dirty working tree.

    This default is only consumed by local `docker build` invocations that omit
    `--build-arg`; CI builds always pass an explicit version resolved by
    docker/resolve-version.sh, which becomes the source of truth in the
    published image.
    """
    try:
        tag = (
            subprocess
            .check_output(
                ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return re.sub(r"^v", "", tag)


version = _default_version()

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
        version=version,
        security_patches=variant.get("security_patches", []),
    )

    out = root / "docker" / f"Dockerfile.{name}"
    out.write_text(rendered)
    print(f"Generated {out}")
