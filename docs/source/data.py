"""Download the datasets the docs notebooks use, as listed in docs/datasets.toml.

conf.py runs `download()` when a notebook is about to execute, so a local `nox -s docs`
fetches what the notebooks read. The docs CI job restores the same datasets from the Harbor
dataset cache first (docs/datasets.toml is the manifest it reads, too), after which the
loaders find the files on disk and download nothing. The notebooks still call the loaders with
`download=True` themselves, for the same reason.

Only raw downloads belong in the manifest. The `as_datamaite=True` exports are derived, and
the notebooks regenerate them.
"""

import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from os import chdir, getcwd, makedirs, path

MANIFEST = path.join(path.dirname(__file__), "..", "datasets.toml")


@contextmanager
def cd(rel_path: str) -> Iterator[None]:
    """Change folder relative to current file."""
    cur_path = getcwd()
    new_path = path.join(path.dirname(__file__), rel_path)
    print(f"push: {cur_path} -> {new_path}")
    makedirs(new_path, exist_ok=True)
    chdir(new_path)
    try:
        yield
    finally:
        print(f"pop: {cur_path} <- {new_path}")
        chdir(cur_path)


def download() -> None:
    with open(MANIFEST, "rb") as f:
        datasets = tomllib.load(f)["dataset"]

    # Notebooks execute with docs/source/notebooks as the working directory and use root="./data".
    with cd("notebooks"):
        for entry in datasets:
            module, name = entry["loader"].rsplit(".", 1)
            kwargs = {k: v for k, v in entry.items() if k != "loader"}
            getattr(import_module(module), name)(root="./data", download=True, **kwargs)


if __name__ == "__main__":
    download()
