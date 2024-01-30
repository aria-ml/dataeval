from typing import Tuple

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runfunctional",
        action="store_true",
        default=False,
        help="run functional tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runfunctional"):
        # --runfunctional given in cli: do not skip slow tests
        return
    skip_func = pytest.mark.skip(reason="need --runfunctional option to run")
    for item in items:
        if "functional" in item.keywords:
            item.add_marker(skip_func)


@pytest.fixture
def mnist() -> Tuple[np.ndarray, np.ndarray]:
    path = "tests/datasets/mnist.npz"
    with np.load(path, allow_pickle=True) as fp:
        covariates, labels = fp["x_train"][:1000], fp["y_train"][:1000]

    return covariates, labels
