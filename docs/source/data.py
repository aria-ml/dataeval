from contextlib import contextmanager
from os import chdir, getcwd, makedirs, path


@contextmanager
def cd(rel_path: str):
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


def download():
    from maite_datasets.image_classification import CIFAR10, MNIST
    from maite_datasets.object_detection import SeaDrone, VOCDetection

    with cd("notebooks"):
        CIFAR10(root="./data", download=True, image_set="test")
        MNIST(root="./data", download=True, image_set="train", corruption="translate")
        SeaDrone(root="./data", download=True, image_set="val")
        VOCDetection("./data", download=True, image_set="train", year="2012")
