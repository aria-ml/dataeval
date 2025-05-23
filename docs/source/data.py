from contextlib import contextmanager
from os import chdir, getcwd, makedirs, path


@contextmanager
def cd(rel_path: str):
    """Change folder relative to current file"""
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
    from dataeval.utils.datasets import CIFAR10, MNIST, VOCDetection

    with cd("notebooks"):
        MNIST(root="./data", download=True, image_set="train", corruption="translate")
        CIFAR10(root="./data", download=True, image_set="test")
        VOCDetection("./data", download=True, image_set="train", year="2011")
