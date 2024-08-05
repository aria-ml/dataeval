from contextlib import contextmanager
from os import chdir, getcwd, path, walk


@contextmanager
def cwd(rel_path):
    old_path = getcwd()
    new_path = path.abspath(rel_path)
    print(f"{old_path} -> {new_path}")
    chdir(new_path)
    try:
        yield
    finally:
        print(f"{old_path} <- {new_path}")
        chdir(old_path)


def download():
    import tensorflow_datasets as tfds
    from torchvision.datasets import CIFAR10, MNIST, VOCDetection

    for root, dirs, _ in walk(getcwd()):
        for d in dirs:
            print(path.join(root, d))

    # Assume we are running in the docs directory with notebooks in tutorials/notebooks
    with cwd("tutorials/notebooks"):
        # AETrainerTutorial.ipynb
        # BayesErrorRateEstimationTutorial.ipynb
        # ClassLearningCurvesTutorial.ipynb
        # ClassLabelAnalysisTutorial.ipynb
        MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]
        MNIST(root="./data", train=True, download=True)
        MNIST(root="./data", train=False, download=True)

        # LintingTutorial.ipynb
        CIFAR10(root="./data", train=False, download=True)

        # DriftDetectionTutorial.ipynb
        # HPDivergenceTutorial.ipynb
        # OODDetectionTutorial.ipynb
        tfds.load("mnist", split="train")
        tfds.load("mnist_corrupted/translate", split="train")

    with cwd("how_to"):
        # EDA.ipynb
        VOCDetection("./data", year="2011", image_set="train", download=True)
