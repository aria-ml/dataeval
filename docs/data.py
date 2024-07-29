from contextlib import contextmanager
from os import chdir, getcwd


@contextmanager
def cwd(path):
    old_path = getcwd()
    chdir(path)
    try:
        yield
    finally:
        chdir(old_path)


def download():
    import tensorflow_datasets as tfds
    from torchvision.datasets import CIFAR10, MNIST, VOCDetection

    # Assume we are running in the docs directory with notebooks in tutorials/notebooks
    with cwd("./tutorials/notebooks"):
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

    with cwd("./how_to"):
        # EDA.ipynb
        VOCDetection("./data", year="2011", image_set="train", download=True)
