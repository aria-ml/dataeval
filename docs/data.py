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
    import keras.datasets as kds
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
