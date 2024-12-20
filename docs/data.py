from contextlib import contextmanager
from os import chdir, getcwd, makedirs, path


@contextmanager
def cwd(rel_path):
    old_path = getcwd()
    new_path = path.abspath(rel_path)
    print(f"push: {old_path} -> {new_path}")
    makedirs(new_path, exist_ok=True)
    chdir(new_path)
    try:
        yield
    finally:
        print(f"pop: {old_path} <- {new_path}")
        chdir(old_path)


def download():
    from dataeval.utils.torch.datasets import CIFAR10, MNIST, VOCDetection

    # Assume we are running in the docs directory with notebooks in tutorials/notebooks
    with cwd("how_to/notebooks"):
        # AETrainerTutorial.ipynb
        # BayesErrorRateEstimationTutorial.ipynb
        # ClassLearningCurvesTutorial.ipynb
        # ClassLabelAnalysisTutorial.ipynb
        # DriftDetectionTutorial.ipynb
        # DuplicatesTutorial.ipynb
        # HPDivergenceTutorial.ipynb
        # OODDetectionTutorial.ipynb
        MNIST(root="./data", train=True, download=True)
        MNIST(root="./data", train=True, download=True, corruption="translate")

        # LintingTutorial.ipynb
        CIFAR10(root="./data", train=False, download=True)

    with cwd("tutorials"):
        # EDA_Part1.ipynb
        # EDA_Part2.ipynb
        # EDA_Part3.ipynb
        # Data_Monitoring.ipynb
        VOCDetection("./data", year="2011", image_set="train", download=True)
