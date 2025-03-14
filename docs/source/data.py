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
    from dataeval.utils.data.datasets import CIFAR10, MNIST, VOCDetection

    with cd("how_to"):
        # AETrainerTutorial.ipynb
        # BayesErrorRateEstimationTutorial.ipynb
        # ClassLearningCurvesTutorial.ipynb
        # ClassLabelAnalysisTutorial.ipynb
        # DuplicatesTutorial.ipynb
        # HPDivergenceTutorial.ipynb
        MNIST(root="./data", download=True, image_set="train", corruption="translate")

        # LintingTutorial.ipynb
        CIFAR10(root="./data", download=True, image_set="test")

    with cd("tutorials"):
        # EDA_Part1.ipynb
        # EDA_Part2.ipynb
        # EDA_Part3.ipynb
        # Data_Monitoring.ipynb
        VOCDetection("./data", download=True, image_set="train", year="2011")
