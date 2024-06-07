from pathlib import Path

# import maite.protocols.object_detection as od
# import maite.protocols.image_classification as ic
from typing import Any, Dict

import torch.nn as nn
from torch.utils.data import Dataset


class TestStage:
    """Base class with common methods and structure"""

    # list of attributes which are universal across all tools
    model = None
    dev_dataset = None
    operational_dataset = None
    comparison_model = None
    target_performance = None

    def __init__(self, *args, **kwargs):
        self.outputs = {}

    def load_model(self, model: nn.Module) -> None:
        """Provide the model under test to the test stage"""
        self.model = model

    def load_comparison_model(self, model: nn.Module) -> None:
        """Provide the comparison model to the test stage"""
        self.comparison_model = model

    def load_development_dataset(self, dataset: Dataset) -> None:
        """Provide the development dataset to the test stage"""
        self.dev_dataset = dataset

    def load_operational_dataset(self, dataset: Dataset) -> None:
        """Provide the operational dataset to the test stage"""
        self.operational_dataset = dataset

    def load_target_performance(self, target: float) -> None:
        """Provide a target performance that the test stage may need to evaluate against"""
        self.target_performance = target

    def load_cached_results(self, results: Path) -> None:
        """Load cached results from a previous run so that they may be accessed with the collect_metrics and the
        collect_report_consumables methods"""
        pass

    def run(self) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""
        pass

    def collect_metrics(self) -> Dict[str, float]:
        """Access any top-level metrics that the test stage computes in the run function or
        loads in the load_cached_results method"""
        return {}

    def collect_report_comsumables(self) -> Dict[str, Any]:
        """Access the in-depth data needed by Gradient to produce a report generated in the run method or in the
        load_cached_results method"""
        return {}

    @property
    def name(self):
        return self.__class__.__name__
