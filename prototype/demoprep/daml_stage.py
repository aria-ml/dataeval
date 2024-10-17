import json
from collections import defaultdict
from pathlib import Path

import torch
from test_stage import TestStage
from torch.utils.data import DataLoader
from torchmetrics.utilities.data import dim_zero_cat

from dataeval.detectors.ood import OOD_AE
from dataeval.metrics.estimators import ber
from dataeval.utils.tensorflow.models import AE, create_model

BASE_OPTS = ["Base", "Both"]
TARGET_OPTS = ["Target", "Both"]


class DataEvalStage(TestStage):
    def __init__(
        self,
        feasibility_dataset="Both",
        bias_dataset="Both",
        linting_dataset="Both",
        sufficiency_dataset="Both",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.feasibility_dataset = feasibility_dataset
        self.bias_dataset = bias_dataset
        self.linting_dataset = linting_dataset
        self.sufficiency_dataset = sufficiency_dataset

        self.ood_detector = OOD_AE(create_model(AE, (28, 28, 1)))

        self.base_str = "dev_train"
        self.target_str = "op_val"

        # Runtime caching < Multiple metrics need preprocessing by an AE >
        self.cached_ae_path = None
        self.cached_images = {}  # {dataset: embeddings}
        self.cached_labels = {}  # {dataset: embeddings}

        self.load_cached_results(Path(".dataeval_cache/cache.json"))

    def run(self) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""
        self.base_dataset = self.dev_dataset if "dev" in self.base_str else self.operational_dataset
        self.target_dataset = self.dev_dataset if "dev" in self.target_str else self.operational_dataset

        metric_outputs = defaultdict(dict)

        for k, v in self.feasibility().items():
            metric_outputs[k].update(v)
        for k, v in self.bias().items():
            metric_outputs[k].update(v)
        for k, v in self.linting().items():
            metric_outputs[k].update(v)
        for k, v in self.sufficiency().items():
            metric_outputs[k].update(v)

        self.outputs = metric_outputs

        # TODO: Update and save outputs to cache file (Could split into multiple cache files?)

    def load_cached_results(self, results: Path) -> None:
        """Load cached results from a previous run so that they may be accessed with the collect_metrics and the
        collect_report_consumables methods"""
        if Path.is_file(results):
            print("Cache hit")
            with results.open() as f:
                self.cache = json.load(f)
        else:
            print("Cache miss")
            self.cache = {}

    def collect_metrics(self):
        print("Returning metrics")
        return self.outputs

    def collect_report_comsumables(self):
        print("Returning Gradient parameters")
        return self.outputs

    def _run_base(self, func):
        return func(self.base_dataset)

    def _run_target(self, func):
        return func(self.target_dataset)

    def feasibility(self):
        feasibility_output = defaultdict(dict)
        feasibility_cache = self.cache.get("feasibility", {})
        METRIC = self._ber

        # First run on base, then target. Only run if no cache found
        if self.feasibility_dataset in BASE_OPTS:
            cache = feasibility_cache.get(self.base_str)
            result = cache if cache else self._run_base(METRIC)
            feasibility_output[self.base_str].update(result)
        if self.feasibility_dataset in TARGET_OPTS:
            cache = feasibility_cache.get(self.base_str)
            result = cache if cache else self._run_target(METRIC)
            feasibility_output[self.target_str].update(result)

        return feasibility_output

    def _ber(self, dataset) -> dict:
        images, labels = [], []

        # Using a dataloader transforms CHW to NCHW needed for dim_zero_cat
        for i, (image, label, _) in enumerate(DataLoader(dataset)):
            if i == 100:  # Only need a subset for testing
                break
            images.append(image if isinstance(image, torch.Tensor) else torch.tensor(image))
            labels.append(label if isinstance(label, torch.Tensor) else torch.tensor(label))

        images = dim_zero_cat(images).detach().cpu().numpy()
        labels = dim_zero_cat(labels).detach().cpu().numpy()

        return ber(images, labels).dict()

    def bias(self) -> dict:
        bias_output = defaultdict(dict)
        bias_cache = self.cache.get("bias", {})
        METRICS = [self._balance, self._coverage, self._parity]

        if self.bias_dataset in BASE_OPTS:
            cache = bias_cache.get(self.base_str)
            if cache is not None:
                result = cache
            else:
                result = {}
                for metric in METRICS:
                    result.update(self._run_base(metric))
            bias_output[self.base_str].update(result)
        if self.bias_dataset in TARGET_OPTS:
            cache = bias_cache.get(self.target_str)
            if cache is not None:
                result = cache
            else:
                result = {}
                for metric in METRICS:
                    result.update(self._run_target(metric))
            bias_output[self.target_str].update(result)

        return bias_output

    def _coverage(self, dataset) -> dict:
        return {"coverage": 0.90}

    def _parity(self, dataset) -> dict:
        return {"parity": 0.25}

    def _balance(self, dataset) -> dict:
        return {"balance": 0.5}

    def linting(self) -> dict:
        return {}

    def sufficiency(self) -> dict:
        return {}

    def ood_detection(self) -> dict:
        # TODO: Need to swap PyTorch NCHW to TensorFlow NHWC
        # self.outlier_detector.fit(train_dataset)
        # output = self.outlier_detector.predict(val_dataset)
        # return output
        return {}

    def drift_detection(self) -> dict:
        return {}
