import json
from collections import defaultdict
from pathlib import Path

import torch
from test_stage import TestStage
from torch.utils.data import DataLoader
from torchmetrics.utilities.data import dim_zero_cat

from daml.metrics import BER
from daml.metrics.outlier import AEOutlier
from daml.models.tensorflow import AE, create_model

BASE_OPTS = ["Base", "Both"]
TARGET_OPTS = ["Target", "Both"]


class DamlStage(TestStage):
    def __init__(
        self,
        feasibility_opt="Both",
        bias_opt="Both",
        linting_opt="Both",
        sufficiency_opt="Both",
        outlier_detector=AEOutlier(create_model(AE, (100, 10, 10))),
        *args,
        **kwargs,
    ):
        super().__init__()

        # TODO: All of these should be in args/kwargs from Panel
        self.feasibility_opt = feasibility_opt
        self.bias_opt = bias_opt
        self.linting_opt = linting_opt
        self.sufficiency_opt = sufficiency_opt

        # TODO: Instantiated class done in backend script
        self.outlier_detector = outlier_detector

        self.base_str = "dev_train"
        self.target_str = "op_val"

        self.cached_ae = None

    def run(self):
        cache_path = Path(".daml_cache/cache.json")
        self.load_cached_results(cache_path)

        base_dataset = self.dev_dataset if "dev" in self.base_str else self.operational_dataset
        target_dataset = self.dev_dataset if "dev" in self.target_str else self.operational_dataset

        metric_outputs = defaultdict(dict)

        # Feasibility
        if self.feasibility_opt in BASE_OPTS:
            # Check cache for cached split
            f_cache = self.cache.get(self.base_str)
            # Use cache if found, else calculate
            result = f_cache if f_cache is not None else self.feasibility(base_dataset)
            metric_outputs[self.base_str].update(result)
        if self.feasibility_opt in TARGET_OPTS:
            # Check cache for cached split
            f_cache = self.cache.get(self.target_str)
            # Use cache if found, else calculate
            result = f_cache if f_cache is not None else self.feasibility(target_dataset)
            metric_outputs[self.target_str].update(result)

        # Bias
        if self.feasibility_opt in BASE_OPTS:
            # Check cache for cached split
            f_cache = self.cache.get(self.base_str)
            # Use cache if found, else calculate
            result = f_cache if f_cache is not None else self.bias(base_dataset)
            metric_outputs[self.base_str].update(result)
        if self.feasibility_opt in TARGET_OPTS:
            # Check cache for cached split
            f_cache = self.cache.get(self.target_str)
            # Use cache if found, else calculate
            result = f_cache if f_cache is not None else self.bias(base_dataset)
            metric_outputs[self.target_str].update(result)

        # # Linting
        # if self.feasibility_opt in BASE_OPTS:
        #     self.outputs.update(self.linting(base_dataset))
        # if self.feasibility_opt in TARGET_OPTS:
        #     self.outputs.update(self.linting(target_dataset))

        # # Sufficiency
        # if self.feasibility_opt in BASE_OPTS:
        #     self.outputs.update(self.sufficiency(base_dataset))
        # if self.feasibility_opt in TARGET_OPTS:
        #     self.outputs.update(self.sufficiency(target_dataset))

        self.outputs = metric_outputs

    def load_cached_results(self, results: Path) -> None:
        if Path.is_file(results):
            with results.open() as f:
                cache = json.load(f)
                print("Cache hit")
        else:
            self.cache = {}
            print("Cache miss")
            return

        """
        Get all feasibility cached results
        Find split specific cached results for cache ID (if cached)
        """
        # TODO: Handle missing cache/None values -> Potentailly use dict.update(dict.get(str))?
        loaded_cache = {}
        feasibility_cache = cache.get("feasibility_cache")
        if self.feasibility_opt in BASE_OPTS:
            loaded_cache[self.base_str] = feasibility_cache[self.base_str]
        if self.feasibility_opt in TARGET_OPTS:
            loaded_cache[self.target_str] = feasibility_cache[self.target_str]

        self.cache = loaded_cache

    def collect_metrics(self):
        print("Returning metrics")
        return self.outputs

    def collect_report_comsumables(self):
        print("Returning Gradient parameters")
        return self.outputs

    def feasibility(self, dataset) -> dict:
        images = []
        labels = []

        ae = self.cached_ae if self.cached_ae is not None else lambda x: x

        # Using a dataloader transforms CHW to NCHW needed for dim_zero_cat
        dataloader = DataLoader(dataset)
        for i, (image, label, _) in enumerate(dataloader):
            # Currently no preprocessing, so BER takes a long time
            if i == 100:
                break

            images = ae(images)

            images.append(torch.tensor(image))
            labels.append(torch.tensor(label))

        images = dim_zero_cat(images)
        labels = dim_zero_cat(labels)

        ber = BER(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), "MST")
        ber_output = ber.evaluate()

        return ber_output

    def bias(self, dataset) -> dict:
        return {"bias": 0.50}

    def linting(self, dataset) -> dict:
        return {}

    def sufficiency(self, dataset) -> dict:
        return {}

    def outlier_detection(self, train_dataset, val_dataset) -> dict:
        # TODO: Need to swap PyTorch NCHW to TensorFlow NHWC
        # self.outlier_detector.fit(train_dataset)
        # output = self.outlier_detector.predict(val_dataset)
        # return output
        return {}

    def drift_detection(self, dataset) -> dict:
        return {}
