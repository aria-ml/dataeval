import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, TypeAlias, Union

from test_stage import TestStage
from torch.utils.data import DataLoader
from torchmetrics.utilities.data import dim_zero_cat
from tqdm import tqdm

import daml.metrics as dm

Dataset_T: TypeAlias = List[Literal["development", "operational"]]
Checkbox: TypeAlias = Optional[List[str]]


class DamlTestStage(TestStage):
    _cache_dir = Path(".daml_cache")
    _cache_file = Path("cache.json")

    def __init__(
        self,
        linting_options: Checkbox = None,
        linting_dataset: Dataset_T = ["development", "operational"],
        bias_options: Checkbox = ["balance", "coverage", "parity"],
        bias_dataset: Dataset_T = ["development", "operational"],
        drift: bool = False,
        ood_detection: bool = False,
        feasibility_dataset: Dataset_T = ["development", "operational"],
        sufficiency_dataset: Dataset_T = ["development", "operational"],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.linting_opts = linting_options if linting_options else []
        self.linting_ds = linting_dataset

        self.bias_opts = bias_options if bias_options else []
        self.bias_dataset = bias_dataset

        self.do_drift = drift
        self.do_ood_detection = ood_detection

        self.feasibility_dataset = feasibility_dataset
        self.sufficiency_dataset = sufficiency_dataset

        self.dataset_dict = {}

        # Runtime caching <Multiple metrics need preprocessing by an AE>
        self.cached_ae_path = None
        self.cached_images = {}  # {dataset: embeddings}
        self.cached_labels = {}  # {dataset: embeddings}

    def _split_images_labels(self, name, dataset):
        images, labels = [], []
        dev_loader = DataLoader(dataset, batch_size=8)
        limit = 100
        for i, batch in tqdm(enumerate(dev_loader), total=limit):
            if i == 100:
                break
            images.append(batch[0])
            labels.append(batch[1])
        images = dim_zero_cat(images).detach().cpu().numpy()
        labels = dim_zero_cat(labels).detach().cpu().numpy()
        self.cached_images[name] = images
        self.cached_labels[name] = labels
        print(f"Saved {name} data")

    def run(self) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""

        # Immediately use cache if exists
        self.load_cached_results(self._cache_dir / self._cache_file)
        if self.cache:
            self.outputs = self.cache
            return

        # Split images and labels and save results for metrics that require them separately
        if self.dev_dataset is not None:
            self._split_images_labels("development", self.dev_dataset)
        if self.operational_dataset is not None:
            self._split_images_labels("operational", self.operational_dataset)

        self.dataset_dict = {"development": self.dev_dataset, "operational": self.operational_dataset}

        def update_dict(d1: dict, d2: dict):
            """Updates d1 with d2s content"""
            for k, v in d2.items():
                d1[k].update(v)
            return d1

        metric_outputs = defaultdict(dict)

        metric_outputs = update_dict(metric_outputs, self.feasibility())
        metric_outputs = update_dict(metric_outputs, self.bias())
        metric_outputs = update_dict(metric_outputs, self.sufficiency())
        metric_outputs = update_dict(metric_outputs, self.drift_detection())
        metric_outputs = update_dict(metric_outputs, self.outlier_detection())

        self.outputs = metric_outputs

        print(self.outputs)
        self._save_cache()

    def load_cached_results(self, results: Path) -> None:
        """Load cached results from a previous run so that they may be accessed with the collect_metrics and the
        collect_report_consumables methods"""
        if Path.is_file(results):
            print("Cache hit")
            with results.open() as f:
                self.cache = json.load(f)
            print(self.cache)
        else:
            print("Cache miss")
            self.cache = {}

    def _save_cache(self) -> None:
        # Check path
        folder = self._cache_dir
        folder.mkdir(exist_ok=True)
        cache_file_path = folder / self._cache_file

        # Save results
        with cache_file_path.open("w+") as f:
            json.dump(self.outputs, f, indent=4, sort_keys=True)

    def run_metric(self, select: Dataset_T, metric_dict: Dict[str, Callable], opts: List[str]):
        """
        Sets the dataset name as outer key, metric as inner key, dict output as value

        ex. {"development": "ber": {"ber": 0.18, "ber_lower": 0.095, "max_accuracy": .9105}}
        """
        output = defaultdict(dict)

        for name in select:
            output[name].update({metric: metric_dict[metric](name) for metric in opts})

        return output

    def run_comparison_metric(self, metric_dict):
        """Run metrics that take in two datasets"""
        output = {name: metric(self.dev_dataset, self.operational_dataset) for name, metric in metric_dict.items()}
        return output

    def feasibility(self):
        METRICS_DICT = {"ber": self._ber}
        opts: Checkbox = ["ber"]

        return self.run_metric(self.feasibility_dataset, METRICS_DICT, opts)

    def bias(self) -> dict:
        METRICS_DICT = {"balance": self._balance, "coverage": self._coverage, "parity": self._parity}
        opts: Checkbox = self.bias_opts

        return self.run_metric(self.bias_dataset, METRICS_DICT, opts)

    def sufficiency(self) -> dict:
        METRICS_DICT = {"sufficiency": self._sufficiency}
        opts: Checkbox = ["sufficiency"]

        return self.run_metric(self.sufficiency_dataset, METRICS_DICT, opts)

    def drift_detection(self) -> dict:
        METRICS_DICT = {"mmd": self._mmd, "ks": self._ks, "cvm": self._cvm}

        return self.run_comparison_metric(METRICS_DICT)

    def linting(self) -> dict:
        return {}

    def outlier_detection(self) -> dict:
        # outlier_detector = AEOutlier(create_model(AE, (28, 28, 1)))
        # TODO: Need to swap PyTorch NCHW to TensorFlow NHWC
        # self.outlier_detector.fit(train_dataset)
        # output = self.outlier_detector.predict(val_dataset)
        # return output
        return {}

    def _ber(self, name) -> dict:
        images = self.cached_images[name]
        labels = self.cached_labels[name]

        ber = dm.BER(data=images[:10], labels=labels[:10])
        result = ber.evaluate()
        result.update({"max_accuracy": 1 - result["ber_lower"]})
        return result

    def _coverage(self, name) -> dict:
        return {"coverage": 0.90}

    def _parity(self, name) -> dict:
        return {"parity": 0.25}

    def _balance(self, name) -> dict:
        return {"factors": [], "classes": []}

    def _sufficiency(self, name) -> dict:
        # res = dm.Sufficiency(model, dataset).evaluate()
        res = {
            "Accuracy": [1, 2, 3],
            "__STEPS__": [4, 5, 6],
            "__PARAMS__": [7, 8, 9],
        }

        # res2 = dm.Sufficiency(self.comparison_model, dataset).evaluate()
        res2 = {
            "Accuracy": [1, 2, 3],
            "__STEPS__": [4, 5, 6],
            "__PARAMS__": [7, 8, 9],
        }

        return {"Sufficiency": {"model": res, "comparison_model": res2}}

    def _mmd(self, dataset1, dataset2):
        return {"pvalue": 1.0, "is_drift": False, "val": 0.7}

    def _ks(self, dataset1, dataset2):
        return {"pvalue": 0.8, "is_drift": True, "val": 0.1}

    def _cvm(self, dataset1, dataset2):
        return {"pvalue": 0.6, "is_drift": False, "val": 0.4}

    def create_linting_rollup(self, results, dataset):
        count = 10
        percent = count / 100
        rollup = {}
        issue_rollup = {f"Lint: Issues (%) - {dataset}": round(percent, 3)}
        dupe_rollup = {f"Lint: Duplicates (%) - {dataset}": round(percent, 3)}
        outlier_rollup = {f"Lint: Outliers (%) - {dataset}": round(percent, 3)}

        rollup.update(issue_rollup)
        rollup.update(dupe_rollup)
        rollup.update(outlier_rollup)
        return rollup

    def create_bias_rollup(self, results, dataset: str):
        total = 75_000
        percent = 0.10

        coverage = results["coverage"]
        cov_count = coverage["count"]
        cov_percent = cov_count / total

        rollup = {}
        balance_rollup = {f"Bias: Balance (%) - {dataset}": round(percent, 3)}
        coverage_rollup = {f"Bias: Coverage (%) - {dataset}": round(cov_percent, 3)}
        parity_rollup = {f"Bias: Parity (%) - {dataset}": round(percent, 3)}

        rollup.update(balance_rollup)
        rollup.update(coverage_rollup)
        rollup.update(parity_rollup)
        return rollup

    def create_feasibility_rollup(self, results, dataset):
        return {f"Max Performance - {dataset}": results["ber"]["max_accuracy"]}

    def create_sufficiency_rollup(self, results, dataset):
        return {f"Is {dataset} sufficient": True}

    def create_drift_rollup(self, results):
        drift_methods = ["ks", "mmd", "cvm"]
        return {"Has drifted": any(results[method]["is_drift"] for method in drift_methods)}

    def create_ood_rollup(self, results):
        count = 10
        percent = count / 100
        return {"OOD": round(percent, 3)}

    def collect_metrics(self) -> Dict[str, Union[int, float, bool, str]]:
        print("Returning metrics")
        rollup_dict = {}
        outputs = self.outputs
        short_name_key = {"development": "dev", "operational": "op"}

        # Per dataset metrics
        for dataset_name in ["development", "operational"]:
            dataset = outputs.get(dataset_name, {})
            name = short_name_key.get(dataset_name, "")
            linting_rollup = self.create_linting_rollup(dataset, name)
            bias_rollup = self.create_bias_rollup(dataset, name)
            feasibility_rollup = self.create_feasibility_rollup(dataset, name)
            sufficiency_rollup = self.create_sufficiency_rollup(dataset, name)

            rollup_dict.update(linting_rollup)
            rollup_dict.update(bias_rollup)
            rollup_dict.update(feasibility_rollup)
            rollup_dict.update(sufficiency_rollup)

        # Comparison metrics
        drift_rollup = self.create_drift_rollup(outputs)
        ood_rollup = self.create_ood_rollup(outputs)

        rollup_dict.update(drift_rollup)
        rollup_dict.update(ood_rollup)

        return rollup_dict

    def collect_report_consumables(self):
        print("Returning Gradient parameters")
        return self.outputs
