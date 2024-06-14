import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, TypeAlias

from test_stage import TestStage

Dataset_T: TypeAlias = List[Literal["development", "operational"]]
Checkbox: TypeAlias = Optional[List[str]]


class DamlTestStage(TestStage):
    _cache_dir = Path(".daml_cache")
    _cache_file = Path("cache.json")

    def __init__(
        self,
        target_performance: float = 0.0,
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
        self.target_performance = target_performance

        self.linting_opts = linting_options if linting_options else []
        self.linting_ds = linting_dataset

        self.bias_opts = bias_options if bias_options else []
        self.bias_dataset = bias_dataset

        self.do_drift = drift
        self.do_ood_detection = ood_detection

        self.feasibility_dataset = feasibility_dataset
        self.sufficiency_dataset = sufficiency_dataset

        # Runtime caching <Multiple metrics need preprocessing by an AE>
        self.cached_ae_path = None
        self.cached_images = {}  # {dataset: embeddings}
        self.cached_labels = {}  # {dataset: embeddings}

    def run(self) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""

        self.load_cached_results(self._cache_dir / self._cache_file)
        if self.cache:
            self.outputs = self.cache
            # return

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

    def collect_metrics(self):
        print("Returning metrics")
        return self.outputs

    def collect_report_comsumables(self):
        print("Returning Gradient parameters")
        return self.outputs

    def get_dataset(self, select: Dataset_T) -> dict:
        ds = {"development": self.dev_dataset, "operational": self.operational_dataset}
        return {name: ds[name] for name in select}

    def run_metric(self, select: Dataset_T, metric_dict: Dict[str, Callable], opts: List[str]):
        """
        Sets the dataset name as outer key, metric as inner key, dict output as value

        ex. {"development": "ber": {"ber": 0.18, "ber_lower"}}
        """
        output = defaultdict(dict)

        for name, dataset in self.get_dataset(select).items():
            output[name].update({metric: metric_dict[metric](dataset) for metric in opts})

        return output

    def run_comparison_metric(self, metric_dict):
        """Run metrics that take in two datasets"""
        output = {name: metric(self.dev_dataset, self.operational_dataset) for name, metric in metric_dict.items()}
        return output

    def feasibility(self):
        METRICS_DICT = {"ber": self._ber}
        opts: Checkbox = ["ber"]

        feasibility_output = self.run_metric(self.feasibility_dataset, METRICS_DICT, opts)

        return feasibility_output

    def bias(self) -> dict:
        METRICS_DICT = {"balance": self._balance, "coverage": self._coverage, "parity": self._parity}
        opts: Checkbox = self.bias_opts

        bias_output = self.run_metric(self.bias_dataset, METRICS_DICT, opts)

        return bias_output

    def sufficiency(self) -> dict:
        METRICS_DICT = {"sufficiency": self._sufficiency}
        opts: Checkbox = ["sufficiency"]

        sufficiency_output = self.run_metric(self.sufficiency_dataset, METRICS_DICT, opts)

        return sufficiency_output

    def linting(self) -> dict:
        return {}

    def outlier_detection(self) -> dict:
        # outlier_detector = AEOutlier(create_model(AE, (28, 28, 1)))
        # TODO: Need to swap PyTorch NCHW to TensorFlow NHWC
        # self.outlier_detector.fit(train_dataset)
        # output = self.outlier_detector.predict(val_dataset)
        # return output
        return {}

    def drift_detection(self) -> dict:
        METRICS_DICT = {"mmd": self._mmd, "ks": self._ks}
        drift_output = self.run_comparison_metric(METRICS_DICT)

        return drift_output

    def _ber(self, dataset) -> dict:
        return {"ber": 0.18, "ber_lower": 0.095}

    def _coverage(self, dataset) -> dict:
        return {"value": 0.90}

    def _parity(self, dataset) -> dict:
        return {"value": 0.25}

    def _balance(self, dataset) -> dict:
        return {"factors": [], "classes": []}

    def _sufficiency(self, dataset) -> dict:
        """
        res = daml.Sufficiency(model, dataset).evaluate()
        res -->
        {
            "Accuracy": [1, 2, 3],
            "steps": [4, 5, 6]
        }

        res2 = daml.Sufficiency(model, dataset).evaluate()
        res2 -->
        {
            "Accuracy": [1, 2, 3],
            "steps": [4, 5, 6]
        }

        model1_res = { "model": res["Accuracy"], "comp_model": res2["Accuracy"], "steps": res["__STEPS__"] }

        """

        output = {"model": [1, 2, 3], "comp_model": [2, 3, 4], "steps": [1, 2, 3]}

        return output

    def _mmd(self, dataset1, dataset2):
        return {"pvalue": 1.0, "is_drift": True, "val": 0.7}

    def _ks(self, dataset1, dataset2):
        return {"pvalue": 0.8, "is_drift": False, "val": 0.1}
