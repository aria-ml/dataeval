import json
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Optional, TypeAlias

from test_stage import TestStage

Dataset_T: TypeAlias = List[Literal["development", "operational", "both"]]
Checkbox: TypeAlias = Optional[List[str]]


class DamlTestStage(TestStage):
    DEV_OPTS = ["development", "both"]
    OPR_OPTS = ["operational", "both"]
    BOTH = ["development", "operational"]
    _cache_dir = Path(".daml_cache")
    _cache_file = Path("cache.json")

    def __init__(
        self,
        target_performance: float = 0.0,
        linting_options: Checkbox = None,
        linting_dataset: Dataset_T = ["development", "operational"],
        bias_options: Checkbox = ["balance", "coverage", "parity"],
        bias_dataset: Dataset_T = ["development"],
        drift: bool = False,
        ood_detection: bool = False,
        feasibility_dataset: Dataset_T = ["development"],
        sufficiency_dataset: Dataset_T = ["development"],
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

        self.load_cached_results(self._cache_dir / self._cache_file)

    def run(self) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""

        if self.cache:
            self.outputs = self.cache
            return

        metric_outputs = defaultdict(dict)

        # Run metrics that take individual datasets
        for k, v in self.feasibility().items():
            metric_outputs[k].update(v)
        for k, v in self.bias.items():
            metric_outputs[k].update(v)
        for k, v in self.sufficiency().items():
            metric_outputs[k].update(v)

        # metric_outputs.update(self.drift_detection())

        self.outputs = metric_outputs

        # self._save_cache()

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
        # Update cache
        for dataset, metric in self.outputs.items():
            if dataset not in self.cache:
                self.cache[dataset] = {}
            self.cache[dataset].update(metric)

        # Check path
        folder = self._cache_dir
        folder.mkdir(exist_ok=True)
        cache_file_path = folder / self._cache_file

        # Save results
        with cache_file_path.open("w+") as f:
            json.dump(self.cache, f, indent=4, sort_keys=True)

    def collect_metrics(self):
        print("Returning metrics")
        return self.outputs

    def collect_report_comsumables(self):
        print("Returning Gradient parameters")
        return self.outputs

    def get_dataset(self, select: str):
        if select == "development":
            return {"development": self.dev_dataset}
        elif select == "operational":
            return {"operational": self.operational_dataset}
        else:
            return {"development": self.dev_dataset, "operational": self.operational_dataset}

    def run_metric(self, select, metric_dict, opts):
        output = defaultdict(dict)

        for dataset in self.get_dataset(select=select):
            results = {}
            for name in opts:
                result = metric_dict[name](dataset)
                results.update(result)
            output[dataset].update(results)

        return output

    def feasibility(self):
        METRICS_DICT = {"ber": self._ber}
        opts = ["ber"]

        feasibility_output = self.run_metric(self.feasibility_dataset, METRICS_DICT, opts)

        return feasibility_output

    def _ber(self, dataset) -> dict:
        return {"ber": {"ber": 0.18, "ber_lower": 0.095}}

    def bias(self) -> dict:
        METRICS_DICT = {"balance": self._balance, "coverage": self._coverage, "parity": self._parity}
        opts: Checkbox = self.bias_opts

        bias_output = self.run_metric(self.bias_dataset, METRICS_DICT, opts)

        return bias_output

    def _coverage(self, dataset) -> dict:
        return {"coverage": {"value": 0.90}}

    def _parity(self, dataset) -> dict:
        return {"parity": {"value": 0.25}}

    def _balance(self, dataset) -> dict:
        return {"balance": {"value": 0.5}}

    def linting(self) -> dict:
        return {}

    def sufficiency(self) -> dict:
        METRICS_DICT = {"sufficiency": self._sufficiency}
        opts = ["sufficiency"]

        sufficiency_output = self.run_metric(self.sufficiency_dataset, METRICS_DICT, opts)

        return sufficiency_output

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

        output = {"sufficiency": {"model": [1, 2, 3], "comp_model": [2, 3, 4], "steps": [1, 2, 3]}}

        return output

    def outlier_detection(self) -> dict:
        # outlier_detector = AEOutlier(create_model(AE, (28, 28, 1)))
        # TODO: Need to swap PyTorch NCHW to TensorFlow NHWC
        # self.outlier_detector.fit(train_dataset)
        # output = self.outlier_detector.predict(val_dataset)
        # return output
        return {}

    def drift_detection(self) -> dict:
        return {}
