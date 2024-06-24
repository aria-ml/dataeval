import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union

import numpy as np
import pandas as pd
from test_stage import TestStage

Dataset_T: TypeAlias = List[Literal["dev", "op"]]
Checkbox: TypeAlias = Optional[List[str]]


class DamlTestStage(TestStage):
    _cache_dir = Path(".daml_cache")
    # _cache_file = Path("cache.json")
    _cache_file = Path("demo_cache.json")

    def __init__(
        self,
        config_json,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # self.config = json.loads(sample_config_json)
        self.config = config_json

        # Runtime caching <Multiple metrics need preprocessing by an AE>
        self.cached_ae_path = None
        self.cached_images = {}  # {dataset: embeddings}
        self.cached_labels = {}  # {dataset: embeddings}

    def run(self) -> None:
        """Run the test stage, and store any outputs of the evaluation in test stage"""

        self.ds = {
            "dev": self.dev_dataset,
            "op": self.operational_dataset,
        }

        self.load_cached_results(self._cache_dir / self._cache_file)
        if self.cache:
            self.outputs = self.cache
            return

        METRICS_MAP = {
            # LINTING
            "image properties": self._image_properties,
            "visual quality": self._visual_quality,
            "duplicates": self._duplicates,
            "outliers": self._outliers,
            # BIAS
            "balance": self._balance,
            "coverage": self._coverage,
            "parity": self._parity,
            # FEASIBILITY
            "feasibility": self._ber,
            "sufficiency": self._sufficiency,
            # DATASET SHIFT
            "drift": self._drift,
            "out-of-distribution": self._ood,
        }

        outputs = {k: {} for k in self.config}

        for dataset, metric_selection in self.config.items():
            for metric in metric_selection:
                if metric in METRICS_MAP and dataset in self.ds:
                    outputs[dataset].update({metric: METRICS_MAP[metric](self.ds[dataset])})

        self.outputs = outputs

        # self._save_cache()

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

    def _save_cache(self) -> None:
        # Check path
        folder = self._cache_dir
        folder.mkdir(exist_ok=True)
        cache_file_path = folder / self._cache_file

        # Save results
        with cache_file_path.open("w+") as f:
            json.dump(self.outputs, f, indent=4, sort_keys=True)

    def _image_properties(self, dataset) -> dict:
        return {}

    def _visual_quality(self, dataset) -> dict:
        return {}

    def _duplicates(self, dataset) -> dict:
        return {"indices": list(range(10))}

    def _outliers(self, dataset) -> dict:
        return {"indices": list(range(10))}

    def _ood(self, dataset) -> dict:
        return {"indices": list(range(10))}

    def _ber(self, dataset) -> dict:
        return {"ber": 0.18, "ber_lower": 0.095, "max_performance": 0.82}

    def _coverage(self, dataset) -> dict:
        scores = np.arange(100)
        threshold = 90
        uncovered = scores[scores > threshold].tolist()
        count = len(uncovered)

        results = {"count": count, "threshold": threshold, "image_paths": ["img.png"], "scores": uncovered}

        return results

    def _parity(self, dataset) -> dict:
        return {"parity": 0.25}

    def _balance(self, dataset) -> dict:
        return {"factors": [str(x) for x in range(10)], "mutual_information": [[i] * 10 for i in range(10)]}

    def _sufficiency(self, dataset) -> dict:
        # res = dm.Sufficiency(self.model, dataset).evaluate()
        res = {
            "performance": [1, 2, 3],
            "__STEPS__": [4, 5, 6],
            "__PARAMS__": [7, 8, 9],
        }

        # res2 = dm.Sufficiency(self.comparison_model, dataset).evaluate()
        res2 = {
            "performance": [1, 2, 3],
            "__STEPS__": [4, 5, 6],
            "__PARAMS__": [7, 8, 9],
        }

        return {"model": res, "comparison_model": res2}

    def _drift(self, dataset):
        return {
            "mmd": self._mmd(self.dev_dataset, dataset),
            "ks": self._ks(self.dev_dataset, dataset),
            "cvm": self._cvm(self.dev_dataset, dataset),
        }

    def _mmd(self, dataset1, dataset2):
        return {"pvalue": 1.0, "is_drift": False, "val": 0.7}

    def _ks(self, dataset1, dataset2):
        return {"pvalue": 0.8, "is_drift": True, "val": 0.1}

    def _cvm(self, dataset1, dataset2):
        return {"pvalue": 0.6, "is_drift": False, "val": 0.4}

    def collect_metrics(self) -> Dict[str, Union[int, float, bool, str]]:
        print("Returning metrics")
        ROLLUP_MAP = {
            # LINTING
            "image properties": self._rollup_image_properties,
            "visual quality": self._rollup_visual_quality,
            "duplicates": self._rollup_duplicates,
            "outliers": self._rollup_outliers,
            # BIAS
            "balance": self._rollup_balance,
            "coverage": self._rollup_coverage,
            "parity": self._rollup_parity,
            # FEASIBILITY
            "feasibility": self._rollup_ber,
            "sufficiency": self._rollup_sufficiency,
            # DATASET SHIFT
            "drift": self._rollup_drift,
            "out-of-distribution": self._rollup_ood,
        }
        rollup_dict = {}
        outputs = self.outputs

        for dataset_name, metric_selection in self.config.items():
            dataset_results = outputs[dataset_name]
            for metric in metric_selection:
                x = ROLLUP_MAP[metric](dataset_results[metric], dataset_name)
                rollup_dict.update(x)

        return rollup_dict

    def _rollup_image_properties(self, results, name):
        return {f"Issues - {name}": 0}

    def _rollup_visual_quality(self, results, name):
        return {f"Visual Quality - {name}": 1}

    def _rollup_duplicates(self, results, name):
        percent = 0.0 if self.ds[name] is None else len(results["indices"]) / len(self.ds[name])
        return {f"Duplicates (%) - {name}": round(percent, 3)}

    def _rollup_outliers(self, results, name):
        percent = 0.0 if self.ds[name] is None else len(results["indices"]) / len(self.ds[name])
        return {f"Outliers (%) - {name}": round(percent, 3)}

    def _rollup_coverage(self, results, name):
        percent = 0.0 if self.ds[name] is None else results["count"] / len(self.ds[name])
        return {f"Coverage (%) - {name}": round(percent, 3)}

    def _rollup_balance(self, results, name):
        rollup = np.mean(np.array(results["mutual_information"])[0, 1:] > 0.5)

        return {f"Balance (%) - {name}": rollup}

    def _rollup_parity(self, results, name):
        return {f"Parity (%) - {name}": 0.10}

    def _rollup_ber(self, results, name):
        return {f"Max Performance - {name}": results["max_performance"]}

    def _rollup_sufficiency(self, results, name):
        print(results)

        if self.target_performance is None:
            return {f"Is {name} sufficient": True}

        model_res = results.get("model", {})
        comp_model_res = results.get("comparison_model", {})

        model_max_perf = model_res.get("performance", [-1]) if model_res else [-1]
        comp_model_max_perf = comp_model_res.get("performance", [-1]) if comp_model_res else [-1]

        print(self.target_performance)
        print(max(model_max_perf))
        print(max(comp_model_max_perf))

        rollup = (max(model_max_perf) > self.target_performance) and (
            max(comp_model_max_perf) > self.target_performance
        )

        return {f"Is {name} sufficient": rollup}

    def _rollup_drift(self, results, name):
        drift_methods = ["ks", "mmd", "cvm"]
        return {"Has drifted": any(results[method]["is_drift"] for method in drift_methods)}

    def _rollup_ood(self, results, name):
        count = 10
        percent = count / 100
        return {"OOD": round(percent, 3)}

    def collect_report_consumables(self) -> Dict[str, Any]:
        """
        Can currently only test one gradient report slide at a time with return type Dict[str, Any]
        This is being fixed to return List[Dict[str, Any]] so we can return all slides generated
        The return dict format is as follows:
        return {
            "deck_name": "image_classification",
            "slide_name": "Feasibility Table",
            "table_content": mock_df,
            "text": "This is mock data",
        }

        Where the keys are set (for this slide format), and the values are determined by calculations
        """
        print("Returning Gradient parameters")

        # Example feasibility
        mock_dict = {
            "Is Feasible": [True],
            "Bayes Error Rate": [1.0],
            "Lower Bayes Error Rate": [0.0],
            "Maximum Performance": [1.0],
        }
        mock_df = pd.DataFrame.from_dict(mock_dict)

        return {
            "deck_name": "image_classification",
            "slide_name": "Feasibility Table",
            "table_content": mock_df,
            "text": "This is mock data",
        }

    def _report_balance(self):
        pass
