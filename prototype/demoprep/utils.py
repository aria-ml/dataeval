from typing import Any, Dict, Sequence

import torch.nn as nn
from test_stage import TestStage
from torch.utils.data import Dataset

# import maite.protocols.object_detection as od
# import maite.protocols.image_classification as ic


def load_models_and_datasets(
    dev_dataset: Dataset,
    op_dataset: Dataset,
    model: nn.Module,
    comparison_model: nn.Module,
    target_performance: float,
    stages: Sequence[TestStage],
):
    for stage in stages:
        stage.load_model(model)
        stage.load_comparison_model(comparison_model)
        stage.load_development_dataset(dev_dataset)
        stage.load_operational_dataset(op_dataset)
        stage.load_target_performance(target_performance)


def run_stages(stages: Sequence[TestStage]):
    for stage in stages:
        stage.run()


def collect_metrics(stages: Sequence[TestStage]) -> Dict[str, float]:
    return_dict = {}
    for stage in stages:
        return_dict.update(stage.collect_metrics())
    return return_dict


def collect_report_consumables(stages: Sequence[TestStage]) -> Dict[str, Any]:
    return_dict = {}
    for stage in stages:
        return_dict.update(stage.collect_report_comsumables())
    return return_dict
