import logging
import pathlib

import cattrs
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from lib.data import get_task_dir, load_json
from lib.eval import ColumnTaskResults

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/column_type_inference", config_name="config.yaml")
def gather_result_tables(cfg: DictConfig) -> None:
    all_datasets = list(sorted(set(path.name for path in get_task_dir(cfg.task_name).glob("*/"))))
    all_experiments = list(sorted(set(path.name for path in get_task_dir(cfg.task_name).glob("*/experiments/*/"))))

    dataset_and_experiments = get_datasets_and_experiments(all_datasets, all_experiments, cfg)

    ####################
    # main results table
    ####################
    def remove_header_info(s):
        return s.replace("-with-headers", "").replace("-without-headers", "")

    index = list(sorted(set(remove_header_info(experiment) for experiment in all_experiments)))
    columns = []
    for dataset in all_datasets:
        columns += [(dataset, "with"), (dataset, "w/out")]

    table = pd.DataFrame(
        index=index,
        columns=columns,
        data=np.nan
    )

    for dataset, experiment, results_dir in dataset_and_experiments:
        column_task_results: ColumnTaskResults = cattrs.structure(load_json(results_dir / "column_task_results.json"),
                                                                  ColumnTaskResults)

        if "with-headers" in experiment:
            table.at[remove_header_info(experiment), (dataset, "with")] = round_score(
                column_task_results.classification_report["weighted avg"]["f1-score"])
        elif "without-headers" in experiment:
            table.at[remove_header_info(experiment), (dataset, "w/out")] = round_score(
                column_task_results.classification_report["weighted avg"]["f1-score"])
        else:
            assert False, f"The experiment '{experiment}' does not specify whether headers are included!"

    table.to_csv(get_task_dir(cfg.task_name) / "main_results.csv")

    #########################
    # data type results table
    #########################
    columns = []
    for dataset in all_datasets:
        columns += [(dataset, "non-num"), (dataset, "num")]
    table = pd.DataFrame(
        index=all_experiments,
        columns=columns,
        data=np.nan
    )

    for dataset, experiment, results_dir in dataset_and_experiments:
        column_task_results: ColumnTaskResults = cattrs.structure(load_json(results_dir / "column_task_results.json"),
                                                                  ColumnTaskResults)
        table.at[experiment, (dataset, "non-num")] = round_score(
            column_task_results.classification_report_by_data_type["non-numerical"]["weighted avg"]["f1-score"])
        table.at[experiment, (dataset, "num")] = round_score(
            column_task_results.classification_report_by_data_type["numerical"]["weighted avg"]["f1-score"])

    table.to_csv(get_task_dir(cfg.task_name) / "data_type_results.csv")


def get_datasets_and_experiments(all_datasets: list[str], all_experiments: list[str], cfg: DictConfig) -> list[
    tuple[str, str, pathlib.Path]]:
    res = []
    for dataset in all_datasets:
        dataset_dir = get_task_dir(cfg.task_name) / dataset
        for experiment in all_experiments:
            results_dir = dataset_dir / "experiments" / experiment / "results"
            if results_dir.is_dir():
                res.append((dataset, experiment, results_dir))
    return res


def round_score(v: float) -> float:
    return round(v, 2)


if __name__ == "__main__":
    gather_result_tables()
