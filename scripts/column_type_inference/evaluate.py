import collections
import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from lib.data import get_instances_dir, get_results_dir, get_responses_dir, load_json, dump_json
from lib.eval import extract_text_from_response, ColumnTaskResults, compute_table_sparsity
from lib.linearize import delinearize_list

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/column_type_inference", config_name="config.yaml")
def evaluate(cfg: DictConfig) -> None:
    instances_dir = get_instances_dir(cfg.task_name, cfg.dataset.dataset_name, cfg.exp_name)
    responses_dir = get_responses_dir(cfg.task_name, cfg.dataset.dataset_name, cfg.exp_name)
    results_dir = get_results_dir(cfg.task_name, cfg.dataset.dataset_name, cfg.exp_name, clear=True)

    all_column_types = load_json(instances_dir / "all_column_types.json")

    finish_reasons = collections.Counter()

    all_true_column_types = []  # [['type-a', 'type-b', 'type-c'], ['type-x', 'type-z'], ...]
    all_pred_column_types = []  # [['type-a', 'type-b', 'type-c'], ['type-x', 'type-z'], ...]
    all_data_types = []  # [['numerical', 'non-numerical', 'non-numerical'], ['numerical', 'non-numerical'], ...]
    all_sparsities = []  # [[0.4, 0.4, 0.4], [0.7, 0.7], ...]

    instance_dirs = list(sorted(instances_dir.glob("*/")))
    for instance_dir in instance_dirs:
        response_path = responses_dir / f"{instance_dir.name}.json"

        response_json = load_json(response_path)
        response = extract_text_from_response(response_json)
        if response is None:
            pred_column_types = []
            logger.warning("Evaluation on failed API request! ==> Interpret as empty list of column types.")
            finish_reasons["failed_api_request"] += 1
        else:
            finish_reasons[response_json["choices"][0]["finish_reason"]] += 1
            pred_column_types = delinearize_list(response, **cfg.linearize_list)
            if pred_column_types is None:
                pred_column_types = []
                logger.warning("Delinearization of column types failed! ==> Interpret as empty list of column types.")
                finish_reasons["delinearizeation_failed"] += 1
        all_pred_column_types.append(pred_column_types)

        true_column_types = load_json(instance_dir / "column_types.json")
        all_true_column_types.append(true_column_types)
        data_types = load_json(instance_dir / "data_types.json")
        all_data_types.append(data_types)
        df = pd.read_csv(instance_dir / "table.csv")
        sparsity = round(compute_table_sparsity(df), cfg.bucketize_sparsity_decimal_points)
        all_sparsities.append([sparsity] * len(true_column_types))

    annotated_columns = collections.Counter()
    for inst_true_column_types in all_true_column_types:
        for column_type in inst_true_column_types:
            annotated_columns[column_type is not None] += 1
    logger.info(f"Annotated columns: {annotated_columns}")
    dump_json(dict(annotated_columns), results_dir / "annotated_columns.json")
    dump_json(dict(finish_reasons), results_dir / "finish_reasons.json")

    column_level_task_results = ColumnTaskResults.compute(
        all_true_column_types,
        all_pred_column_types,
        all_data_types,
        all_sparsities,
        all_column_types,
        cfg.adjust_missing_columns_up_to,
        f"{cfg.task_name} - {cfg.dataset.dataset_name} - {cfg.exp_name} - evaluate"
    )
    column_level_task_results.save(results_dir / "column_task_results.json")


if __name__ == "__main__":
    evaluate()
