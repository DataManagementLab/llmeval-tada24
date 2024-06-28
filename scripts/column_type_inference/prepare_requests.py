import logging
import random

import hydra
import pandas as pd
import tqdm
from omegaconf import DictConfig, OmegaConf

from lib.data import get_instances_dir, get_requests_dir, load_json, load_str, dump_json
from lib.linearize import linearize_table, linearize_list
from lib.prompt import sample_examples, sample_rows, fill_chat_template, max_tokens_for_ground_truth

logger = logging.getLogger(__name__)

_prepare_requests_random = random.Random(859962185)


@hydra.main(version_base=None, config_path="../../config/column_type_inference", config_name="config.yaml")
def prepare_requests(cfg: DictConfig) -> None:
    instances_dir = get_instances_dir(cfg.task_name, cfg.dataset.dataset_name, cfg.exp_name)
    requests_dir = get_requests_dir(cfg.task_name, cfg.dataset.dataset_name, cfg.exp_name, clear=True)

    all_column_types = load_json(instances_dir / "all_column_types.json")
    all_column_types = list(sorted(set(filter(lambda x: x is not None, all_column_types))))
    linearized_all_column_types = linearize_list(all_column_types, **cfg.linearize_list)

    instance_paths = list(sorted(instances_dir.glob("*/")))
    for path in tqdm.tqdm(instance_paths,
                          f"{cfg.task_name} - {cfg.dataset.dataset_name} - {cfg.exp_name} - prepare requests"):
        logger.debug("Load instance.")
        table_name = load_str(path / "table_name.txt")
        df = pd.read_csv(path / "table.csv")
        column_types = load_json(path / "column_types.json")

        df = sample_rows(df, **cfg.sample_rows)
        linearized_table = linearize_table(df, table_name, **cfg.linearize_table)

        inst_all_column_types = set(column_types)

        examples = []
        for ex_path in sample_examples(path, instance_paths, **cfg.sample_examples):
            logger.debug("Load example.")
            ex_table_name = load_str(ex_path / "table_name.txt")
            ex_df = pd.read_csv(ex_path / "table.csv")
            ex_column_types = load_json(ex_path / "column_types.json")
            inst_all_column_types = inst_all_column_types.union(ex_column_types)

            if cfg.remove_unspecified_columns_in_example:
                ex_columns = [col for ix, col in enumerate(ex_df.columns.tolist()) if ex_column_types[ix] is not None]
                ex_column_types = [col_type for col_type in ex_column_types if col_type is not None]
                ex_df = ex_df[ex_columns]

            if cfg.limit_example_columns is not None:
                if cfg.limit_example_columns < len(ex_df.columns):
                    ex_column_indices = list(sorted(
                        _prepare_requests_random.sample(list(range(len(ex_df.columns))), k=cfg.limit_example_columns)))
                    ex_columns = ex_df.columns.tolist()
                    ex_columns = [ex_columns[ix] for ix in ex_column_indices]
                    ex_column_types = [ex_column_types[ix] for ix in ex_column_indices]
                    ex_df = ex_df[ex_columns]

            ex_df = sample_rows(ex_df, **cfg.sample_rows)
            ex_linearized_table = linearize_table(ex_df, ex_table_name, **cfg.linearize_table)
            ex_linearized_column_types = linearize_list(stringify_unspecified_column_types(ex_column_types, cfg),
                                                        **cfg.linearize_list)

            examples.append(
                {
                    "table": ex_linearized_table,
                    "column_types": ex_linearized_column_types
                }
            )
        ground_truth = linearize_list(stringify_unspecified_column_types(column_types, cfg), **cfg.linearize_list)

        if cfg.use_inst_all_column_types:
            if len(inst_all_column_types) < cfg.num_inst_all_column_types:
                remaining_column_types = set(all_column_types).difference(inst_all_column_types)
                required_num = cfg.num_inst_all_column_types - len(inst_all_column_types)
                if required_num > len(remaining_column_types):
                    logger.warning(
                        f"There are not enough column types in total to achieve `cfg.num_inst_all_column_types`!")
                    required_num = len(remaining_column_types)
                inst_all_column_types = inst_all_column_types.union(
                    _prepare_requests_random.sample(list(remaining_column_types), k=required_num))
            else:
                logger.warning(
                    f"All column types for instance is already more than `cfg.num_inst_all_column_types` ({len(inst_all_column_types)} > {cfg.num_inst_all_column_types})!")

            all_column_types = list(sorted(set(filter(lambda x: x is not None, inst_all_column_types))))
            linearized_all_column_types = linearize_list(all_column_types, **cfg.linearize_list)

        request = {
            "model": cfg.model,
            "max_tokens": max_tokens_for_ground_truth(ground_truth, cfg.api_name, cfg.model,
                                                      cfg.max_tokens_over_ground_truth),
            "temperature": cfg.temperature
        }

        example_messages = []
        for example in examples:
            example_messages += fill_chat_template(OmegaConf.to_container(cfg.example_chat_template), **example)
        request["messages"] = fill_chat_template(
            OmegaConf.to_container(cfg.prompt_chat_template),
            all_column_types=linearized_all_column_types,
            examples=example_messages,
            table=linearized_table
        )

        dump_json(request, requests_dir / f"{path.name}.json")


def stringify_unspecified_column_types(column_types: list[str | None], cfg: DictConfig) -> list[str]:
    return [ct if ct is not None else cfg.unspecified_column_type_string for ct in column_types]


if __name__ == "__main__":
    prepare_requests()
