import collections
import json
import logging
import os
import pathlib
import statistics

import attrs
import hydra
import pandas as pd
import tqdm
from hydra.core.config_store import ConfigStore

from lib.data import get_data_path
from lib.eval import compute_table_sparsity

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    dataset_dir: pathlib.Path = "data/column_type_inference/gittablesCTA/download/tables_without_column_names"
    pattern: str = "*.csv"
    limit: int | None = None
    result_file: str = "gittablesCTA.json"


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    parsing_failed = 0

    num_rows = []
    num_cols = []
    sparsity = []
    data_types = collections.Counter()

    assert cfg.dataset_dir.exists(), f"Couldn't find {cfg.dataset_dir}"

    logger.info("Glob.")
    all_paths = list(cfg.dataset_dir.glob(cfg.pattern))
    logger.info(f"Found {len(all_paths)} files.")

    assert len(all_paths) > 0

    logger.info("Analyze.")
    for ix, path in enumerate(tqdm.tqdm(all_paths[:cfg.limit], desc="analyze")):
        try:
            df = pd.read_csv(path)
            num_rows.append(len(df.index))
            num_cols.append(len(df.columns))
            if len(df.index) * len(df.columns) != 0:
                sparsity.append(compute_table_sparsity(df))
            for d_type in df.dtypes.tolist():
                if d_type.kind in ("i", "f", "u"):
                    data_types["numerical"] += 1
                else:
                    data_types["non-numerical"] += 1
        except:
            parsing_failed += 1

    path = get_data_path() / "analyze_datasets"
    os.makedirs(path, exist_ok=True)
    characteristics = {
        "number of tables": len(all_paths),
        "tables per database": None,  # GitTables has not databases
        "mean columns per table": sum(num_cols) / len(num_cols),
        "median columns per table": statistics.median(num_cols),
        "95th columns per table": statistics.quantiles(num_cols, n=20)[-1],
        "mean rows per table": sum(num_rows) / len(num_rows),
        "median rows per table": statistics.median(num_rows),
        "95th rows per table": statistics.quantiles(num_rows, n=20)[-1],
        "sparsity": sum(sparsity) / len(sparsity),
        "non-numerical columns": data_types["non-numerical"] / data_types.total(),
        "numerical columns": data_types["numerical"] / data_types.total(),
        "parsing_failed": parsing_failed
    }
    with open(path / cfg.result_file, "w", encoding="utf-8") as file:
        json.dump(characteristics, file)
    logger.info("Done!")


if __name__ == "__main__":
    main()
