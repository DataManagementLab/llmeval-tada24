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
    dataset_dir: pathlib.Path = "data/column_type_inference/sportstables/download"
    pattern: str = "*/*.csv"
    limit: int | None = None


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    num_rows = []
    num_cols = []
    sparsity = []
    data_types = collections.Counter()

    logger.info("Glob.")
    all_paths = list(cfg.dataset_dir.glob(cfg.pattern))

    logger.info("Analyze.")
    for path in tqdm.tqdm(all_paths[:cfg.limit], desc="analyze"):
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

    path = get_data_path() / "analyze_datasets"
    os.makedirs(path, exist_ok=True)
    characteristics = {
        "number of tables": len(all_paths),
        "tables per database": None,  # SportsTables has not databases
        "mean columns per table": sum(num_cols) / len(num_cols),
        "median columns per table": statistics.median(num_cols),
        "95th columns per table": statistics.quantiles(num_cols, n=20)[-1],
        "mean rows per table": sum(num_rows) / len(num_rows),
        "median rows per table": statistics.median(num_rows),
        "95th rows per table": statistics.quantiles(num_rows, n=20)[-1],
        "sparsity": sum(sparsity) / len(sparsity),
        "non-numerical columns": data_types["non-numerical"] / data_types.total(),
        "numerical columns": data_types["numerical"] / data_types.total()
    }
    with open(path / "sportstables.json", "w", encoding="utf-8") as file:
        json.dump(characteristics, file)
    logger.info("Done!")


if __name__ == "__main__":
    main()
