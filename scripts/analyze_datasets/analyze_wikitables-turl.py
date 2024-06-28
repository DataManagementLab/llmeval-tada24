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

from lib.data import get_data_path, load_json
from lib.eval import compute_table_sparsity

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    dataset_dir: pathlib.Path = "data/column_type_inference/wikitables-turl/download"
    limit: int | None = None
    result_file: str = "wikitables-turl.json"


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    parsing_failed = 0

    num_rows = []
    num_cols = []
    num_annotated_cols = []
    sparsity = []
    data_types = collections.Counter()

    assert cfg.dataset_dir.exists(), f"Couldn't find {cfg.dataset_dir}"

    logger.info("Read json file.")

    data = load_json(cfg.dataset_dir / "train.table_col_type.json")

    assert len(data) > 0
    logger.info(f"Found {len(data)} tables in the json file")

    logger.info("Analyze.")
    for ix, table in enumerate(tqdm.tqdm(data[:cfg.limit], desc="analyze")):
        try:
            column_headers = table[5]
            table_content = table[6]
            cta_annotations = table[7]

            data_dict = {}

            num_cols.append(len(column_headers))
            num_annotated_cols.append(len(cta_annotations))

            column_row_counts = []

            # Extracting column names and values
            columns = column_headers
            for col_idx, column in enumerate(table_content):
                data_dict[column_headers[col_idx]] = []

                for row in column:
                    data_dict[column_headers[col_idx]].append((row[0], row[1][1]))

                column_row_counts.append(len(column))

            # find out maximum number by looking for the highest row index:
            max_rows = 0
            for column in data_dict.keys():
                highest_row_index_in_col = max([index[0] for index, _ in data_dict[column]])
                max_rows = max(max_rows, highest_row_index_in_col)
            max_rows = max_rows + 1  # need to add one for index 0

            # Every column needs to have max_rows rows, need to check indexes:
            for column in data_dict.keys():
                if len(data_dict[column]) < max_rows:
                    final_row = [None for x in range(max_rows)]
                    assert len(final_row) == max_rows
                    # add values where we have them, the rest stays None
                    for index, value in data_dict[column]:
                        final_row[index[0]] = value

                    data_dict[column] = final_row

            # Creating DataFrame
            df = pd.DataFrame(data_dict, columns=columns)

            num_rows.append(len(df.index))
            num_cols.append(len(df.columns))
            if len(df.index) * len(df.columns) != 0:
                sparsity.append(compute_table_sparsity(df))
            for d_type in df.dtypes.tolist():
                if d_type.kind in ("i", "f", "u"):
                    data_types["numerical"] += 1
                else:
                    data_types["non-numerical"] += 1
        except Exception as e:
            print("-----------")
            print("Error was: ", e)
            print(column_row_counts)
            print(f"Final row: {len(final_row)}, {final_row}")
            print("Table: ", table)
            print(f"Num headers: {len(column_headers)}")
            print(f"Table content length: {len(table_content)}")
            print(f"Data dict {data_dict}")
            parsing_failed += 1
            raise NotImplementedError

    print(f"Num rows and cols list lengths: {len(num_rows)}, {len(num_cols)} ")

    path = get_data_path() / "analyze_datasets"
    os.makedirs(path, exist_ok=True)
    characteristics = {
        "number of tables": len(data),
        "tables per database": None,  # wikitables-turl has not databases
        "mean columns per table": sum(num_cols) / len(num_cols),
        "median columns per table": statistics.median(num_cols),
        "95th columns per table": statistics.quantiles(num_cols, n=20)[-1],
        "mean rows per table": sum(num_rows) / len(num_rows),
        "median rows per table": statistics.median(num_rows),
        "95th rows per table": statistics.quantiles(num_rows, n=20)[-1],
        "sparsity": sum(sparsity) / len(sparsity),
        "non-numerical columns": data_types["non-numerical"] / data_types.total(),
        "numerical columns": data_types["numerical"] / data_types.total(),
        "annotated columns": sum(num_annotated_cols),
        "parsing_failed": parsing_failed
    }
    with open(path / cfg.result_file, "w", encoding="utf-8") as file:
        json.dump(characteristics, file)
    logger.info("Done!")


if __name__ == "__main__":
    main()
