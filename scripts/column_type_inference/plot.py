import logging

import cattrs
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from lib.colors import COLOR_9A, GRADIENT_9A_LIGHT
from lib.data import get_results_dir, load_json, dump_json
from lib.eval import ColumnTaskResults

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/column_type_inference", config_name="config.yaml")
def plot(cfg: DictConfig) -> None:
    results_dir = get_results_dir(cfg.task_name, cfg.dataset.dataset_name, cfg.exp_name)

    column_task_results = cattrs.structure(load_json(results_dir / "column_task_results.json"), ColumnTaskResults)
    logger.info(f"weighted F1 score: {round(column_task_results.classification_report['weighted avg']['f1-score'], 2)}")

    # plot deviation from correct number of columns
    if len(column_task_results.num_columns_deviations) == 0:
        bins = range(0, 1)
    else:
        bins = range(min(column_task_results.num_columns_deviations),
                     max(column_task_results.num_columns_deviations) + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2, 1.4)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)
    plt.hist(column_task_results.num_columns_deviations, bins=bins, color=COLOR_9A)
    plt.xlabel("deviation from correct number of columns")
    plt.ylabel("count")
    plt.savefig(results_dir / "num_columns_deviation.pdf", bbox_inches="tight")
    plt.clf()

    # plot weighted F1 scores by column index
    wf1_by_idx_x = []
    wf1_by_idx_y = []
    for k, report in column_task_results.classification_report_by_idx.items():
        wf1_by_idx_x.append(k)
        wf1_by_idx_y.append(report["weighted avg"]["f1-score"])
    wf1_by_idx_x, wf1_by_idx_y = zip(*sorted(zip(wf1_by_idx_x, wf1_by_idx_y)))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2, 1.4)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)
    plt.plot(wf1_by_idx_x, wf1_by_idx_y, mew=2, color=GRADIENT_9A_LIGHT[1], marker="x", mfc=COLOR_9A, mec=COLOR_9A)
    plt.xlabel("column idx")
    plt.ylabel("F1 score")
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.savefig(results_dir / "weighted_f1_score_by_idx.pdf", bbox_inches="tight")
    plt.clf()

    # plot missing column-adjusted weighted F1 scores at column_index
    wf1_by_idx_x = []
    wf1_by_idx_y = []
    for k, report in column_task_results.missing_column_adjusted_classification_report_by_idx.items():
        wf1_by_idx_x.append(k)
        wf1_by_idx_y.append(report["weighted avg"]["f1-score"])
    wf1_by_idx_x, wf1_by_idx_y = zip(*sorted(zip(wf1_by_idx_x, wf1_by_idx_y)))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2, 1.4)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)
    plt.plot(wf1_by_idx_x, wf1_by_idx_y, mew=2, color=GRADIENT_9A_LIGHT[1], marker="x", mfc=COLOR_9A, mec=COLOR_9A)
    plt.xlabel("column idx")
    plt.ylabel("F1 score")
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.savefig(results_dir / "missing_column_adjusted_weighted_f1_score_by_idx.pdf", bbox_inches="tight")
    plt.clf()

    # plot weighted F1 scores by data type
    wf1_by_data_type_x = []
    wf1_by_data_type_x_tick_labels = []
    wf1_by_data_type_y = []
    for ix, (data_type, report) in enumerate(column_task_results.classification_report_by_data_type.items()):
        wf1_by_data_type_x.append(ix)
        wf1_by_data_type_x_tick_labels.append(data_type)
        wf1_by_data_type_y.append(report["weighted avg"]["f1-score"])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2, 1.4)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)
    plt.bar(wf1_by_data_type_x, wf1_by_data_type_y, color=COLOR_9A)
    plt.xticks(wf1_by_data_type_x, labels=wf1_by_data_type_x_tick_labels)
    plt.ylabel("F1 score")
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.savefig(results_dir / "weighted_f1_score_by_data_type.pdf", bbox_inches="tight")
    plt.clf()

    # plot missing column-adjusted weighted F1 scores by data type
    wf1_by_data_type_x = []
    wf1_by_data_type_x_tick_labels = []
    wf1_by_data_type_y = []
    for ix, (data_type, report) in enumerate(
            column_task_results.missing_column_adjusted_classification_report_by_data_type.items()):
        wf1_by_data_type_x.append(ix)
        wf1_by_data_type_x_tick_labels.append(data_type)
        wf1_by_data_type_y.append(report["weighted avg"]["f1-score"])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2, 1.4)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)
    plt.bar(wf1_by_data_type_x, wf1_by_data_type_y, color=COLOR_9A)
    plt.xticks(wf1_by_data_type_x, labels=wf1_by_data_type_x_tick_labels)
    plt.ylabel("F1 score")
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.savefig(results_dir / "missing_column_adjusted_weighted_f1_score_by_data_type.pdf", bbox_inches="tight")
    plt.clf()

    # plot weighted F1 scores by sparsity
    wf1_by_sparsity_x = []
    wf1_by_sparsity_y = []
    for sparsity, report in column_task_results.classification_report_by_sparsity.items():
        wf1_by_sparsity_x.append(sparsity)
        wf1_by_sparsity_y.append(report["weighted avg"]["f1-score"])
    wf1_by_sparsity_x, wf1_by_sparsity_y = zip(*sorted(zip(wf1_by_sparsity_x, wf1_by_sparsity_y)))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2, 1.4)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)
    plt.plot(wf1_by_sparsity_x, wf1_by_sparsity_y, mew=2, color=GRADIENT_9A_LIGHT[1], marker="x", mfc=COLOR_9A,
             mec=COLOR_9A)
    plt.xlabel("fraction of empty cells")
    plt.ylabel("F1 score")
    plt.xlim((0, 1))
    plt.xticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.savefig(results_dir / "weighted_f1_score_by_sparsity.pdf", bbox_inches="tight")
    plt.clf()

    # plot missing column-adjusted weighted F1 scores by sparsity
    wf1_by_sparsity_x = []
    wf1_by_sparsity_y = []
    for sparsity, report in column_task_results.missing_column_adjusted_classification_report_by_sparsity.items():
        wf1_by_sparsity_x.append(sparsity)
        wf1_by_sparsity_y.append(report["weighted avg"]["f1-score"])
    wf1_by_sparsity_x, wf1_by_sparsity_y = zip(*sorted(zip(wf1_by_sparsity_x, wf1_by_sparsity_y)))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2, 1.4)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)
    plt.plot(wf1_by_sparsity_x, wf1_by_sparsity_y, mew=2, color=GRADIENT_9A_LIGHT[1], marker="x", mfc=COLOR_9A,
             mec=COLOR_9A)
    plt.xlabel("fraction of empty cells")
    plt.ylabel("F1 score")
    plt.xlim((0, 1))
    plt.xticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.savefig(results_dir / "missing_column_adjusted_weighted_f1_score_by_sparsity.pdf", bbox_inches="tight")
    plt.clf()

    # plot weighted F1 scores by number of columns
    wf1_by_num_columns_x = []
    wf1_by_num_columns_y = []
    for num_columns, report in column_task_results.classification_report_by_num_columns.items():
        wf1_by_num_columns_x.append(num_columns)
        wf1_by_num_columns_y.append(report["weighted avg"]["f1-score"])
    wf1_by_num_columns_x, wf1_by_num_columns_y = zip(*sorted(zip(wf1_by_num_columns_x, wf1_by_num_columns_y)))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2, 1.4)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)
    plt.plot(wf1_by_num_columns_x, wf1_by_num_columns_y, mew=2, color=GRADIENT_9A_LIGHT[1], marker="x", mfc=COLOR_9A,
             mec=COLOR_9A)
    plt.xlabel("number of columns")
    plt.ylabel("F1 score")
    plt.xlim((0, 100))
    plt.xticks((0, 25, 50, 75, 100), labels=("0", "", "50", "", "100"))
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.savefig(results_dir / "weighted_f1_score_by_num_columns.pdf", bbox_inches="tight")
    plt.clf()

    # plot missing column-adjusted accuracies by number of columns
    wf1_by_num_columns_x = []
    wf1_by_num_columns_y = []
    for num_columns, report in column_task_results.missing_column_adjusted_classification_report_by_num_columns.items():
        wf1_by_num_columns_x.append(num_columns)
        wf1_by_num_columns_y.append(report["weighted avg"]["f1-score"])
    wf1_by_num_columns_x, wf1_by_num_columns_y = zip(*sorted(zip(wf1_by_num_columns_x, wf1_by_num_columns_y)))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2, 1.4)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)
    plt.plot(wf1_by_num_columns_x, wf1_by_num_columns_y, mew=2, color=GRADIENT_9A_LIGHT[1], marker="x", mfc=COLOR_9A,
             mec=COLOR_9A)
    plt.xlabel("number of columns")
    plt.ylabel("F1 score")
    plt.xlim((0, 100))
    plt.xticks((0, 25, 50, 75, 100), labels=("0", "", "50", "", "100"))
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    plt.savefig(results_dir / "missing_column_adjusted_weighted_f1_score_by_num_columns.pdf", bbox_inches="tight")
    plt.clf()

    # store weighted f1-score by column type
    f1_by_column_type = []
    for column_type, confusion in column_task_results.classification_report.items():
        if column_type not in ("weighted avg", "macro avg", "accuracy"):
            f1_by_column_type.append((column_type, confusion["f1-score"]))

    f1_by_column_type.sort(key=lambda x: x[1], reverse=True)
    dump_json(f1_by_column_type, results_dir / "f1_by_column_type.json")

    dump_json(column_task_results.not_even_a_column_type, results_dir / "not_even_a_column_type.json")


if __name__ == "__main__":
    plot()
