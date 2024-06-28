import logging

import cattrs
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from lib.colors import COLOR_1A, GRADIENT_1A_LIGHT, GRADIENT_9A_LIGHT, COLOR_9A
from lib.data import get_task_dir, load_json, get_results_dir, dump_json
from lib.eval import ColumnTaskResults

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/column_type_inference", config_name="config.yaml")
def gather_plots(cfg: DictConfig) -> None:
    task_dir = get_task_dir(cfg.task_name)
    ################################################################################################################################
    # gather results
    ################################################################################################################################
    num_columns_with_headers_results_dir = get_results_dir(cfg.task_name, cfg.dataset.dataset_name,
                                                           "sampled-columns-gpt-35-turbo-1106-with-headers")
    num_columns_with_headers_results = cattrs.structure(
        load_json(num_columns_with_headers_results_dir / "column_task_results.json"), ColumnTaskResults)
    wf1_by_num_columns_with_headers_x = []
    wf1_by_num_columns_with_headers_y = []
    for num_columns, report in num_columns_with_headers_results.classification_report_by_num_columns.items():
        wf1_by_num_columns_with_headers_x.append(num_columns)
        wf1_by_num_columns_with_headers_y.append(report["weighted avg"]["f1-score"])
    wf1_by_num_columns_with_headers_x, wf1_by_num_columns_with_headers_y = zip(
        *sorted(zip(wf1_by_num_columns_with_headers_x, wf1_by_num_columns_with_headers_y)))

    num_columns_without_headers_results_dir = get_results_dir(cfg.task_name, cfg.dataset.dataset_name,
                                                              "sampled-columns-gpt-35-turbo-1106-without-headers")
    num_columns_without_headers_results = cattrs.structure(
        load_json(num_columns_without_headers_results_dir / "column_task_results.json"), ColumnTaskResults)
    wf1_by_num_columns_without_headers_x = []
    wf1_by_num_columns_without_headers_y = []
    for num_columns, report in num_columns_without_headers_results.classification_report_by_num_columns.items():
        wf1_by_num_columns_without_headers_x.append(num_columns)
        wf1_by_num_columns_without_headers_y.append(report["weighted avg"]["f1-score"])
    wf1_by_num_columns_without_headers_x, wf1_by_num_columns_without_headers_y = zip(
        *sorted(zip(wf1_by_num_columns_without_headers_x, wf1_by_num_columns_without_headers_y)))

    sparsity_with_headers_results_dir = get_results_dir(cfg.task_name, cfg.dataset.dataset_name,
                                                        "sparsity-gpt-35-turbo-1106-with-headers")
    sparsity_with_headers_results = cattrs.structure(
        load_json(sparsity_with_headers_results_dir / "column_task_results.json"), ColumnTaskResults)
    wf1_by_sparsity_with_headers_x = []
    wf1_by_sparsity_with_headers_y = []
    for sparsity, report in sparsity_with_headers_results.classification_report_by_sparsity.items():
        wf1_by_sparsity_with_headers_x.append(sparsity)
        wf1_by_sparsity_with_headers_y.append(report["weighted avg"]["f1-score"])
    wf1_by_sparsity_with_headers_x, wf1_by_sparsity_with_headers_y = zip(
        *sorted(zip(wf1_by_sparsity_with_headers_x, wf1_by_sparsity_with_headers_y)))

    sparsity_without_headers_results_dir = get_results_dir(cfg.task_name, cfg.dataset.dataset_name,
                                                           "sparsity-gpt-35-turbo-1106-without-headers")
    sparsity_without_headers_results = cattrs.structure(
        load_json(sparsity_without_headers_results_dir / "column_task_results.json"), ColumnTaskResults)
    wf1_by_sparsity_without_headers_x = []
    wf1_by_sparsity_without_headers_y = []
    for sparsity, report in sparsity_without_headers_results.classification_report_by_sparsity.items():
        wf1_by_sparsity_without_headers_x.append(sparsity)
        wf1_by_sparsity_without_headers_y.append(report["weighted avg"]["f1-score"])
    wf1_by_sparsity_without_headers_x, wf1_by_sparsity_without_headers_y = zip(
        *sorted(zip(wf1_by_sparsity_without_headers_x, wf1_by_sparsity_without_headers_y)))

    dump_json({"x": wf1_by_num_columns_with_headers_x, "y": wf1_by_num_columns_with_headers_y},
              task_dir / "wf1_by_num_columns_with_headers.json")
    dump_json({"x": wf1_by_num_columns_without_headers_x, "y": wf1_by_num_columns_without_headers_y},
              task_dir / "wf1_by_num_columns_without_headers.json")
    dump_json({"x": wf1_by_sparsity_with_headers_x, "y": wf1_by_sparsity_with_headers_y},
              task_dir / "wf1_by_sparsity_with_headers.json")
    dump_json({"x": wf1_by_sparsity_without_headers_x, "y": wf1_by_sparsity_without_headers_y},
              task_dir / "wf1_by_sparsity_without_headers.json")

    ################################################################################################################################
    # plot
    ################################################################################################################################
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (4, 2.4)
    plt.rcParams["font.size"] = 8
    figure, axis = plt.subplots(2, 2, sharex=False, sharey=False)
    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.7)

    axis[0, 0].plot(wf1_by_num_columns_without_headers_x, wf1_by_num_columns_without_headers_y, mew=2,
                    color=GRADIENT_1A_LIGHT[1], marker="*", mfc=COLOR_1A, mec=COLOR_1A, label="without headers")
    axis[0, 0].plot(wf1_by_num_columns_with_headers_x, wf1_by_num_columns_with_headers_y, mew=2,
                    color=GRADIENT_9A_LIGHT[1], marker="x", mfc=COLOR_9A, mec=COLOR_9A, label="with headers")
    axis[0, 0].set_xlim((0, 100))
    axis[0, 0].set_xticks((0, 25, 50, 75, 100), labels=("0", "", "50", "", "100"))
    axis[0, 0].set_ylim((0, 0.5))
    axis[0, 0].set_yticks((0, 0.125, 0.25, 0.375, 0.50), labels=("0", "", "", "", "0.5"))
    axis[0, 0].set_ylabel("F1 score")
    axis[0, 0].set_xlabel(" ")

    for label in axis[0, 0].get_yticklabels():
        if label.get_text() == "0.5":
            label.set_fontweight("bold")

    axis[0, 1].plot(wf1_by_num_columns_without_headers_x, wf1_by_num_columns_without_headers_y, mew=2,
                    color=GRADIENT_1A_LIGHT[1], marker="*", mfc=COLOR_1A, mec=COLOR_1A, label="without headers")
    axis[0, 1].set_xlim((0, 100))
    axis[0, 1].set_xticks((0, 25, 50, 75, 100), labels=("0", "", "50", "", "100"))
    axis[0, 1].set_ylim((0, 0.05))
    axis[0, 1].set_yticks((0, 0.0125, 0.025, 0.0375, 0.05), labels=("0", "", "", "", "0.05"))
    axis[0, 1].set_ylabel(" ")
    axis[0, 1].set_xlabel(" ")

    for label in axis[0, 1].get_yticklabels():
        if label.get_text() == "0.05":
            label.set_fontweight("bold")
            label.set_fontstyle("italic")

    figure.text(0.535, 0.48, "number of columns", ha="center")

    axis[1, 0].plot(wf1_by_sparsity_without_headers_x, wf1_by_sparsity_without_headers_y, mew=2,
                    color=GRADIENT_1A_LIGHT[1], marker="*", mfc=COLOR_1A, mec=COLOR_1A, label="without headers")
    axis[1, 0].plot(wf1_by_sparsity_with_headers_x, wf1_by_sparsity_with_headers_y, mew=2, color=GRADIENT_9A_LIGHT[1],
                    marker="x", mfc=COLOR_9A, mec=COLOR_9A, label="with headers")
    axis[1, 0].set_xlim((0, 1))
    axis[1, 0].set_xticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1"))
    axis[1, 0].set_ylim((0, 0.5))
    axis[1, 0].set_yticks((0, 0.125, 0.25, 0.375, 0.50), labels=("0", "", "", "", "0.5"))
    axis[1, 0].set_ylabel("F1 score")
    axis[1, 0].set_xlabel(" ")

    for label in axis[1, 0].get_yticklabels():
        if label.get_text() == "0.5":
            label.set_fontweight("bold")

    axis[1, 1].plot(wf1_by_sparsity_without_headers_x, wf1_by_sparsity_without_headers_y, mew=2,
                    color=GRADIENT_1A_LIGHT[1], marker="*", mfc=COLOR_1A, mec=COLOR_1A, label="without headers")
    axis[1, 1].set_xlim((0, 1))
    axis[1, 1].set_xticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1"))
    axis[1, 1].set_ylim((0, 0.05))
    axis[1, 1].set_yticks((0, 0.0125, 0.025, 0.0375, 0.05), labels=("0", "", "", "", "0.05"))
    axis[1, 1].set_ylabel(" ")
    axis[1, 1].set_xlabel(" ")

    for label in axis[1, 1].get_yticklabels():
        if label.get_text() == "0.05":
            label.set_fontweight("bold")
            label.set_fontstyle("italic")

    figure.text(0.535, 0.00, "fraction of empty cells", ha="center")

    plt.savefig(task_dir / "weighted_f1_score_by.pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    gather_plots()
