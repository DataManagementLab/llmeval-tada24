# LLMs for Data Engineering on Enterprise Data

**A recent line of work applies Large Language Models (LLMs) to data engineering tasks on tabular data, suggesting
they can solve a broad spectrum of tasks with high accuracy. However, existing research primarily uses datasets
based on tables from web sources such as Wikipedia, calling the applicability of LLMs for real-world enterprise data
into question. In this paper, we perform a first analysis of LLMs for solving data engineering tasks on a real-world
enterprise dataset. As an exemplary task, we apply recent LLMs to the task of column type annotation to study how
the data characteristics affect the LLMs' accuracy and find that LLMs have severe limitations when dealing with
enterprise data. Based on these findings, we point towards promising directions for adapting LLMs to the enterprise
context.**

Please check out our [paper](https://vldb.org/workshops/2024/proceedings/TaDA/TaDA.4.pdf) and cite our work:

```bibtex
@inproceedings{DBLP:conf/vldb/BodensohnBVUSB24,
  author       = {Jan{-}Micha Bodensohn and
                  Ulf Brackmann and
                  Liane Vogel and
                  Matthias Urban and
                  Anupam Sanghi and
                  Carsten Binnig},
  title        = {LLMs for Data Engineering on Enterprise Data},
  booktitle    = {Proceedings of Workshops at the 50th International Conference on Very
                  Large Data Bases, {VLDB} 2024, Guangzhou, China, August 26-30, 2024},
  publisher    = {VLDB.org},
  year         = {2024},
  url          = {https://vldb.org/workshops/2024/proceedings/TaDA/TaDA.4.pdf}
}
```

## Setup

Make sure you have [Conda](https://docs.conda.io/projects/miniconda/en/latest/) installed.

Create a new Conda environment, activate it, and add the project to the Python path:

```bash
conda env create -f environment.yml
conda activate llmeval-tada24
export PYTHONPATH=${PYTHONPATH}:./
```

## Reproducibility

We provide all code to reproduce the experiments on the public datasets and a subset of the code to reproduce the
experiments on the enterprise data.

Reproducing the exact results from the paper requires the following artifacts:

* `openai_cache.zip` the OpenAI API requests and responses for the public datasets, which you must unpack
  into `data/openai_cache`
* `sportstables_download.zip` the crawled version of the SportsTables dataset, which you must unpack
  into `data/column_type_inference/sportstables/download`
* `gittablesCTA_download.zip` the GitTables CTA benchmark dataset augmented with column names from the original
  GitTables dataset, which you must unpack into `data/column_type_inference/gittablesCTA/download`
* `sotab_download.zip` the SOTAB dataset, which you must unpack into `data/column_type_inference/sotab/download`
* `wikitables-turl_download.zip` the WikiTables-TURL dataset, which you must unpack
  into `data/column_type_inference/wikitables-turl/download`
* `sapdata.zip` the results from the experiments on enterprise data, which you must unpack
  into `data/column_type_inference/sapdata`

To reproduce the results from the paper, run:

```bash
bash reproduce.sh
```

The results are:

* `data/analyze_datasets/<dataset-name>.json` Table 1 (data characteristics)
* `data/column_type_inference/main_results.csv` Table 2 (enterprise vs. web tables)
* `data/column_type_inference/data_type_results.csv` Table 3 (non-numeric vs. numeric data)
* `data/column_type_inference/weighted_f1_score_by.pdf` Figure 2 (varying numbers of columns and sparsities)
