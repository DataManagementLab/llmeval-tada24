# LLMs for Data Engineering on Enterprise Data

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
  into `data/openai`
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
