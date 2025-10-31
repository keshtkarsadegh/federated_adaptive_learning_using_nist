# Federated Adaptive Learning on NIST Writers

This repository implements a **modular framework for federated adaptive learning** on the [NIST Special Database of Handwritten Characters](https://www.nist.gov/srd/nist-special-database-19).  

The project tackles the problem of **adapting strong global models to new or outlier clients without destroying global knowledge**. It systematically compares:

- **Communication scheduling:** concurrent vs sequential  
- **Aggregation forms:** full weights vs model deltas, client-only vs client+global  
- **Regularization:** FedProx, Elastic Weight Consolidation (EWC), knowledge distillation, selective layer freezing, logit consistency  
- **Loss designs:** cross-entropy vs knowledge distillation  

A strong global baseline (≈ 99.6% test accuracy) is used, and experiments quantify trade-offs between **stability** (knowledge preservation) and **plasticity** (client adaptation).

---

## Repository Structure


```
federated_adaptive_learning_nist/
├── logs/
├── src/
│   └── federated_adaptive_learning_nist/
│       ├── aggregation_methods/
│       ├── global_trainings/
│       ├── grid_searchs/
│       ├── outliers/
│       ├── results/
│       ├── runners/
│       ├── trainers/
│       ├── trainings/
│       ├── utils/
│       ├── __init__.py
│       ├── constants.py
│       ├── data_utils.py
│       ├── label_generation.py
│       ├── load_nist_data_by_writer.py
│       ├── model.py
│       ├── nist_downloader_extractor.py
│       ├── nist_logger.py
│       ├── plots.py
│       ├── split_hash_by_digits.py
│       └── training_utils.py
├── .gitignore
├── check.sh
├── LICENSE
├── poetry.lock
├── pyproject.toml
└── README.md
```
---

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.  
Make sure Poetry is installed (`pip install poetry` or follow their docs), then run:

```bash
git clone https://github.com/<your-username>/federated_adaptive_learning_using_nist.git
cd federated_adaptive_learning_using_nist

# Install dependencies
poetry install

# Activate the environment
poetry shell
````

All commands in the following sections should be run inside the Poetry shell, or you can prefix them with `poetry run`, e.g.:

```bash
poetry run python src/federated_adaptive_learning_nist/trainings/global_training.py
```

---

## Dataset Setup

1. **Download NIST dataset**
   (recommended to download manually, but script provided):

   ```bash
   poetry run python src/federated_adaptive_learning_nist/nist_downloader_extractor.py
   ```

2. **Generate labels**:

   ```bash
   poetry run python src/federated_adaptive_learning_nist/label_generation.py
   ```

3. **Data utilities** (`data_utils.py`) handle:

   * splitting clients/global sets
   * constructing global datasets
   * writer-based partitions

---

## Usage

### 1. Global Training

Train and evaluate the baseline global model:

```bash
poetry run python src/federated_adaptive_learning_nist/trainings/global_training.py
```

Train global+clients combined model:

```bash
poetry run python src/federated_adaptive_learning_nist/trainings/global_clients_training.py
```

Run baseline FedAvg experiments:

```bash
poetry run python src/federated_adaptive_learning_nist/trainings/base_trainings_fl.py
```

Run all aggregation variants:

```bash
poetry run python src/federated_adaptive_learning_nist/trainings/training_all_aggs.py
```
To generate heatmap of all aggregation variants, just run all_`agg_heatmap.py` from `utils` in the result folder of this experiment.


### 2. Regularization Experiments

Each regularization method follows this workflow:

1. **Grid search**

   ```bash
   poetry run python src/federated_adaptive_learning_nist/grid_searchs/<method>_grid_search.py
   ```

   (define hyperparameter ranges inside the script)

2. **Select top configs**

   ```bash
   poetry run python src/federated_adaptive_learning_nist/utils/top_6_selector.py --results <path>
   ```

3. **Plot heatmaps**

   ```bash
   poetry run python src/federated_adaptive_learning_nist/utils/plot_grid_heatmaps.py --results <path> --name <method>
   ```

4. **Final training**
   Run the corresponding trainer with chosen hyperparameters, e.g.:

   ```bash
   poetry run python src/federated_adaptive_learning_nist/trainers/EWCTrainer.py
   ```

5. **Parallel experiment launcher**

   ```bash
   poetry run python src/federated_adaptive_learning_nist/trainings/generate_experiments.py \
       --trainer KDTrainer \
       --outer_max_workers 4 \
       --inner_max_workers 20 \
       --parent_name final_result
   ```

---

### 3. Plotting & Analysis

* Overlay plots (compare aggregation variants):

  ```bash
  poetry run python src/federated_adaptive_learning_nist/utils/overlay_plots.py
  ```

* Scenario plots (per scheduling scenario):

  ```bash
  poetry run python src/federated_adaptive_learning_nist/utils/scenario_plots.py
  ```
* plot the best accuracies of KD, the best method(the values are copied from generated json file from training):
```bash
 poetry run python src/federated_adaptive_learning_nist/utils/heat_map_min_max.py
```
---



### 4. Extreme Cases

In addition to normal many-client federated setups, we evaluate **extreme scenarios** where client diversity is limited:

* **Single-client adaptation** – train/test with only one outlier client.
* **Two-client setups** – restrict to exactly two clients for stability/plasticity stress tests.
* **Dual repetition of a single client** – simulate pathological cases where one client is duplicated twice.

How to run:

1. **Modify `simulate()` in runners** (`BaseConcurrentRunner` or `BaseSequentialRunner`):
   Set the `single_outlier` parameter.

   ```python
   # dual repetition
   single_outlier = ["f3503_07", "f3503_07"]

   # single client
   single_outlier = ["f3503_07"]

   # two clients
   single_outlier = ["f3503_07", "f3504_09"]
   ```

   If left as `"None"`, all selected outliers are included.

2. **Launch grid search or training** using the `DistillationTrainer`, which is the best-performing regularization approach:

   ```bash
   poetry run python src/federated_adaptive_learning_nist/trainings/generate_experiments.py \
       --trainer DistillationTrainer \
       --outer_max_workers 1 \
       --inner_max_workers 20 \
       --parent_name extreme_case_experiment
   ```

3. **Analysis & plots** follow the same workflow as normal experiments.
   Results are stored separately under `results/extreme_case_experiment...`.




---

## Key Features

* **Unified framework** for federated adaptive learning experiments
* **Reproducible trainers** across scheduling, aggregation, and regularization methods
* **Grid search utilities** for hyperparameter exploration
* **JSON + plots** stored for every experiment (re-plot without retraining)
* **Support for extreme cases** (few-client adaptation)

---
## Citation

This repository accompanies the manuscript:

S. Keshtkar, J. Kunkel, *Federated Adaptive Learning with Knowledge Preservation: A Comprehensive Study of Training and Optimization Strategies*, submitted to IEEE Access, 2025.

---

## License

This project is released under the MIT License.
