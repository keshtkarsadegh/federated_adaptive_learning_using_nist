# ewc_grid_search.py  — PARALLEL-ONLY
import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class

"""
EWC Grid Search (parallel-only).

This script runs a grid search over different EWC λ values 
(elasticity coefficients) and aggregation methods for federated 
adaptive learning on NIST.

Execution design:
    - Outer level: multiple grid_search tasks in parallel via ThreadPoolExecutor
    - Inner level: each grid_search executes parameter configs in parallel 
      via ProcessPoolExecutor

Workflow:
    1. Build parameter grid (ewc_lambda × aggregation methods)
    2. For each config, initialize EWCTrainer with given λ
    3. Run federated simulation with sequential or concurrent runner
    4. Collect accuracy results
    5. Save merged outputs to JSON under results/.../ewc_grid_search/

Entrypoint: run with `python ewc_grid_search.py`
"""

# ---------- utils ----------
def _to_list(v: Any):
    """
    Recursively convert numpy arrays into plain Python lists.

    Args:
        v (Any): Arbitrary nested structure.

    Returns:
        Any: Same structure with numpy arrays replaced by Python lists.
    """

    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_to_list(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_list(val) for k, val in v.items()}
    return v

def _limit_cpu_threads_for_child_process():
    """
    Restrict thread usage inside child processes.

    Sets environment variables (OMP, MKL, OPENBLAS, NUMEXPR) and
    torch.set_num_threads(1) to prevent oversubscription.
    """

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


# ---------- grid ----------
def get_param_grid(scenario: str, metadata: str, agg_method_name: str | None = None):
    """
    Build parameter grid for EWC λ × aggregation methods.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        agg_method_name (str | None): Specific aggregation method to run, or all.

    Returns:
        list[tuple[float, str]]: Grid of (ewc_lambda, aggregation_method_name).
    """

    ewc_lambda_values = [0.2, 1, 4, 8, 10, 20, 100, 200]
    cls = select_class(scenario, metadata)
    if agg_method_name is None:
        names, _ = cls.list_methods()
        agg_names = list(names)
        if not agg_names:
            raise ValueError(f"No aggregation methods found on {cls.__name__}.")
    else:
        if not hasattr(cls, agg_method_name):
            raise ValueError(f"Aggregation method '{agg_method_name}' not found on {cls.__name__}.")
        agg_names = [agg_method_name]
    return [(lam, name) for lam in ewc_lambda_values for name in agg_names]


# ---------- worker (inner unit) ----------
def run_ewc_grid(scenario: str, metadata: str, cfg: tuple[float, str],single_outlier=None):
    """
    Run a single EWC grid search configuration.

    Initializes EWCTrainer with the given λ, runs the federated simulation
    (sequential or concurrent), and collects accuracy results.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        cfg (tuple[float, str]): (ewc_lambda, aggregation method name).
        single_outlier: either None or list of selected outliers' ids


    Returns:
        tuple:
            - str: status message
            - dict: {experiment_name: accuracies}
            - str: result path
    """

    _limit_cpu_threads_for_child_process()

    from src.federated_adaptive_learning_nist.trainers.ewc_trainer import EWCTrainer
    from src.federated_adaptive_learning_nist.runners.base_concurrent_runner import BaseConcurrentRunner
    from src.federated_adaptive_learning_nist.runners.base_sequential_runner import BaseSequentialRunner
    from src.federated_adaptive_learning_nist.nist_logger import NistLogger
    from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class

    ewc_lambda, agg_method_name = cfg
    agg_cls = select_class(scenario, metadata)
    agg_method = getattr(agg_cls, agg_method_name)

    trainer = EWCTrainer(ewc_lambda=ewc_lambda)
    runner = BaseSequentialRunner(trainer=trainer,single_outlier=single_outlier) if scenario == "sequential" else BaseConcurrentRunner(trainer=trainer,single_outlier=single_outlier)

    exp_name = f"ewc_{ewc_lambda}_agg_{agg_method_name}_{metadata}_{scenario}"
    NistLogger.debug(f"[Parallel] {exp_name}")

    accs, result_path_str = runner.simulate(
        exp_name=exp_name,
        global_name="global",
        aggregate_method=agg_method,
        batch_size=512,
        epochs=50,
        max_round=50,
        grid_Search=True,
    )
    return (f"✅ Finished: {exp_name}", {exp_name: _to_list(accs)}, result_path_str)


# ---------- inner driver (per task) ----------
def grid_search(
    scenario: str,
    metadata: str,
    index: int,
    agg_method_name: str | None = None,
    inner_max_workers: int = 8,
        single_outlier=None,
):
    """
    Run grid search for one (scenario, metadata) pair.

    Executes all λ configs in parallel with ProcessPoolExecutor, merges results,
    and writes accuracies JSON under ewc_grid_search/.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        index (int): Identifier for output JSON filename.
        agg_method_name (str | None): Specific method to run, or all.
        inner_max_workers (int): Processes per task.
        single_outlier: either None or list of selected outliers' ids


    Returns:
        dict: Merged accuracy results across configs.
    """

    param_grid = get_param_grid(scenario, metadata, agg_method_name)
    if not param_grid:
        raise ValueError("Empty param grid.")

    mp_ctx = get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=inner_max_workers, mp_context=mp_ctx) as ex:
        futs = [ex.submit(run_ewc_grid, scenario, metadata, cfg,single_outlier) for cfg in param_grid]
        for f in as_completed(futs):
            results.append(f.result())

    msgs, acc_dicts, result_paths = zip(*results)
    merged = {}
    for d in acc_dicts:
        merged.update(d)

    parent_dir = Path(result_paths[0]) / "ewc_grid_search"
    out_dir = parent_dir / f"{scenario}_{metadata}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"accuracies_points_{index}.json"
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2)

    print("\n".join(msgs))
    print(f"📝 Saved: {out_json}")
    return merged


# ---------- outer driver (parallel only) ----------
def run_all_parallel(outer_max_workers: int = 3, inner_max_workers: int = 8,single_outlier=None):
    """
    Run all predefined EWC grid search tasks in parallel.

    Outer level distributes multiple grid_search tasks with ThreadPoolExecutor.

    Args:
        outer_max_workers (int): Number of tasks to run concurrently.
        inner_max_workers (int): Processes per grid_search.
        single_outlier: either None or list of selected outliers' ids


    Returns:
        list[dict]: Results for all tasks.
    """

    tasks = [
        ("concurrent", "delta",   100, "agg_weighted_cgd"),
        ("concurrent", "weights", 100, "agg_weighted_cgw"),
        ("sequential", "delta",   100, "sequential_agg_delta_fedavg_update"),
        ("sequential", "weights", 100, "sequential_agg_weights_fedavg_update"),
    ]

    results = []
    with ThreadPoolExecutor(max_workers=outer_max_workers) as outer:
        futs = [
            outer.submit(grid_search, sc, md, idx, name, inner_max_workers,single_outlier)
            for (sc, md, idx, name) in tasks
        ]
        for f in as_completed(futs):
            results.append(f.result())
    return results

def ewc_grid_search():
    OUTER_MAX = 2  # how many tasks at once
    INNER_MAX = 8  # processes per task
    for _ in run_all_parallel(outer_max_workers=OUTER_MAX, inner_max_workers=INNER_MAX,single_outlier=None):
        pass
if __name__ == "__main__":
    OUTER_MAX = 2   # how many tasks at once
    INNER_MAX = 8   # processes per task
    for _ in run_all_parallel(outer_max_workers=OUTER_MAX, inner_max_workers=INNER_MAX):
        pass
