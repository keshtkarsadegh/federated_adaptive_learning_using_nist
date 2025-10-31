import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class

"""
Feature Aligned Grid Search (parallel-only).

This script runs a grid search over beta (λ) values and aggregation 
methods for Aligned feature regularization training.  
It is intentionally parallel-only: 
    - Outer level: multiple grid tasks in parallel via ThreadPoolExecutor
    - Inner level: each task spawns multiple runs via ProcessPoolExecutor

Workflow:
    1. Define parameter grid (beta × aggregation methods)
    2. For each configuration, run a federated simulation with a trainer
    3. Collect and merge accuracy results
    4. Save per-task JSON outputs under results/…/aligned_feature_grid_search/

Entrypoint: run with `python aligned_feature_grid_search.py`
"""

# ---------- utils ----------
def _to_list(v: Any):
    """
    Recursively convert numpy arrays (and nested lists/tuples/dicts) to plain Python lists.

    Args:
        v (Any): Arbitrary nested structure.

    Returns:
        Any: Structure with all numpy arrays converted to lists.
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
    Limit number of threads in child processes.

    Sets environment variables and Torch num_threads to 1 to avoid
    oversubscription when using process pools.
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
    Generate parameter grid (beta × aggregation method).

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        agg_method_name (str | None): Specific method name; if None,
            use all methods defined on the selected aggregation class.

    Returns:
        list[tuple[float, str]]: List of (beta, aggregation_method_name) configs.
    """

    beta_values = [ 1e-3,1e-2, 0.1,.5,.9, 1.0]

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
    return [(lam, name) for lam in beta_values for name in agg_names]


# ---------- worker (inner unit) ----------
def run_grid(scenario: str, metadata: str, cfg: tuple[float, str]):
    """
    Run a single grid search configuration (one beta + one agg method).

    Sets up trainer and runner for the given scenario, executes simulation,
    and returns accuracy metrics.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        cfg (tuple[float, str]): (beta, agg_method_name).

    Returns:
        tuple:
            - str: status message
            - dict: mapping from exp_name -> accuracies
            - str: result path
    """

    _limit_cpu_threads_for_child_process()

    from src.federated_adaptive_learning_nist.trainers.cf_aligned_feature_trainer import CFAlignedFeatureTrainer
    from src.federated_adaptive_learning_nist.runners.base_concurrent_runner import BaseConcurrentRunner
    from src.federated_adaptive_learning_nist.runners.base_sequential_runner import BaseSequentialRunner
    from src.federated_adaptive_learning_nist.nist_logger import NistLogger
    from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class

    beta, agg_method_name = cfg
    agg_cls = select_class(scenario, metadata)
    agg_method = getattr(agg_cls, agg_method_name)

    trainer = CFAlignedFeatureTrainer(beta=beta)
    runner = BaseSequentialRunner(trainer=trainer) if scenario == "sequential" else BaseConcurrentRunner(trainer=trainer)

    exp_name = f"aligned_feature_{beta}_agg_{agg_method_name}_{metadata}_{scenario}"
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
):
    """
    Run grid search for a given scenario/metadata pair.

    Executes all configurations in parallel via ProcessPoolExecutor.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        index (int): Index used in output file name.
        agg_method_name (str | None): Specific aggregation method or all if None.
        inner_max_workers (int): Processes per task.

    Returns:
        dict: Merged accuracies across all configs.
    """

    param_grid = get_param_grid(scenario, metadata, agg_method_name)
    if not param_grid:
        raise ValueError("Empty param grid.")

    mp_ctx = get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=inner_max_workers, mp_context=mp_ctx) as ex:
        futs = [ex.submit(run_grid, scenario, metadata, cfg) for cfg in param_grid]
        for f in as_completed(futs):
            results.append(f.result())

    msgs, acc_dicts, result_paths = zip(*results)
    merged = {}
    for d in acc_dicts:
        merged.update(d)

    parent_dir = Path(result_paths[0]) / "aligned_feature_grid_search"
    out_dir = parent_dir / f"{scenario}_{metadata}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"accuracies_points_{index}.json"
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2)

    print("\n".join(msgs))
    print(f"📝 Saved: {out_json}")
    return merged


# ---------- outer driver (parallel only) ----------
def run_all_parallel(outer_max_workers: int = 3, inner_max_workers: int = 8):
    """
    Run all predefined grid search tasks in parallel.

    Outer level uses ThreadPoolExecutor to execute multiple grid_search tasks.

    Args:
        outer_max_workers (int): How many tasks to run at once.
        inner_max_workers (int): Processes per grid_search task.

    Returns:
        list[dict]: Results from all tasks.
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
            outer.submit(grid_search, sc, md, idx, name, inner_max_workers)
            for (sc, md, idx, name) in tasks
        ]
        for f in as_completed(futs):
            results.append(f.result())
    return results


if __name__ == "__main__":
    OUTER_MAX = 4   # how many tasks at once
    INNER_MAX = 4   # processes per task
    for _ in run_all_parallel(outer_max_workers=OUTER_MAX, inner_max_workers=INNER_MAX):
        pass
