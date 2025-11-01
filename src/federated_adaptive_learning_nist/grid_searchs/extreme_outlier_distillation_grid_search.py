import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class

"""
distillation_grid_search_parallel.py — PARALLEL-ONLY

Purpose:
    Grid search for knowledge distillation hyperparameters in federated adaptive learning.

Sweeps across:
    - T (softmax temperature): [1, 2, 4, 8, 10, 50]
    - α (distillation weight): [0.1, 0.2, 0.5, 0.7, 0.95, 0.98]
    - Aggregation methods (per scenario/metadata)

Scenarios:
    - Concurrent vs Sequential
    - Delta vs Weights metadata

Parallelism:
    - Outer level: threads → each (scenario, metadata, method)
    - Inner level: processes → each (T, α) config

Outputs:
    - Accuracies JSON: results/dual_outlier_distillation_grid_search/<scenario>_<metadata>/accuracies_points_XXX.json
    - Console log: experiment completions and file paths

Usage:
    $ python distillation_grid_search_parallel.py
    (adjust OUTER_MAX / INNER_MAX at bottom to control concurrency)
"""



# ---------- utils ----------
def _to_list(v: Any):
    """
    Recursively convert numpy arrays (and nested structures) into JSON-safe Python types.

    Args:
        v (Any): Arbitrary value or nested structure (lists/tuples/dicts/np.ndarrays).

    Returns:
        Any: Same structure with all numpy arrays converted to Python lists.
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
    Limit thread usage inside child processes to avoid CPU oversubscription.

    Sets common BLAS/NumExpr env vars (OMP, MKL, OPENBLAS, NUMEXPR) to 1 and, if
    available, calls torch.set_num_threads(1). Safe to call multiple times.
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
Build the Cartesian grid of distillation hyperparameters and aggregation methods.

Constructs (T, alpha, agg_method_name) tuples for the requested scenario/metadata.
Method names are returned (not callables) to keep items picklable for multiprocessing.

Args:
    scenario (str): "sequential" or "concurrent".
    metadata (str): "weights" or "delta" (or any type supported by select_class).
    agg_method_name (str | None): If provided, restricts grid to this single method;
        otherwise, all methods from the resolved aggregation class are used.

Returns:
    list[tuple[float, float, str]]: List of (T, alpha, agg_method_name) configurations.

"""

    T_values = [1, 2, 4, 8, 10, 50]
    alpha_values = [0.1, .2, 0.5, 0.7, 0.95, 0.98]

    cls = select_class(scenario, metadata)

    if agg_method_name is None:
        names, _ = cls.list_methods()
        agg_method_names = list(names)
        if not agg_method_names:
            raise ValueError(f"No aggregation methods found on {cls.__name__}.")
    else:
        if not hasattr(cls, agg_method_name):
            raise ValueError(f"Aggregation method '{agg_method_name}' not found on {cls.__name__}.")
        agg_method_names = [agg_method_name]

    return [(T, a, name) for T in T_values for a in alpha_values for name in agg_method_names]


# ---------- worker (inner unit) ----------
def run_distillation_grid(scenario: str, metadata: str, cfg: tuple[float, float, str],single_outlier):
    """
    Execute a single distillation configuration and collect results.

    Initializes DistillationTrainer(T, alpha), resolves the aggregation method for the
    (scenario, metadata) pair, selects the appropriate runner (sequential/concurrent),
    and runs a simulation. Designed to be invoked inside a process pool worker.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        cfg (tuple[float, float, str]): (T, alpha, agg_method_name).

    Returns:
        tuple:
            msg (str): Human-readable completion message.
            results (dict): {experiment_name: accuracies_or_metrics}.
            result_path_str (str): Filesystem path where the run stored outputs.
    """

    _limit_cpu_threads_for_child_process()

    # Per-process imports (keep master light)
    from src.federated_adaptive_learning_nist.trainers.distillation_trainer import DistillationTrainer
    from src.federated_adaptive_learning_nist.runners.base_concurrent_runner import BaseConcurrentRunner
    from src.federated_adaptive_learning_nist.runners.base_sequential_runner import BaseSequentialRunner
    from src.federated_adaptive_learning_nist.nist_logger import NistLogger
    from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class

    T, alpha, agg_method_name = cfg

    # Resolve aggregation method locally
    agg_cls = select_class(scenario, metadata)
    agg_method = getattr(agg_cls, agg_method_name)

    trainer = DistillationTrainer(T=T, alpha=alpha)
    runner = BaseSequentialRunner(trainer=trainer,single_outlier=single_outlier) if scenario == "sequential" else BaseConcurrentRunner(trainer=trainer,single_outlier=single_outlier)

    exp_name = f"distill_T{T}_a{alpha}_agg_{agg_method_name}_{metadata}_{scenario}"
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
        single_oulier=None,
extreme_case="single"
):
    """
    Run parallel grid search over (T, alpha, agg_method) for one (scenario, metadata) pair.

    Spawns a ProcessPoolExecutor with 'spawn' context, submits all configurations from
    get_param_grid(...), aggregates the returned accuracy dicts, and writes a merged JSON
    file to results/dual_outlier_distillation_grid_search/<scenario>_<metadata>/.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        index (int): Identifier appended to the output JSON filename.
        agg_method_name (str | None): Optional specific aggregation method to test.
        inner_max_workers (int): Number of worker processes for this grid.

    Returns:
        dict: Merged mapping {experiment_name: accuracies_or_metrics} across all configs.

    """

    param_grid = get_param_grid(scenario, metadata, agg_method_name)
    if not param_grid:
        raise ValueError("Empty param grid.")

    mp_ctx = get_context("spawn")  # safe when outer is threads
    results = []
    with ProcessPoolExecutor(max_workers=inner_max_workers, mp_context=mp_ctx) as ex:
        futs = [ex.submit(run_distillation_grid, scenario, metadata, cfg,single_oulier) for cfg in param_grid]
        for f in as_completed(futs):
            results.append(f.result())

    msgs, acc_dicts, result_paths = zip(*results)

    merged = {}
    for d in acc_dicts:
        merged.update(d)

    parent_dir = Path(result_paths[0]) / f"{extreme_case}_outlier_distillation_grid_search"
    out_dir = parent_dir / f"{scenario}_{metadata}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"accuracies_points_{index}.json"
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2)

    print("\n".join(msgs))
    print(f"📝 Saved: {out_json}")
    return merged


# ---------- outer driver (parallel only) ----------
def run_all_parallel(outer_max_workers: int = 3, inner_max_workers: int = 8,single_outlier=None, extreme_case="single"):
    """
    Top-level launcher: schedule multiple grid_search tasks concurrently (threads).

    Each thread runs grid_search(...) for a particular (scenario, metadata, method) task,
    which in turn spins up a process pool to evaluate (T, alpha) configs.

    Args:
        outer_max_workers (int): Number of grid_search tasks (threads) to run in parallel.
        inner_max_workers (int): Number of worker processes each grid_search may use.

    Returns:
        list[dict]: One merged results dict per scheduled task, in completion order.
    """

    tasks = [
        ("concurrent", "delta",   100, "con_delta_capped_cgd"),
        ("concurrent", "weights", 100, "con_capped_cgw"),
        ("sequential", "delta",   100, "seq_delta_fedavg_update"),
        ("sequential", "weights", 100, "seq_fedavg_update"),
    ]
    results = []
    with ThreadPoolExecutor(max_workers=outer_max_workers) as outer:
        futs = [
            outer.submit(grid_search, sc, md, idx, name, inner_max_workers,single_outlier,extreme_case)
            for (sc, md, idx, name) in tasks
        ]
        for f in as_completed(futs):
            results.append(f.result())
    return results

def extreme_cases_grid_search(single_outlier, extreme_case):
    OUTER_MAX = 2  # how many tasks at once
    INNER_MAX = 8  # processes per task
    for _ in run_all_parallel(outer_max_workers=OUTER_MAX, inner_max_workers=INNER_MAX,single_outlier=single_outlier,extreme_case=extreme_case):
        pass
if __name__ == "__main__":
    OUTER_MAX = 4   # how many tasks at once
    INNER_MAX =5   # processes per task
    for _ in run_all_parallel(outer_max_workers=OUTER_MAX, inner_max_workers=INNER_MAX):
        pass


