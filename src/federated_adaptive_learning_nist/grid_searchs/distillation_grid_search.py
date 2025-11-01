# distillation_grid_search_parallel.py  — PARALLEL-ONLY
import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class
"""
Distillation Grid Search (parallel-only).

This script launches a parallel grid search over knowledge distillation 
parameters (temperature T and interpolation α) and aggregation methods.

Execution model:
    - Outer level: ThreadPoolExecutor runs multiple grid_search tasks
    - Inner level: Each grid_search explores (T, α, aggregation) configs 
      in parallel using ProcessPoolExecutor

Workflow:
    1. Build parameter grid (T × α × aggregation methods)
    2. For each config, initialize DistillationTrainer
    3. Run simulation with sequential or concurrent runner
    4. Collect accuracy results and merge
    5. Save merged JSON under results/.../distillation_grid_search/

Entrypoint: run with `python distillation_grid_search_parallel.py`
"""

# ---- Param ranges ----



# ---------- utils ----------
def _to_list(v: Any):
    """
    Recursively convert numpy arrays into JSON-safe Python lists.

    Args:
        v (Any): Arbitrary nested structure.

    Returns:
        Any: Same structure with arrays/lists converted.
    """

    """Make JSON-safe (numpy/tuples/dicts)."""
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
    Prevent CPU oversubscription in child processes.

    Sets environment variables (OMP, MKL, OPENBLAS, NUMEXPR) and
    torch.set_num_threads(1) where available.
    """

    """Avoid CPU oversubscription in each worker process."""
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
Build parameter grid (T, α, aggregation_method).

Args:
    scenario (str): "sequential" or "concurrent".
    metadata (str): "weights" or "delta".
    agg_method_name (str | None): Specific aggregation method, or all if None.

Returns:
    list[tuple[float, float, str]]: Configurations of (T, α, method).
"""
    T_values = [1, 2, 4, 8, 10, 50]
    alpha_values = [0.1, 0.5, 0.95]
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
def run_distillation_grid(scenario: str, metadata: str, cfg: tuple[float, float, str],single_outlier=None):
    """
Run one distillation configuration.

Initializes DistillationTrainer with (T, α), runs simulation with the
chosen aggregation method, and collects results.

Args:
    scenario (str): "sequential" or "concurrent".
    metadata (str): "weights" or "delta".
    cfg (tuple[float, float, str]): (T, α, method name).

Returns:
    tuple:
        - str: Status message
        - dict: {experiment_name: accuracies}
        - str: Result path
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
        single_outlier = None,
):
    """
    Run grid search for one (scenario, metadata) pair.

    Spawns a process pool to evaluate all (T, α, method) configs in parallel,
    merges results, and writes them to JSON.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        index (int): Index for naming output JSON.
        agg_method_name (str | None): Specific method to test, or all.
        inner_max_workers (int): Processes for this search.
        single_outlier: either None or list of selected outliers' ids
    Returns:
        dict: Merged accuracy results.
    """

    param_grid = get_param_grid(scenario, metadata, agg_method_name)
    if not param_grid:
        raise ValueError("Empty param grid.")

    mp_ctx = get_context("spawn")  # safe when outer is threads
    results = []
    with ProcessPoolExecutor(max_workers=inner_max_workers, mp_context=mp_ctx) as ex:
        futs = [ex.submit(run_distillation_grid, scenario, metadata, cfg,single_outlier) for cfg in param_grid]
        for f in as_completed(futs):
            results.append(f.result())

    msgs, acc_dicts, result_paths = zip(*results)

    merged = {}
    for d in acc_dicts:
        merged.update(d)

    parent_dir = Path(result_paths[0]) / "distillation_grid_search"
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
    Run predefined distillation grid search tasks in parallel.

    Outer layer runs tasks as threads, each spawning processes internally.

    Args:
        outer_max_workers (int): Tasks to run in parallel.
        inner_max_workers (int): Processes per grid_search.
        single_outlier: either None or list of selected outliers' ids

    Returns:
        list[dict]: Results for all tasks.
    """

    tasks = [
        ("concurrent", "delta",   100, "con_delta_capped_cgd"),
        ("concurrent", "weights", 100, "con_capped_cgw"),
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

def distillation_grid_search():
    OUTER_MAX = 1   # how many tasks at once
    INNER_MAX =20   # processes per task
    for _ in run_all_parallel(outer_max_workers=OUTER_MAX, inner_max_workers=INNER_MAX,single_outlier=None):
        pass


if __name__ == "__main__":
    OUTER_MAX = 1   # how many tasks at once
    INNER_MAX =20   # processes per task
    for _ in run_all_parallel(outer_max_workers=OUTER_MAX, inner_max_workers=INNER_MAX):
        pass
