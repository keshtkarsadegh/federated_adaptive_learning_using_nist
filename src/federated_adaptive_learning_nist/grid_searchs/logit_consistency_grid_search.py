import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class
"""
Logit Consistency Grid Search (parallel-only).

Runs a parallel grid search over λ_consis values and aggregation methods 
for logit-consistency–based aligned feature training in federated learning.

Execution model:
    - Outer layer: ThreadPoolExecutor distributes multiple grid_search tasks
    - Inner layer: Each grid_search runs multiple configs in parallel with 
      ProcessPoolExecutor

Pipeline:
    1. Build parameter grid (λ_consis × aggregation methods)
    2. For each config, initialize CFLogitConsistencyTrainer
    3. Run simulation with sequential or concurrent runner
    4. Collect accuracy results and merge
    5. Save results to JSON under results/.../logit_consistency_grid_search/

Entrypoint: run with `python logit_consistency_grid_search.py`
"""


# ---------- utils ----------
def _to_list(v: Any):
    """
    Convert nested numpy arrays into plain Python lists.

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
    Limit number of threads inside child processes.

    Sets environment variables (OMP, MKL, OPENBLAS, NUMEXPR) and
    torch.set_num_threads(1) to prevent CPU oversubscription.
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
    Generate parameter grid for λ_consis × aggregation methods.

    Args:
        scenario (str): Either "sequential" or "concurrent".
        metadata (str): Either "weights" or "delta".
        agg_method_name (str | None): Specific method to use, or all if None.

    Returns:
        list[tuple[float, str]]: List of (λ_consis, aggregation_method_name) configs.
    """

    lambda_consis_values = [ 1e-3,1e-2, 0.1,.5,.9, 1.0]

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
    return [(lam, name) for lam in lambda_consis_values for name in agg_names]


# ---------- worker (inner unit) ----------
def run_grid(scenario: str, metadata: str, cfg: tuple[float, str]):
    """
    Run a single logit-consistency grid configuration.

    Initializes CFLogitConsistencyTrainer with given λ_consis,
    selects aggregation method, runs the appropriate runner,
    and collects accuracy results.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        cfg (tuple[float, str]): (λ_consis, aggregation method name).

    Returns:
        tuple:
            - str: Status message
            - dict: {experiment_name: accuracies}
            - str: Path to results directory
    """

    _limit_cpu_threads_for_child_process()

    from src.federated_adaptive_learning_nist.trainers.cf_logit_consistency_trainer import CFLogitConsistencyTrainer
    from src.federated_adaptive_learning_nist.runners.base_concurrent_runner import BaseConcurrentRunner
    from src.federated_adaptive_learning_nist.runners.base_sequential_runner import BaseSequentialRunner
    from src.federated_adaptive_learning_nist.nist_logger import NistLogger
    from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class

    lambda_consis, agg_method_name = cfg
    agg_cls = select_class(scenario, metadata)
    agg_method = getattr(agg_cls, agg_method_name)

    trainer = CFLogitConsistencyTrainer(lambda_consis=lambda_consis)
    runner = BaseSequentialRunner(trainer=trainer) if scenario == "sequential" else BaseConcurrentRunner(trainer=trainer)

    exp_name = f"logit_consistency_{lambda_consis}_agg_{agg_method_name}_{metadata}_{scenario}"
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
    Run grid search for one (scenario, metadata) pair.

    Executes all parameter configs in parallel, aggregates accuracy results,
    and saves them to JSON under logit_consistency_grid_search/.

    Args:
        scenario (str): "sequential" or "concurrent".
        metadata (str): "weights" or "delta".
        index (int): Index for output JSON filename.
        agg_method_name (str | None): If provided, run only this method.
        inner_max_workers (int): Number of processes for this grid task.

    Returns:
        dict: Merged accuracy results across configurations.
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

    parent_dir = Path(result_paths[0]) / "logit_consistency_grid_search"
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
    Run all predefined logit-consistency grid search tasks in parallel.

    Outer layer distributes multiple grid_search tasks using ThreadPoolExecutor.

    Args:
        outer_max_workers (int): Number of tasks to run concurrently.
        inner_max_workers (int): Number of processes per grid_search.

    Returns:
        list[dict]: Accuracy results for all tasks.
    """

    tasks = [
        ("concurrent", "delta",   100, "con_weighted_cgd"),
        ("concurrent", "weights", 100, "con_weighted_cgw"),
        ("sequential", "delta",   100, "seq_delta_fedavg_update"),
        ("sequential", "weights", 100, "seq_fedavg_update"),
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
    INNER_MAX =4   # processes per task
    for _ in run_all_parallel(outer_max_workers=OUTER_MAX, inner_max_workers=INNER_MAX):
        pass
