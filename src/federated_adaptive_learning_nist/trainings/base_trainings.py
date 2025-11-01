# grid_search.py
import inspect
import json
import itertools
import os
from pathlib import Path
import argparse
from typing import Any
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

# --- shared utils ---
from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class
from src.federated_adaptive_learning_nist.trainings.training_utils import get_trainer_class
from src.federated_adaptive_learning_nist.utils.plots import (
    plot_accuracies_over_rounds,
    plot_overlay_accuracies_per_scenario,
)

"""
base_trainings.py — Federated Training Driver

Purpose:
    Provides the unified driver that wraps `grid_search` for different
    federated learning trainers and scenarios. This is the glue layer
    between low-level simulation logic (BaseConcurrentRunner,
    BaseSequentialRunner, aggregation methods) and high-level experiment
    scripts (e.g., training_all_aggs.py, base_trainings_fl.py).

Capabilities:
    • Accepts configuration for trainer, scenario, metadata, and aggregation.
    • Spawns inner process pools for method-level fan-out.
    • Used by multiple run_all scripts to benchmark aggregation strategies.

Usage Modes:
    - Called directly by scripts like `training_all_aggs.py`.
    - Thread-level outer parallelism with per-job process pools inside.
    - Can run single aggregation method or sweep all available methods.

Inputs:
    trainer_name     Name of the trainer class to instantiate
                     (e.g., BaseTrainer, DistillationTrainer, EWCTrainer).
    scenario         "sequential" or "concurrent" execution mode.
    metadata         "weights" or "delta", determines aggregation base.
    index            Index suffix for per-job JSON outputs.
    agg_method_name  Optional. Specific aggregation method to run;
                     use "none" to sweep all methods for the given class.
    parent_name      Experiment tag; defines the parent output directory.
    inner_max_workers
                     Size of per-job process pool; controls parallelism.

Outputs:
    results/<parent_name>_<trainer>_grid_search/<scenario>_<metadata>/
        ├── accuracies_<index>.json        (per-job results)
        ├── summary_<index>.json           (all methods summary)
        ├── accuracies_over_rounds_*.png   (per-method plots)
        └── overlay_accuracies.png         (overlay comparison plot)

"""


# ----------------- helpers -----------------
def _to_list(v: Any):
    """Make JSON-safe (handles numpy)"""
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_to_list(x) for x in v]
    return v

def _normalize_accuracies(accuracies_any) -> list[tuple[float, float]]:
    """
    Accepts:
      - numpy array shape [R, 2]
      - list of [a, b] / (a, b)
    Returns: list[(clients_acc, global_acc), ...] as floats
    """
    acc = _to_list(accuracies_any)
    return [(float(a), float(b)) for (a, b) in acc]

# --------------- param grid ----------------
def get_param_grid(scenario: str, metadata: str, agg_method_name: str | None = None):
    """
    Return a list of aggregation method NAMES to run.
    Only pass names to keep items picklable for multiprocessing.
    """
    cls = select_class(scenario, metadata)

    if agg_method_name is None:
        # Expect list_methods() -> (names, callables)
        names, _ = cls.list_methods()
        names = list(names)
        if not names:
            raise ValueError(f"No aggregation methods found on {cls.__name__}.")
        return names
    else:
        if not hasattr(cls, agg_method_name):
            raise ValueError(f"Aggregation method '{agg_method_name}' not found on {cls.__name__}.")
        return [agg_method_name]

# --------------- worker --------------------
def _limit_cpu_threads_for_child_process():
    """
    HARD guard against CPU over-subscription when many processes launch BLAS/OpenMP threads.
    Only affects the current process, so safe to call inside workers.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch
        # limit intra-op threads (even if most work is GPU, data loaders / small CPU ops exist)
        torch.set_num_threads(1)
    except Exception:
        pass

def run_base_train(trainer_name, scenario: str, metadata: str, agg_method_name: str,single_outlier=None):
    """
    Worker entrypoint. Resolves the agg method by name inside the process.
    Returns: (msg, exp_name, accuracies_array, g_all_clients_acc, g_global_acc, result_path_str, agg_method_name)
    """
    # IMPORTANT: do this before heavy libs spin up threads:
    _limit_cpu_threads_for_child_process()

    trainer_ref = get_trainer_class(trainer_name)
    # Ensure we have an *instance*
    if inspect.isclass(trainer_ref):
        trainer = trainer_ref()  # <-- instantiate (add args if your trainer needs them)
    else:
        trainer = trainer_ref  # already an instance

    # Per-process imports (avoid heavy imports in master)
    from src.federated_adaptive_learning_nist.runners.base_concurrent_runner import BaseConcurrentRunner
    from src.federated_adaptive_learning_nist.runners.base_sequential_runner import BaseSequentialRunner
    from src.federated_adaptive_learning_nist.nist_logger import NistLogger
    from src.federated_adaptive_learning_nist.grid_searchs.grid_search_utils import select_class

    agg_cls = select_class(scenario, metadata)
    agg_method = getattr(agg_cls, agg_method_name)

    if scenario == "sequential":
        runner = BaseSequentialRunner(trainer=trainer,single_outlier=single_outlier)
    else:
        runner = BaseConcurrentRunner(trainer=trainer,single_outlier=single_outlier)

    exp_name = f"base_agg_{agg_method_name}_{metadata}_{scenario}"
    NistLogger.info(f"[Parallel] {exp_name}")

    accuracies, g_all_clients_acc, g_global_acc, result_path_str = runner.simulate(
        exp_name=exp_name,
        global_name="global",
        aggregate_method=agg_method,
        batch_size=64,
        epochs=100,
        max_round=100,
        grid_Search=False,
    )

    return (
        f"✅ Finished: {exp_name}",
        exp_name,
        accuracies,
        g_all_clients_acc,
        g_global_acc,
        result_path_str,   # NOTE: this is a str
        agg_method_name,
    )

# --------------- driver --------------------
def final_training(
    trainer_name,
    scenario: str,
    metadata: str,
    index: int,
    agg_method_name: str | None = None,
    parent_name: str = "",
    inner_max_workers: int | None = None,   # <-- NEW: lets the caller give inner pool size
    single_outlier=None,

):
    """
    Runs one job (defined by scenario/metadata/agg selection).
    If multiple agg methods resolve, fan out with a ProcessPoolExecutor (inner parallel).
    Designed to be safely called from a *threaded* outer layer (use mp_context='spawn').
    """
    print(f"WE ARE RUNNING: {trainer_name}")
    if agg_method_name == "none":
        agg_method_name = None

    param_grid = get_param_grid(scenario=scenario, metadata=metadata, agg_method_name=agg_method_name)
    print(f"GRID PARAMS: {param_grid}")
    if not param_grid:
        raise ValueError(f"No aggregation methods resolved for scenario={scenario}, metadata={metadata}")

    # run and collect (msg, exp_name, accuracies, g_all_clients_acc, g_global_acc, result_path_str, agg_method_name)
    if len(param_grid) == 1:
        single_name = param_grid[0]
        single_result = run_base_train(trainer_name, scenario, metadata, single_name,single_outlier)
        results = [single_result]
        first_result_path_str = single_result[5]
    else:
        # ---- INNER PARALLEL (processes) ----
        # Let caller overspecify inner size; else keep it bounded by CPU count / param count.
        max_workers = inner_max_workers or min(len(param_grid), (os.cpu_count() or 1))
        print(f"max_workers={max_workers}")
        # CRUCIAL: when launching processes from threads, use 'spawn'
        mp_ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
            results = list(
                executor.map(
                    run_base_train,
                    itertools.repeat(trainer_name),
                    itertools.repeat(scenario),
                    itertools.repeat(metadata),
                    param_grid,  # iterable of method names
                    single_outlier
                )
            )
        first_result_path_str = results[0][5]

    # ---- Output collation and plots ----
    parent_dir = Path(first_result_path_str) / f"{parent_name}_{trainer_name}_grid_search" / f"{scenario}_{metadata}"
    parent_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    lines = []
    overlay_bucket: dict[str, list[tuple[float, float]]] = {}

    first_ref_clients_acc = float(results[0][3])
    first_ref_global_acc = float(results[0][4])

    for (msg, exp_name, accuracies_any, g_all_clients_acc, g_global_acc, result_path_str, agg_name) in results:
        lines.append(msg)
        out_dir = parent_dir / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)

        accuracies = _normalize_accuracies(accuracies_any)

        payload = {
            "scenario": scenario,
            "metadata": metadata,
            "agg_method_name": agg_name,
            "accuracies": accuracies,
            "global_clients_all_metrics_acc": float(g_all_clients_acc),
            "global_clients_metric_acc": float(g_global_acc),
        }
        with open(out_dir / f"accuracies_{index}.json", "w") as f:
            json.dump(payload, f, indent=2)

        plot_accuracies_over_rounds(
            accuracies=accuracies,
            global_clients_all_metrics_acc=float(g_all_clients_acc),
            global_clients_metric_acc=float(g_global_acc),
            results_path=out_dir,
            scenario=scenario,
            metadata=metadata,
            tag=agg_name,
        )

        overlay_bucket[agg_name] = accuracies

        summary[exp_name] = {
            "agg_method_name": agg_name,
            "global_clients_all_metrics_acc": float(g_all_clients_acc),
            "global_clients_metric_acc": float(g_global_acc),
            "plot_path": str(out_dir / f"accuracies_over_rounds_{scenario}_{metadata}_{agg_name}.png"),
            "json_path": str(out_dir / f"accuracies_{index}.json"),
        }

    with open(parent_dir / f"summary_{index}.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_overlay_accuracies_per_scenario(
        results_by_method=overlay_bucket,
        results_path=parent_dir,
        scenario=scenario,
        metadata=metadata,
        ref_clients_acc=first_ref_clients_acc,
        ref_global_acc=first_ref_global_acc,
    )

    print("\n".join(lines))
    print(f"\nOutputs written under: {parent_dir}")

    return summary  # keep returning something usable by callers

# --------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser(description="Run base aggregation sweeps.")
    parser.add_argument("--trainer", type=str, default="FLTrainer",
                        help="Trainer class to use.")
    parser.add_argument("--scenario", type=str, default="sequential",
                        choices=["concurrent", "sequential"],   # fixed
                        help="Scenario to run.")
    parser.add_argument("--metadata", type=str, default="weights",
                        choices=["weights", "delta"],
                        help="Metadata type.")
    parser.add_argument("--index", type=int, default=0,
                        help="Index suffix for output file names.")
    parser.add_argument("--agg_method_name", type=str, default="none",
                        help="Specific aggregation method name. Use 'none' to run all methods.")
    parser.add_argument("--parent_name", type=str, default="",
                        help="Parent folder name for results.")
    parser.add_argument("--inner_max_workers", type=int, default=0,
                        help="Max workers for inner ProcessPool (0 -> auto).")
    args = parser.parse_args()

    final_training(
        trainer_name=args.trainer,
        scenario=args.scenario,
        metadata=args.metadata,
        index=args.index,
        agg_method_name=args.agg_method_name,
        parent_name=args.parent_name,
        inner_max_workers=(args.inner_max_workers or None),
        single_outlier=None
    )

if __name__ == "__main__":
    main()
