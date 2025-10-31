import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the real driver
from src.federated_adaptive_learning_nist.trainings.base_trainings import grid_search
"""

Purpose:
    Orchestrates grid search runs for federated training with EWC (Elastic Weight Consolidation)
    or any other trainer passed in (e.g., KDTrainer, BaseTrainer).

Capabilities:
    - Wraps grid_search() calls for different (scenario, metadata) combinations.
    - Supports sequential vs. concurrent and weights vs. delta setups.
    - Runs jobs in parallel using ThreadPoolExecutor.
    - CLI arguments to configure trainer, concurrency, and results folder.

Usage:
    $ python generate_experiments.py --trainer EWCTrainer --outer_max_workers 4 --inner_max_workers 16 --parent_name results_run1

Inputs:
    --trainer            Trainer class name (EWCTrainer, FLTrainer, KDTrainer, etc.)
    --outer_max_workers  Number of parallel top-level jobs (default=4).
    --inner_max_workers  Number of workers for the process pool inside grid_search (default=20).
    --parent_name        Parent folder name under results/ (default="final_result").

Outputs:
    - Results JSON/plots saved under results/<parent_name>/.
    - Console logs summarizing completion per job.
"""


def generate(trainer_name: str,
             scenario: str,
             metadata: str,
             index: int,
             agg_method_name: str,
             parent_name: str,
             inner_max_workers: int = 12):
    """
    Run a single grid_search (from base trainer) job with the specified configuration.

    Args:
        trainer_name (str): Trainer class to instantiate (e.g., EWCTrainer).
        scenario (str): Training scenario, either "sequential" or "concurrent".
        metadata (str): Aggregation style, "weights" or "delta".
        index (int): Index for experiment repetition.
        agg_method_name (str): Aggregation method name, or "none" for default.
        parent_name (str): Parent folder for saving results.
        inner_max_workers (int): Number of processes in the inner pool.

    Returns:
        Any: The result object returned by grid_search().
    """
    if agg_method_name == "none":
        agg_method_name = None
    return grid_search(
        trainer_name=trainer_name,
        scenario=scenario,
        metadata=metadata,
        index=index,
        agg_method_name=agg_method_name,
        parent_name=parent_name,
        inner_max_workers=inner_max_workers,
    )


def run_all_parallel(trainer_name="EWCTrainer",
                     outer_max_workers: int = 4,
                     inner_max_workers: int = 8,
                     parent_name="final_result"):
    """
    Run all standard jobs (sequential/concurrent × weights/delta) in parallel.

    Args:
        trainer_name (str): Trainer class (default="EWCTrainer").
        outer_max_workers (int): Number of parallel jobs at the thread level.
        inner_max_workers (int): Number of processes per grid_search call.
        parent_name (str): Parent folder name for results.

    Returns:
        list: List of results from all grid_search (from base trainer) calls.
    """
    jobs = [
        dict(trainer_name=trainer_name,
             scenario="sequential",
             metadata="weights",
             index=0,
             agg_method_name="none",
             parent_name=parent_name),

        dict(trainer_name=trainer_name,
             scenario="sequential",
             metadata="delta",
             index=0,
             agg_method_name="none",
             parent_name=parent_name),

        dict(trainer_name=trainer_name,
             scenario="concurrent",
             metadata="weights",
             index=0,
             agg_method_name="none",
             parent_name=parent_name),

        dict(trainer_name=trainer_name,
             scenario="concurrent",
             metadata="delta",
             index=0,
             agg_method_name="none",
             parent_name=parent_name),
    ]


    results = []
    with ThreadPoolExecutor(max_workers=outer_max_workers) as outer:
        futs = [
            outer.submit(generate, inner_max_workers=inner_max_workers, **job)
            for job in jobs
        ]
        for f in as_completed(futs):
            results.append(f.result())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, required=True,
                        help="Trainer class name, e.g. EWCTrainer, FLTrainer, KDTrainer")
    parser.add_argument("--outer_max_workers", type=int, default=4,
                        help="Number of parallel top-level jobs")
    parser.add_argument("--inner_max_workers", type=int, default=20,
                        help="Number of workers for inner ProcessPool in grid_search")
    parser.add_argument("--parent_name", type=str, default="final_result",
                        help="Parent folder for results")

    args = parser.parse_args()

    all_results = run_all_parallel(
        trainer_name=args.trainer,
        outer_max_workers=args.outer_max_workers,
        inner_max_workers=args.inner_max_workers,
        parent_name=args.parent_name,
    )

    print("All jobs done:", all_results)

