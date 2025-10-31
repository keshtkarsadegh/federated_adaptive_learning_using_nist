# run_all.py
import sys
print(">>> Job started, running base_trainings_fl.py", flush=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from concurrent.futures import ThreadPoolExecutor, as_completed
from src.federated_adaptive_learning_nist.trainings.base_trainings import grid_search
"""
base_trainings_fl.py — PARALLEL JOB DRIVER

Purpose:
    Orchestrates federated learning training runs in parallel across multiple
    scenarios (sequential vs concurrent, weights vs delta) using the BaseTrainer.

Capabilities:
    - Defines a set of federated training jobs with specific aggregation methods.
    - Executes jobs in parallel using a thread pool.
    - Delegates training logic to `grid_search` from base_trainings.
    - Collects and returns results from all jobs.

Jobs configured:
    • Sequential Weights (FedAvg update)
    • Sequential Delta (FedAvg update)
    • Concurrent Weights (weighted_cgw, weighted_cw)
    • Concurrent Delta (weighted_cgd)

Parallelism:
    - Outer parallelism (threads): distributes jobs across workers.
    - Inner parallelism (processes): controlled by `inner_max_workers`,
      passed through to `grid_search`.

Usage:
    $ python base_trainings_fl.py

    Main loop calls `run_all_parallel`, launches all jobs, and prints results.
"""

def run_all_parallel(outer_max_workers: int = 3, inner_max_workers: int = 12):
    """
       Launch all predefined federated training jobs in parallel.

       Args:
           outer_max_workers (int): Max number of jobs to run in parallel (thread pool).
           inner_max_workers (int): Max workers for each job's inner pool (process pool in grid_search).

       Returns:
           list: A list of results from all completed jobs.
                 Each result is whatever `grid_search` returns for its configuration.

       Jobs include:
           - Sequential Weights:   agg_method = seq_fedavg_update
           - Sequential Delta:     agg_method = seq_delta_fedavg_update
           - Concurrent Weights:   agg_method = con_weighted_cgw, con_weighted_cw
           - Concurrent Delta:     agg_method = con_delta_weighted_cgd
       """
    jobs = [
        # sequential — weights
        dict(trainer_name="BaseTrainer",
             scenario="sequential",
             metadata="weights",
             index=0,
             agg_method_name="seq_fedavg_update",
             parent_name="prove_fl"),

        # sequential — delta
        dict(trainer_name="BaseTrainer",
             scenario="sequential",
             metadata="delta",
             index=0,
             agg_method_name="seq_delta_fedavg_update",
             parent_name="prove_fl"),

        # concurrent — weights (cgw)
        dict(trainer_name="BaseTrainer",
             scenario="concurrent",
             metadata="weights",
             index=0,
             agg_method_name="con_weighted_cgw",
             parent_name="prove_fl"),

        # concurrent — weights (cw)
        dict(trainer_name="BaseTrainer",
             scenario="concurrent",
             metadata="weights",
             index=0,
             agg_method_name="con_weighted_cw",
             parent_name="prove_fl"),

        # concurrent — delta (cgd)
        dict(trainer_name="BaseTrainer",
             scenario="concurrent",
             metadata="delta",
             index=0,
             agg_method_name="con_delta_weighted_cgd",
             parent_name="prove_fl"),
    ]


    results = []
    with ThreadPoolExecutor(max_workers=outer_max_workers) as outer:
        futs = [
            outer.submit(
                grid_search,
                inner_max_workers=inner_max_workers,  # <-- inner pool larger than outer
                **job
            )
            for job in jobs
        ]
        for f in as_completed(futs):
            results.append(f.result())
    return results


def base_fl_training():
    for r in run_all_parallel(outer_max_workers=4, inner_max_workers=1):
        print("✅", r)

if __name__ == "__main__":

   base_fl_training()
