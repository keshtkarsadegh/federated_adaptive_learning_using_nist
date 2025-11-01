from concurrent.futures import ThreadPoolExecutor, as_completed
from src.federated_adaptive_learning_nist.trainings.base_trainings import final_training
"""
training_all_aggs.py — Baseline Federated Learning Aggregation Experiments

Purpose:
    Provides a unified driver to train the BaseTrainer across *all* aggregation
    method variants defined in the framework:
        - Sequential (weights / delta)
        - Concurrent (weights / delta)

    The goal is to systematically benchmark and validate each aggregation method
    with a consistent baseline trainer, ensuring reproducibility and correctness
    of all implementations.

Execution Model:
    - Outer parallelism (threads):
        One thread per (scenario, metadata, aggregation) configuration.
    - Inner parallelism (processes):
        Each thread calls grid_search(), which performs parameter sweeps
        and training inside a process pool.

Usage:
    $ python training_all_aggs.py
    (tune outer_max_workers and inner_max_workers as needed)

Outputs:
    - Trained models, metrics JSONs, and accuracy logs under results/<parent_name>/.
    - Each run captures:
        • Global test accuracy
        • Outlier/client accuracy
        • Round-by-round training/validation metrics

Notes:
    - This module is intended for **baseline experiments** only.
      For advanced methods (EWC, KD, prox, feature-alignment, etc.),
      separate launchers are provided.
    - By default, only a subset of jobs may be active for debugging
      (see the jobs[] list). Re-enable the full set to sweep all methods.
"""



def run_all_parallel(outer_max_workers: int = 3, inner_max_workers: int = 12):
    """
        Launch all configured jobs in parallel.

        Args:
            outer_max_workers (int): Number of threads for the outer pool
                (each handles one job config).
            inner_max_workers (int): Number of processes used by grid_search
                inside each job.

        Returns:
            list: Results from each job’s grid_search call.
        """
    jobs = [
        # sequential — weights
        dict(trainer_name="BaseTrainer",
             scenario="sequential",
             metadata="weights",
             index=0,
             agg_method_name="none",
             parent_name="prove_fl"),

        # sequential — delta
        dict(trainer_name="BaseTrainer",
             scenario="sequential",
             metadata="delta",
             index=0,
             agg_method_name="none",
             parent_name="all_aggs_fl"),

        # concurrent — weights (cgw)
        dict(trainer_name="BaseTrainer",
             scenario="concurrent",
             metadata="weights",
             index=0,
             agg_method_name="none",
             parent_name="all_aggs_fl"),

        # concurrent — delta (cgd)
        dict(trainer_name="BaseTrainer",
             scenario="concurrent",
             metadata="delta",
             index=0,
             agg_method_name="none",
             parent_name="all_aggs_fl"),
    ]

    results = []
    with ThreadPoolExecutor(max_workers=outer_max_workers) as outer:
        futs = [
            outer.submit(
                final_training,
                inner_max_workers=inner_max_workers,  # <-- inner pool larger than outer
                **job
            )
            for job in jobs
        ]
        for f in as_completed(futs):
            results.append(f.result())
    return results

def all_aggregation_variants():
    for r in run_all_parallel(outer_max_workers=1, inner_max_workers=20):
        print("✅", r)
if __name__ == "__main__":

    all_aggregation_variants()

