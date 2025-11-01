import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the real driver
from src.federated_adaptive_learning_nist.trainings.base_trainings import final_training
"""
extreme_outlier_trainings.py

Purpose:
    Launches grid_search runs specifically for "extreme case" experiments in
    federated adaptive learning with NIST using knowledge distillation.

Extreme Cases Supported:
    - Single outlier: restrict training to one writer ID.
    - Two outliers: restrict training to exactly two writer IDs.
    - Dual outlier: repeat the same writer ID twice to simulate duplication.

Key Notes:
    - Trainer is fixed to DistillationTrainer (knowledge distillation),
      as this is the best-performing approach in our study.
    - Jobs use only concurrent/weights setup here (the most relevant for extremes).
    - To actually run these cases, you must also edit the runner call
      (BaseConcurrentRunner.simulate or BaseSequentialRunner.simulate) to pass
      the correct `single_outlier` list of client IDs. For example:
          ["f3503_07"]               → single outlier
          ["f3503_07", "f2801_11"]   → two outliers
          ["f3503_07", "f3503_07"]   → dual/repeated outlier

Parallelism:
    - Outer level: threads (jobs).
    - Inner level: processes inside grid_search for parameter sweeps.

Outputs:
    - Results stored under results/<parent_name>/.
    - Trained models and accuracy JSON files for each extreme setup.

Usage:
    $ python extreme_outlier_trainings.py
"""


def generate(trainer_name: str,
             scenario: str,
             metadata: str,
             index: int,
             agg_method_name: str,
             parent_name: str,
             inner_max_workers: int = 12,single_outlier=None):

    """
    Run one grid_search() job with the specified configuration.

    Args:
        trainer_name (str): Trainer class to use (here always "DistillationTrainer").
        scenario (str): Training scenario ("concurrent" or "sequential").
        metadata (str): Aggregation metadata ("weights" or "delta").
        index (int): Index for experiment repetition.
        agg_method_name (str): Aggregation method name ("none" for default).
        parent_name (str): Parent folder name for results.
        inner_max_workers (int): Number of workers for inner process pool.

    Returns:
        Any: Result object returned by grid_search().
    """
    if agg_method_name == "none":
        agg_method_name = None
    return final_training(
        trainer_name=trainer_name,
        scenario=scenario,
        metadata=metadata,
        index=index,
        agg_method_name=agg_method_name,
        parent_name=parent_name,
        inner_max_workers=inner_max_workers,
        single_outlier=single_outlier
    )


def run_all_parallel(trainer_name="EWCTrainer",
                     outer_max_workers: int = 4,
                     inner_max_workers: int = 8,
                     parent_name="final_result",
                     single_outlier=None):
    """
    Run all extreme-case jobs in parallel threads.

    Jobs here are simplified (concurrent × weights), since extreme-case
    experiments focus on that configuration with knowledge distillation.

    Args:
        trainer_name (str): Trainer class (fixed to "DistillationTrainer").
        outer_max_workers (int): Number of parallel jobs at thread level.
        inner_max_workers (int): Number of workers for inner process pool.
        parent_name (str): Folder to save results under results/.

    Returns:
        list: Results from all submitted grid_search jobs.
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
            outer.submit(generate, inner_max_workers=inner_max_workers,single_outlier=single_outlier, **job)
            for job in jobs
        ]
        for f in as_completed(futs):
            results.append(f.result())
    return results


def extreme_cases_final_training(single_outlier=None,parent_name="",):
    all_results = run_all_parallel(
        trainer_name="DistillationTrainer",
        outer_max_workers=4,
        inner_max_workers=4,
        parent_name=parent_name,
        single_outlier=single_outlier
    )

    print("All jobs done:", all_results)


if __name__ == "__main__":

    all_results = run_all_parallel(
        trainer_name="DistillationTrainer",
        outer_max_workers=1,
        inner_max_workers=20,
        parent_name="dual_outlier_final_results_final",
    )

    print("All jobs done:", all_results)

