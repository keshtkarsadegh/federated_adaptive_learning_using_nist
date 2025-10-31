
"""

Purpose:
    Centralized selector for aggregation method classes in federated adaptive learning.

Provides:
    - CLASS_MAP: dictionary mapping (scenario, metadata) to the correct aggregation class.
    - select_class: safe lookup function that returns the appropriate class or raises.

Scenarios:
    - "sequential" vs "concurrent"
Metadata:
    - "weights" vs "delta"

"""


from src.federated_adaptive_learning_nist.aggregation_methods.concurrent_base_agg_methods import \
    ConcurrentWeightsAGGMETHODS,ConcurrentDeltaAGGMETHODS
from src.federated_adaptive_learning_nist.aggregation_methods.sequential_base_agg_methods import \
    SequentialWeightsAGGMETHODS, SequentialDeltaAGGMETHODS


# Example: map by (module_id, class_id)
CLASS_MAP = {
    "sequential": {"weights": SequentialWeightsAGGMETHODS, "delta":SequentialDeltaAGGMETHODS},
    "concurrent": {"weights": ConcurrentWeightsAGGMETHODS, "delta":ConcurrentDeltaAGGMETHODS}
}

def select_class(scenario, metadata):
    try:
        return CLASS_MAP[scenario][metadata]
    except KeyError:
        raise ValueError(f"Invalid selection: {scenario}, {metadata}")
