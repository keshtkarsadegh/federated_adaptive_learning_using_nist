from copy import deepcopy

import torch

"""
Concurrent aggregation strategies for federated learning.

This module implements several variants:
- *_cw  : client-only weighted/scaled/capped averaging
- *_cgw : client + global mixed averaging
- *_cgd : delta-based updates (client update relative to global)

Grouped into:
    - ConcurrentWeightsAGGMETHODS: operates directly on weights
    - ConcurrentDeltaAGGMETHODS : operates on deltas
"""


def con_weighted_cw(global_weight, clients_weights_list, client_sample_counts):
    """
        Concurrent Weighted Aggregation (only client weights).

        Each client model is weighted by its sample count relative to the total,
        and then averaged to form the new model. No global mixing.

        Args:
            global_weight (dict): Current global model state_dict.
            clients_weights_list (list[dict]): List of client state_dicts.
            client_sample_counts (list[int]): Number of samples per client.

        Returns:
            dict: Averaged model weights.
        """
    avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    total = sum(client_sample_counts)
    for key in avg:
        for w, cnt in zip(clients_weights_list, client_sample_counts):
            avg[key] += w[key] * (cnt / total)
    return avg

def con_weighted_cgw(global_weight, clients_weights_list, client_sample_counts):
    """
    Concurrent Weighted Aggregation (clients and global model).

    Clients are aggregated weighted by sample counts (like con_weighted_cw),
    then the result is mixed 50/50 with the existing global model.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    client_avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    total = sum(client_sample_counts)
    for key in client_avg:
        for w, cnt in zip(clients_weights_list, client_sample_counts):
            client_avg[key] += w[key] * (cnt / total)
    # mix with global
    avg = {}
    for key in global_weight:
        avg[key] = 0.5 * global_weight[key] + 0.5 * client_avg[key]
    return avg


def con_scaled_cw(global_weight, clients_weights_list, client_sample_counts):
    """
    Concurrent scaled Aggregation (client weights).

    Each client contributes equally regardless of data size.
    Average is uniform across clients. No global mixing.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """

    K = len(client_sample_counts)
    avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    scale = 1.0 / K
    for key in avg:
        for w in clients_weights_list:
            avg[key] += w[key] * (scale )
    return avg

def con_scaled_cgw(global_weight, clients_weights_list, client_sample_counts):
    """
    Concurrent scaled Aggregation (client and global weights).

    Clients are averaged uniformly (equal weights), then result is blended
    50/50 with the current global model.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    K = len(client_sample_counts)
    # compute scaled client avg
    client_avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    scale = 1.0 / K
    for key in client_avg:
        for w in clients_weights_list:
            client_avg[key] += w[key] * (scale )
    # mix with global
    avg = {}
    for key in global_weight:
        avg[key] = 0.5* global_weight[key] + 0.5 * client_avg[key]
    return avg


def con_capped_cw(global_weight, clients_weights_list, client_sample_counts):
    """
     Concurrent Capped Aggregation (client weights).

     Client contributions are capped so no single client dominates:
    - Proportions = min(n_i / total, 1/K)
    - Then renormalized and used for weighted averaging.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    K = len(client_sample_counts)
    total = sum(client_sample_counts)
    # Step 1: cap proportions
    raw = [min(cnt/total, 1.0/K) for cnt in client_sample_counts]
    # Step 2: renormalize
    Z = sum(raw)
    q = [r / Z for r in raw]
    # aggregate
    avg = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    for key in avg:
        for w, qk in zip(clients_weights_list, q):
            avg[key] += w[key] * qk
    return avg


def con_capped_cgw(global_weight, clients_weights_list, client_sample_counts):
    """
     Concurrent Capped Aggregation (client and global weights).

     Client contributions are capped so no single client dominates:
    - Proportions = min(n_i / total, 1/K)
    - Then renormalized and used for weighted averaging.
    - then blend 50/50 with the global model.

    Args:
        global_weight (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    K = len(client_sample_counts)
    total = sum(client_sample_counts)
    raw = [min(cnt/total, 1.0/K) for cnt in client_sample_counts]
    Z = sum(raw)
    q = [r / Z for r in raw]
    # client contribution
    client_part = {k: torch.zeros_like(v) for k, v in global_weight.items()}
    for key in client_part:
        for w, qk in zip(clients_weights_list, q):
            client_part[key] += w[key] * qk
    avg = {}
    for key in global_weight:
        avg[key] = 0.5 * global_weight[key] + 0.5 * client_part[key]
    return avg
class ConcurrentWeightsAGGMETHODS:
    """
    Container class exposing concurrent aggregation methods
    operating directly on client weights (no deltas).
    """
    con_weighted_cw=staticmethod(con_weighted_cw)
    con_weighted_cgw=staticmethod(con_weighted_cgw)
    con_scaled_cw=staticmethod(con_scaled_cw)
    con_scaled_cgw=staticmethod(con_scaled_cgw)
    con_capped_cw=staticmethod(con_capped_cw)
    con_capped_cgw=staticmethod(con_capped_cgw)
    def __iter__(self):
        for name in dir(self):
            if name.startswith("con_"):
                method = getattr(self, name)
                if callable(method):
                    yield method

    @classmethod
    def list_methods(cls):
        """
        Returns:
            (list[str], list[function]):
            Names and function handles of all con_* methods.
        """
        names, fns = [], []
        for name in dir(cls):
            if name.startswith("con_"):
                # getattr on a staticmethod returns the underlying function
                fn = getattr(cls, name)
                if callable(fn):
                    names.append(name)
                    fns.append(fn)
        return names, fns






def con_delta_weighted_cgd(global_weights, clients_weights_list, client_sample_counts):
    """
    Concurrent Delta Aggregation (Weighted).

    Clients contribute deltas (difference from global).
    Each delta is weighted by client sample proportion,
    then added to the global model.
    Args:
        global_weights (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    total = sum(client_sample_counts)
    new_w = deepcopy(global_weights)

    for key in global_weights:
        delta_sum = torch.zeros_like(global_weights[key])

        for w_k, n_k in zip(clients_weights_list, client_sample_counts):
            delta = w_k[key] - global_weights[key]
            delta_sum += (n_k / total) * delta

        # Weighted combination: 0.7 * global + 0.3 * delta_sum
        new_w[key] =  global_weights[key] + delta_sum

    return new_w


def con_delta_scaled_cgd(global_weights, clients_weights_list, client_sample_counts):
    """
    Concurrent Delta Aggregation (Scaled).

    Each client contributes equally (1/K factor),
    deltas are averaged uniformly and added to global.
    Args:
        global_weights (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    K = len(client_sample_counts)
    factor = 1.0 / (K )

    new_w = deepcopy(global_weights)
    for key in global_weights:
        delta_sum = torch.zeros_like(global_weights[key])
        for w_k in clients_weights_list:
            delta = w_k[key] - global_weights[key]
            delta_sum += factor  * delta
        new_w[key] =  global_weights[key] +  delta_sum


    return new_w

def con_delta_capped_cgd(global_weights, clients_weights_list, client_sample_counts):
    """
    Concurrent Delta Aggregation (Capped).

    Client proportions are capped at 1/K to prevent domination.
    Normalized capped proportions weight the client deltas,
    which are then added to the global model.
    Args:
        global_weights (dict): Current global model state_dict.
        clients_weights_list (list[dict]): List of client state_dicts.
        client_sample_counts (list[int]): Number of samples per client.

    Returns:
        dict: Averaged model weights.
    """
    total = sum(client_sample_counts)
    K = len(client_sample_counts)
    max_cap = 1.0 / K

    # Step 1: compute raw proportions and cap
    raw = [min(n_k / total, max_cap) for n_k in client_sample_counts]
    # Step 2: renormalize
    Z = sum(raw)
    q = [r / Z for r in raw]

    new_w = deepcopy(global_weights)
    for key in global_weights:
        delta_sum = torch.zeros_like(global_weights[key])
        for w_k, q_k in zip(clients_weights_list, q):
            delta = w_k[key] - global_weights[key]
            delta_sum +=   q_k * delta
        new_w[key] =  global_weights[key] + delta_sum


    return new_w

class ConcurrentDeltaAGGMETHODS:
    """
    Container class exposing concurrent aggregation methods
    that operate on deltas (client update relative to global).
    """
    con_delta_weighted_cgd=staticmethod(con_delta_weighted_cgd)
    con_delta_scaled_cgd=staticmethod(con_delta_scaled_cgd)
    con_delta_capped_cgd=staticmethod(con_delta_capped_cgd)

    def __iter__(self):
        for name in dir(self):
            if name.startswith("con_delta_"):
                method = getattr(self, name)
                if callable(method):
                    yield method

    @classmethod
    def list_methods(cls):
        """
        Returns:
            (list[str], list[function]):
            Names and function handles of all con_delta_* methods.
        """
        names, fns = [], []
        for name in dir(cls):
            if name.startswith("con_delta_"):
                # getattr on a staticmethod returns the underlying function
                fn = getattr(cls, name)
                if callable(fn):
                    names.append(name)
                    fns.append(fn)
        return names, fns