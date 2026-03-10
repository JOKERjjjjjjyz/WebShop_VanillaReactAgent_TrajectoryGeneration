#!/usr/bin/env python3
"""
compute_metrics_v0.py

Metric utilities for synthetic hallucination propagation simulations.
V0 scope: EPC, HTP, PRI-ready hooks, CD, CM, TTI + bootstrap CI.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random


@dataclass
class MetricConfig:
    tti_threshold: float = 0.1
    tti_consecutive_depths: int = 2
    bootstrap_samples: int = 200
    bootstrap_alpha: float = 0.05


def _parents_map(edges: Sequence[Tuple[int, int, float]]) -> Dict[int, List[Tuple[int, float]]]:
    parents = defaultdict(list)
    for src, dst, w in edges:
        parents[dst].append((src, w))
    return parents


def _children_map(edges: Sequence[Tuple[int, int, float]]) -> Dict[int, List[Tuple[int, float]]]:
    children = defaultdict(list)
    for src, dst, w in edges:
        children[src].append((dst, w))
    return children


def _depth_groups(depths: Dict[int, int]) -> Dict[int, List[int]]:
    groups = defaultdict(list)
    for nid, d in depths.items():
        groups[d].append(nid)
    return dict(groups)


def contamination_mass(h_true: Dict[int, int]) -> float:
    if not h_true:
        return 0.0
    return sum(h_true.values()) / float(len(h_true))


def contamination_depth(depths: Dict[int, int], h_true: Dict[int, int]) -> Dict[str, float]:
    contaminated_depths = [depths[nid] for nid, val in h_true.items() if val == 1]
    if not contaminated_depths:
        return {"cd_max": 0.0, "cd_mean": 0.0}
    return {
        "cd_max": float(max(contaminated_depths)),
        "cd_mean": float(sum(contaminated_depths) / len(contaminated_depths)),
    }


def hallucination_transmission_probability(
    edges: Sequence[Tuple[int, int, float]],
    h_true: Dict[int, int],
) -> float:
    """
    Empirical P(child contaminated | contaminated parent) at edge level.
    """
    exposed_edges = 0
    transmitted = 0
    for src, dst, _w in edges:
        if h_true.get(src, 0) == 1:
            exposed_edges += 1
            if h_true.get(dst, 0) == 1:
                transmitted += 1
    if exposed_edges == 0:
        return 0.0
    return transmitted / float(exposed_edges)


def expected_propagation_cost(
    edges: Sequence[Tuple[int, int, float]],
    depths: Dict[int, int],
    h_true: Dict[int, int],
    criticality_by_depth: Optional[Dict[int, float]] = None,
) -> float:
    """
    V0 estimator: weighted contaminated descendants count per contaminated node,
    aggregated then normalized by node count.
    """
    if criticality_by_depth is None:
        criticality_by_depth = defaultdict(lambda: 1.0)

    children = _children_map(edges)

    def descendant_cost(root: int) -> float:
        seen = set()
        stack = [root]
        total = 0.0
        while stack:
            node = stack.pop()
            for child, _w in children.get(node, []):
                if child in seen:
                    continue
                seen.add(child)
                stack.append(child)
                if h_true.get(child, 0) == 1:
                    total += float(criticality_by_depth.get(depths.get(child, 0), 1.0))
        return total

    contaminated = [nid for nid, hv in h_true.items() if hv == 1]
    if not contaminated:
        return 0.0

    raw = sum(descendant_cost(nid) for nid in contaminated)
    return raw / float(max(1, len(h_true)))


def time_to_isolation(
    depths: Dict[int, int],
    p_h: Dict[int, float],
    threshold: float,
    consecutive_depths: int,
) -> int:
    """
    Depth index where mean p_h falls below threshold for m consecutive layers.
    Returns -1 if never isolated.
    """
    groups = _depth_groups(depths)
    if not groups:
        return -1

    sorted_depths = sorted(groups.keys())
    streak = 0
    for d in sorted_depths:
        vals = [p_h.get(nid, 0.0) for nid in groups[d]]
        mean_p = sum(vals) / float(max(1, len(vals)))
        if mean_p < threshold:
            streak += 1
            if streak >= consecutive_depths:
                return d
        else:
            streak = 0
    return -1


def verifier_confusion(h_true: Dict[int, int], v_obs: Dict[int, int]) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    for nid, hv in h_true.items():
        vv = v_obs.get(nid, 0)
        if hv == 1 and vv == 1:
            tp += 1
        elif hv == 0 and vv == 0:
            tn += 1
        elif hv == 0 and vv == 1:
            fp += 1
        elif hv == 1 and vv == 0:
            fn += 1

    total = max(1, tp + tn + fp + fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total,
        "precision": tp / max(1, (tp + fp)),
        "recall": tp / max(1, (tp + fn)),
        "fpr_empirical": fp / max(1, (fp + tn)),
        "fnr_empirical": fn / max(1, (fn + tp)),
    }


def _bootstrap_ci(values: Sequence[float], samples: int, alpha: float, seed: int = 17) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        pick = [values[rng.randrange(n)] for __ in range(n)]
        means.append(sum(pick) / n)
    means.sort()
    lo_idx = int((alpha / 2.0) * (len(means) - 1))
    hi_idx = int((1 - alpha / 2.0) * (len(means) - 1))
    return means[lo_idx], means[hi_idx]


def compute_all_metrics(
    edges: Sequence[Tuple[int, int, float]],
    depths: Dict[int, int],
    h_true: Dict[int, int],
    v_obs: Dict[int, int],
    p_h: Dict[int, float],
    criticality_by_depth: Optional[Dict[int, float]] = None,
    cfg: Optional[MetricConfig] = None,
) -> Dict[str, object]:
    if cfg is None:
        cfg = MetricConfig()

    cm = contamination_mass(h_true)
    cd = contamination_depth(depths, h_true)
    htp = hallucination_transmission_probability(edges, h_true)
    epc = expected_propagation_cost(edges, depths, h_true, criticality_by_depth)
    tti = time_to_isolation(
        depths,
        p_h,
        threshold=cfg.tti_threshold,
        consecutive_depths=cfg.tti_consecutive_depths,
    )
    confusion = verifier_confusion(h_true, v_obs)

    # Node-level bootstrap inputs for metric uncertainty approximation.
    pvals = list(p_h.values())
    hvals = [float(v) for v in h_true.values()]
    bootstrap = {
        "cm_ci": _bootstrap_ci(hvals, cfg.bootstrap_samples, cfg.bootstrap_alpha),
        "p_h_mean_ci": _bootstrap_ci(pvals, cfg.bootstrap_samples, cfg.bootstrap_alpha),
    }

    return {
        "EPC": epc,
        "HTP": htp,
        "PRI": None,  # computed in paired-run analysis stage
        "CD": cd,
        "CM": cm,
        "TTI": tti,
        "bootstrap": bootstrap,
        "verifier_confusion": confusion,
    }
