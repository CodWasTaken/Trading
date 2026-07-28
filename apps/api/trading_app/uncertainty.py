from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any

from .research import _ScoredRow, _selected_groups


def simulation_period_returns(
    scored_rows: list[_ScoredRow],
    *,
    threshold: float,
    transaction_cost_bps: float,
    always_long: bool = False,
) -> list[float]:
    selected_groups = _selected_groups(scored_rows)
    previous_positions: dict[str, float] = defaultdict(float)
    period_returns: list[float] = []
    for timestamp_rows in selected_groups:
        allocation = 1.0 / len(timestamp_rows)
        period_net = 0.0
        for item in timestamp_rows:
            row = item.row
            position = 1.0 if always_long or item.prediction > threshold else 0.0
            turnover = abs(position - previous_positions[row.symbol])
            cost = turnover * transaction_cost_bps / 10_000
            period_net += allocation * (position * row.target_return - cost)
            previous_positions[row.symbol] = position
        period_returns.append(period_net)
    if not period_returns:
        raise ValueError("Simulation produced no non-overlapping periods")
    return period_returns


def block_bootstrap_evidence(
    candidate_returns: list[float],
    benchmark_returns: list[float],
    *,
    samples: int = 2000,
    confidence_level: float = 0.95,
    candidate_family_size: int = 1,
    block_size: int | None = None,
    seed_material: str = "",
) -> dict[str, object]:
    if len(candidate_returns) != len(benchmark_returns):
        raise ValueError("Candidate and benchmark period returns must align")
    if not candidate_returns:
        raise ValueError("Bootstrap requires period returns")
    if samples < 200:
        raise ValueError("Bootstrap samples must be at least 200")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1")
    if candidate_family_size < 1:
        raise ValueError("candidate_family_size must be positive")

    period_count = len(candidate_returns)
    effective_block_size = (
        max(1, round(period_count ** (1 / 3))) if block_size is None else block_size
    )
    if effective_block_size < 1 or effective_block_size > period_count:
        raise ValueError("block_size must be between 1 and the period count")

    family_alpha = (1.0 - confidence_level) / candidate_family_size
    adjusted_confidence = 1.0 - family_alpha
    lower_probability = family_alpha / 2.0
    upper_probability = 1.0 - lower_probability
    seed = int.from_bytes(
        hashlib.sha256(
            (
                f"{seed_material}|{samples}|{confidence_level}|"
                f"{candidate_family_size}|{effective_block_size}"
            ).encode()
        ).digest()[:8],
        "big",
    )
    generator = random.Random(seed)

    candidate_samples: list[float] = []
    benchmark_samples: list[float] = []
    excess_samples: list[float] = []
    for _ in range(samples):
        indices: list[int] = []
        while len(indices) < period_count:
            start = generator.randrange(period_count)
            indices.extend(
                (start + offset) % period_count
                for offset in range(effective_block_size)
            )
        selected = indices[:period_count]
        candidate = _compound(candidate_returns[index] for index in selected)
        benchmark = _compound(benchmark_returns[index] for index in selected)
        candidate_samples.append(candidate)
        benchmark_samples.append(benchmark)
        excess_samples.append(candidate - benchmark)

    candidate_samples.sort()
    benchmark_samples.sort()
    excess_samples.sort()
    candidate_point = _compound(candidate_returns)
    benchmark_point = _compound(benchmark_returns)
    excess_point = candidate_point - benchmark_point
    return {
        "method": "deterministic_circular_moving_block_bootstrap",
        "samples": samples,
        "periods": period_count,
        "block_size": effective_block_size,
        "nominal_confidence_level": confidence_level,
        "candidate_family_size": candidate_family_size,
        "adjusted_confidence_level": adjusted_confidence,
        "multiple_comparison_adjustment": "bonferroni",
        "seed_sha256": hashlib.sha256(seed_material.encode()).hexdigest(),
        "net_return": _interval(
            candidate_point,
            candidate_samples,
            lower_probability,
            upper_probability,
        ),
        "benchmark_net_return": _interval(
            benchmark_point,
            benchmark_samples,
            lower_probability,
            upper_probability,
        ),
        "excess_return_vs_benchmark": _interval(
            excess_point,
            excess_samples,
            lower_probability,
            upper_probability,
        ),
        "probabilities": {
            "net_return_positive": sum(value > 0 for value in candidate_samples) / samples,
            "excess_return_positive": sum(value > 0 for value in excess_samples) / samples,
        },
        "interpretation": [
            "Intervals describe resampled historical uncertainty, not guaranteed future outcomes.",
            "Bonferroni adjustment uses the predeclared candidate family size from the sealed split.",
            "Serial dependence is approximated with circular moving blocks.",
        ],
    }


def _compound(values: Any) -> float:
    return math.prod(1.0 + float(value) for value in values) - 1.0


def _interval(
    point: float,
    samples: list[float],
    lower_probability: float,
    upper_probability: float,
) -> dict[str, float]:
    return {
        "point": point,
        "lower": _quantile(samples, lower_probability),
        "upper": _quantile(samples, upper_probability),
    }


def _quantile(ordered: list[float], probability: float) -> float:
    if not ordered:
        raise ValueError("Quantile requires samples")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
