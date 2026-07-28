import pytest

from trading_app.uncertainty import block_bootstrap_evidence


def test_block_bootstrap_is_deterministic_and_family_adjusted() -> None:
    candidate = [0.01, -0.004, 0.012, 0.003, 0.008, -0.002] * 6
    benchmark = [0.004, -0.003, 0.005, 0.002, 0.004, -0.001] * 6

    first = block_bootstrap_evidence(
        candidate,
        benchmark,
        samples=400,
        confidence_level=0.95,
        candidate_family_size=5,
        block_size=3,
        seed_material="sealed-split|model-v1",
    )
    second = block_bootstrap_evidence(
        candidate,
        benchmark,
        samples=400,
        confidence_level=0.95,
        candidate_family_size=5,
        block_size=3,
        seed_material="sealed-split|model-v1",
    )

    assert first == second
    assert first["method"] == "deterministic_circular_moving_block_bootstrap"
    assert first["adjusted_confidence_level"] == pytest.approx(0.99)
    assert first["multiple_comparison_adjustment"] == "bonferroni"
    assert first["block_size"] == 3
    assert 0 <= first["probabilities"]["net_return_positive"] <= 1
    assert 0 <= first["probabilities"]["excess_return_positive"] <= 1


def test_strong_positive_candidate_has_positive_adjusted_lower_bounds() -> None:
    candidate = [0.01, 0.012, 0.008, 0.015, 0.009, 0.011] * 10
    benchmark = [0.001, 0.002, 0.0, 0.001, 0.002, 0.001] * 10

    evidence = block_bootstrap_evidence(
        candidate,
        benchmark,
        samples=500,
        confidence_level=0.95,
        candidate_family_size=4,
        block_size=4,
        seed_material="positive-evidence",
    )

    assert evidence["net_return"]["lower"] > 0
    assert evidence["excess_return_vs_benchmark"]["lower"] > 0
    assert evidence["probabilities"]["net_return_positive"] == 1.0
    assert evidence["probabilities"]["excess_return_positive"] == 1.0


def test_bootstrap_rejects_misaligned_returns() -> None:
    with pytest.raises(ValueError, match="must align"):
        block_bootstrap_evidence(
            [0.01, 0.02],
            [0.01],
            samples=200,
        )
