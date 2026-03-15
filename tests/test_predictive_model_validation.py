"""Integration tests for PredictiveModel.validate_predictions() and
PredictiveModel.identify_high_growth_low_rank().

Task 9.5 — Prediction validation and high-growth identification.
Validates: Requirements 8.3, 8.4
"""

import numpy as np
import pandas as pd
import pytest

from fortune500_kg.predictive_model import PredictiveModel, ValidationMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trained_model(n_samples: int = 120, n_features: int = 4, seed: int = 42):
    """Return a trained PredictiveModel with deterministic data."""
    rng = np.random.default_rng(seed)
    embeddings = rng.random((n_samples, n_features))
    metrics_df = pd.DataFrame(
        {"revenue_growth": rng.uniform(0.01, 0.30, n_samples)},
    )
    model = PredictiveModel()
    model.train(embeddings, metrics_df)
    return model


# ---------------------------------------------------------------------------
# validate_predictions() tests
# ---------------------------------------------------------------------------

class TestValidatePredictions:
    """Tests for validate_predictions() — Requirements 8.3, 8.4."""

    def test_returns_validation_metrics_type(self):
        """validate_predictions() should return a ValidationMetrics instance."""
        model = _make_trained_model()
        actual = pd.DataFrame({
            "company_id": model.company_ids[:20],
            "actual_revenue_growth": np.random.default_rng(0).uniform(0.01, 0.30, 20),
        })
        result = model.validate_predictions(actual)
        assert isinstance(result, ValidationMetrics)

    def test_sample_size_matches_overlap(self):
        """sample_size should equal the number of companies present in both
        the model and the actual_outcomes DataFrame."""
        model = _make_trained_model()
        n_overlap = 30
        actual = pd.DataFrame({
            "company_id": model.company_ids[:n_overlap],
            "actual_revenue_growth": np.random.default_rng(1).uniform(0.01, 0.30, n_overlap),
        })
        result = model.validate_predictions(actual)
        assert result.sample_size == n_overlap

    def test_accuracy_clamped_to_unit_interval(self):
        """accuracy must always be in [0, 1]."""
        model = _make_trained_model()
        # Use very different actuals to stress-test clamping
        actual = pd.DataFrame({
            "company_id": model.company_ids[:50],
            "actual_revenue_growth": np.ones(50) * 100.0,  # far from predictions
        })
        result = model.validate_predictions(actual)
        assert 0.0 <= result.accuracy <= 1.0

    def test_accuracy_formula(self):
        """accuracy = 1 - (MAE / mean_actual), clamped to [0, 1]."""
        model = _make_trained_model()
        n = 40
        actual = pd.DataFrame({
            "company_id": model.company_ids[:n],
            "actual_revenue_growth": np.random.default_rng(7).uniform(0.05, 0.25, n),
        })
        result = model.validate_predictions(actual)

        # Recompute expected accuracy manually
        expected_accuracy = max(0.0, min(1.0, 1.0 - result.mae / np.mean(
            np.abs(actual["actual_revenue_growth"].values)
        )))
        assert abs(result.accuracy - expected_accuracy) < 1e-9

    def test_mae_and_rmse_are_non_negative(self):
        """MAE and RMSE must be >= 0."""
        model = _make_trained_model()
        actual = pd.DataFrame({
            "company_id": model.company_ids[:25],
            "actual_revenue_growth": np.random.default_rng(3).uniform(0.01, 0.20, 25),
        })
        result = model.validate_predictions(actual)
        assert result.mae >= 0.0
        assert result.rmse >= 0.0

    def test_rmse_geq_mae(self):
        """RMSE >= MAE always (by Cauchy-Schwarz)."""
        model = _make_trained_model()
        actual = pd.DataFrame({
            "company_id": model.company_ids[:50],
            "actual_revenue_growth": np.random.default_rng(5).uniform(0.01, 0.30, 50),
        })
        result = model.validate_predictions(actual)
        assert result.rmse >= result.mae - 1e-9

    def test_empty_overlap_returns_zero_metrics(self):
        """When no company_ids overlap, all metrics should be zero."""
        model = _make_trained_model()
        actual = pd.DataFrame({
            "company_id": ["UNKNOWN_A", "UNKNOWN_B"],
            "actual_revenue_growth": [0.1, 0.2],
        })
        result = model.validate_predictions(actual)
        assert result.sample_size == 0
        assert result.accuracy == 0.0
        assert result.mae == 0.0
        assert result.rmse == 0.0

    def test_raises_before_training(self):
        """validate_predictions() must raise RuntimeError if model not trained."""
        model = PredictiveModel()
        actual = pd.DataFrame({
            "company_id": ["C1"],
            "actual_revenue_growth": [0.1],
        })
        with pytest.raises(RuntimeError):
            model.validate_predictions(actual)

    def test_raises_on_missing_columns(self):
        """validate_predictions() must raise ValueError for missing columns."""
        model = _make_trained_model()
        bad_df = pd.DataFrame({"company_id": model.company_ids[:5]})
        with pytest.raises(ValueError):
            model.validate_predictions(bad_df)

    def test_perfect_predictions_give_high_accuracy(self):
        """When predictions exactly match actuals, accuracy should be 1.0."""
        model = _make_trained_model()
        # Collect actual predictions from the model itself
        company_ids = model.company_ids[:30]
        predicted_growths = [
            model.predict_revenue_growth(cid).predicted_growth for cid in company_ids
        ]
        actual = pd.DataFrame({
            "company_id": company_ids,
            "actual_revenue_growth": predicted_growths,
        })
        result = model.validate_predictions(actual)
        # MAE should be ~0, so accuracy should be ~1.0
        assert result.accuracy == pytest.approx(1.0, abs=1e-6)
        assert result.mae == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# identify_high_growth_low_rank() tests
# ---------------------------------------------------------------------------

class TestIdentifyHighGrowthLowRank:
    """Tests for identify_high_growth_low_rank() — Requirement 8.3."""

    def test_returns_list(self):
        """identify_high_growth_low_rank() should return a list."""
        model = _make_trained_model()
        ranks = {cid: i + 1 for i, cid in enumerate(model.company_ids)}
        result = model.identify_high_growth_low_rank(ranks)
        assert isinstance(result, list)

    def test_all_results_have_high_growth(self):
        """Every returned company must have predicted growth > median growth."""
        model = _make_trained_model()
        ranks = {cid: i + 1 for i, cid in enumerate(model.company_ids)}
        candidates = model.identify_high_growth_low_rank(ranks)

        # Compute median growth across all companies
        all_growths = [
            model.predict_revenue_growth(cid).predicted_growth
            for cid in model.company_ids
        ]
        median_growth = float(np.median(all_growths))

        for cid in candidates:
            growth = model.predict_revenue_growth(cid).predicted_growth
            assert growth > median_growth, (
                f"{cid} has growth {growth:.4f} <= median {median_growth:.4f}"
            )

    def test_all_results_have_low_rank(self):
        """Every returned company must have rank > 75th-percentile rank value."""
        model = _make_trained_model()
        ranks = {cid: i + 1 for i, cid in enumerate(model.company_ids)}
        candidates = model.identify_high_growth_low_rank(ranks)

        rank_values = list(ranks.values())
        rank_threshold = float(np.percentile(rank_values, 75.0))

        for cid in candidates:
            assert ranks[cid] > rank_threshold, (
                f"{cid} has rank {ranks[cid]} <= threshold {rank_threshold}"
            )

    def test_no_false_negatives_for_clear_cases(self):
        """Companies clearly above median growth AND above 75th-pct rank must appear."""
        model = _make_trained_model()

        # Assign ranks so the first half of companies have very high ranks (low position)
        n = len(model.company_ids)
        ranks = {}
        for i, cid in enumerate(model.company_ids):
            # First half: rank 400-500 (high rank number = low position)
            # Second half: rank 1-50 (low rank number = top position)
            ranks[cid] = 400 + i if i < n // 2 else i - n // 2 + 1

        candidates = set(model.identify_high_growth_low_rank(ranks))

        # Compute median growth
        all_growths = {
            cid: model.predict_revenue_growth(cid).predicted_growth
            for cid in model.company_ids
        }
        median_growth = float(np.median(list(all_growths.values())))
        rank_threshold = float(np.percentile(list(ranks.values()), 75.0))

        for cid in model.company_ids:
            if all_growths[cid] > median_growth and ranks[cid] > rank_threshold:
                assert cid in candidates, (
                    f"{cid} should be a candidate but was not returned"
                )

    def test_empty_rank_dict_returns_empty(self):
        """Empty company_rank dict should return an empty list."""
        model = _make_trained_model()
        result = model.identify_high_growth_low_rank({})
        assert result == []

    def test_raises_before_training(self):
        """identify_high_growth_low_rank() must raise RuntimeError if not trained."""
        model = PredictiveModel()
        with pytest.raises(RuntimeError):
            model.identify_high_growth_low_rank({"C1": 1})

    def test_result_contains_only_known_companies(self):
        """All returned company_ids must be in the model's company_ids list."""
        model = _make_trained_model()
        ranks = {cid: i + 1 for i, cid in enumerate(model.company_ids)}
        candidates = model.identify_high_growth_low_rank(ranks)
        known = set(model.company_ids)
        for cid in candidates:
            assert cid in known
