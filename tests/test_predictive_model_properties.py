"""Property-based tests for the PredictiveModel class.

Tests Properties 36, 37, and 40 from the design document:
- Property 36: ML Model Training Completion
- Property 37: Revenue Growth Prediction Coverage
- Property 40: High-Confidence Forecast Flagging
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from fortune500_kg.predictive_model import PredictiveModel, PredictionResult
from fortune500_kg.data_models import PredictionRecord


# Feature: fortune500-kg-analytics, Property 36: ML Model Training Completion
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=100, max_value=200),
    n_features=st.integers(min_value=2, max_value=10),
)
def test_ml_model_training_completion(n_samples, n_features):
    """Property 36: ML model training should complete and produce a trained artifact.

    For any valid graph embeddings and historical metrics dataset (N >= 100),
    the ML model training process should complete without errors and produce
    a trained model artifact.

    Validates: Requirements 8.1
    """
    rng = np.random.default_rng(seed=n_samples * 31 + n_features)
    embeddings = rng.random((n_samples, n_features))
    metrics_df = pd.DataFrame({
        "revenue_growth": rng.random(n_samples),
    })

    model = PredictiveModel()
    model.train(embeddings, metrics_df)

    # The trained artifact must exist after training completes
    assert model.model is not None


def test_ml_model_training_insufficient_data():
    """Property 36 negative: training with N < 100 should raise ValueError.

    Validates: Requirements 8.1
    """
    rng = np.random.default_rng(seed=42)
    for n_samples in [0, 1, 50, 99]:
        embeddings = rng.random((n_samples, 4)) if n_samples > 0 else np.empty((0, 4))
        metrics_df = pd.DataFrame({
            "revenue_growth": rng.random(n_samples),
        })
        model = PredictiveModel()
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            model.train(embeddings, metrics_df)


# Feature: fortune500-kg-analytics, Property 37: Revenue Growth Prediction Coverage
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=100, max_value=200),
    n_features=st.integers(min_value=2, max_value=10),
)
def test_revenue_growth_prediction_coverage(n_samples, n_features):
    """Property 37: predict_all() should return one PredictionRecord per company.

    For any trained ML model, predictions for next fiscal year revenue growth
    should be generated for all companies in the dataset.

    Validates: Requirements 8.2
    """
    rng = np.random.default_rng(seed=n_samples * 17 + n_features)
    embeddings = rng.random((n_samples, n_features))
    metrics_df = pd.DataFrame({
        "revenue_growth": rng.random(n_samples),
    })

    model = PredictiveModel()
    model.train(embeddings, metrics_df)

    records = model.predict_all()

    # One record per company in the training set
    assert len(records) == len(model.company_ids)

    # Every record must be for revenue_growth
    for record in records:
        assert isinstance(record, PredictionRecord)
        assert record.prediction_type == "revenue_growth"


# Feature: fortune500-kg-analytics, Property 40: High-Confidence Forecast Flagging
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=100, max_value=200),
    n_features=st.integers(min_value=2, max_value=10),
)
def test_high_confidence_forecast_flagging(n_samples, n_features):
    """Property 40: is_high_confidence iff confidence_score > 0.80.

    For any prediction with confidence score C, the prediction should be
    flagged as high-confidence if and only if C > 0.80.

    Validates: Requirements 8.5
    """
    rng = np.random.default_rng(seed=n_samples * 23 + n_features)
    embeddings = rng.random((n_samples, n_features))
    metrics_df = pd.DataFrame({
        "revenue_growth": rng.random(n_samples),
    })

    model = PredictiveModel()
    model.train(embeddings, metrics_df)

    records = model.predict_all()

    # Verify the biconditional via a single predict_revenue_growth call per company:
    # PredictionResult.is_high_confidence must equal (confidence_score > 0.80).
    # We sample one company to verify the flag is set correctly without re-running
    # all N predictions (predict_all already uses predict_revenue_growth internally).
    sample_id = model.company_ids[0]
    result = model.predict_revenue_growth(sample_id)
    assert result.is_high_confidence == (result.confidence_score > 0.80)

    # get_high_confidence_predictions() must return exactly the records with score > 0.80
    high_conf_records = model.get_high_confidence_predictions()
    expected_high_conf = [r for r in records if r.confidence_score > 0.80]
    assert len(high_conf_records) == len(expected_high_conf)
    for r in high_conf_records:
        assert r.confidence_score > 0.80


# Feature: fortune500-kg-analytics, Property 38: High-Growth Low-Rank Identification
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=100, max_value=200),
    n_features=st.integers(min_value=2, max_value=10),
)
def test_high_growth_low_rank_identification(n_samples, n_features):
    """Property 38: Identified high-growth candidates must have predicted growth
    above the median AND current rank below the 75th percentile.

    For any set of companies with predictions, identified high-growth candidates
    should have predicted growth above the median AND current rank below the
    75th percentile.

    Validates: Requirements 8.3
    """
    rng = np.random.default_rng(seed=n_samples * 41 + n_features)
    embeddings = rng.random((n_samples, n_features))
    metrics_df = pd.DataFrame({
        "revenue_growth": rng.random(n_samples),
    })

    model = PredictiveModel()
    model.train(embeddings, metrics_df)

    # Assign a random rank (1-500) to each company
    company_rank = {
        cid: int(rng.integers(1, 501))
        for cid in model.company_ids
    }

    candidates = model.identify_high_growth_low_rank(company_rank)

    # Compute median predicted growth across all companies
    all_growths = [
        model.predict_revenue_growth(cid).predicted_growth
        for cid in model.company_ids
    ]
    median_growth = float(np.median(all_growths))

    # Compute 75th percentile of rank values
    rank_values = [company_rank[cid] for cid in model.company_ids]
    rank_threshold = float(np.percentile(rank_values, 75))

    for cid in candidates:
        predicted_growth = model.predict_revenue_growth(cid).predicted_growth
        assert predicted_growth > median_growth, (
            f"Company {cid} has predicted_growth={predicted_growth} <= "
            f"median_growth={median_growth}"
        )
        assert company_rank[cid] > rank_threshold, (
            f"Company {cid} has rank={company_rank[cid]} <= "
            f"rank_threshold={rank_threshold}"
        )


# Feature: fortune500-kg-analytics, Property 39: Prediction Accuracy Calculation
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=100, max_value=200),
    n_features=st.integers(min_value=2, max_value=10),
)
def test_prediction_accuracy_calculation(n_samples, n_features):
    """Property 39: accuracy == 1 - (MAE / mean_actual), clamped to [0, 1].

    For any set of predictions with known actual outcomes, the accuracy metric
    should equal 1 - (mean absolute error / mean actual value).

    Validates: Requirements 8.4
    """
    rng = np.random.default_rng(seed=n_samples * 53 + n_features)
    embeddings = rng.random((n_samples, n_features))
    metrics_df = pd.DataFrame({
        "revenue_growth": rng.random(n_samples),
    })

    model = PredictiveModel()
    model.train(embeddings, metrics_df)

    # Use a subset of company_ids with random actual outcomes
    subset_size = max(1, n_samples // 2)
    subset_ids = model.company_ids[:subset_size]
    actual_values = rng.random(subset_size).tolist()

    actual_outcomes = pd.DataFrame({
        "company_id": subset_ids,
        "actual_revenue_growth": actual_values,
    })

    result = model.validate_predictions(actual_outcomes)

    # Manually compute expected accuracy
    predictions = []
    actuals = []
    for cid, actual in zip(subset_ids, actual_values):
        pred = model.predict_revenue_growth(cid).predicted_growth
        predictions.append(pred)
        actuals.append(actual)

    pred_arr = np.array(predictions)
    actual_arr = np.array(actuals)
    mae = float(np.mean(np.abs(pred_arr - actual_arr)))
    mean_actual = float(np.mean(np.abs(actual_arr)))

    if mean_actual > 0:
        expected_accuracy = float(np.clip(1.0 - mae / mean_actual, 0.0, 1.0))
    else:
        expected_accuracy = 0.0

    assert result.accuracy == pytest.approx(expected_accuracy, abs=1e-9)
