"""Predictive Model for Fortune 500 Knowledge Graph Analytics.

Implements:
- PredictiveModel class with train() method (Requirements 8.1)
- Revenue growth prediction (Requirements 8.2, 8.5)
- Prediction validation (Requirements 8.3, 8.4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fortune500_kg.data_models import PredictionRecord

logger = logging.getLogger(__name__)

# Minimum number of training samples required before fitting the model.
MIN_TRAINING_SAMPLES = 100


@dataclass
class PredictionResult:
    """Result of a single revenue-growth prediction."""

    company_id: str
    predicted_growth: float
    confidence_score: float  # 0.0 – 1.0
    is_high_confidence: bool  # True when confidence_score > 0.80
    prediction_date: datetime = field(default_factory=datetime.now)
    target_date: Optional[datetime] = None


@dataclass
class ValidationMetrics:
    """Metrics produced by comparing predictions to actual outcomes."""

    accuracy: float  # 1 - (MAE / mean_actual)
    rmse: float
    mae: float
    sample_size: int


class PredictiveModel:
    """
    ML model that trains on graph embeddings and historical metrics to predict
    revenue growth for Fortune 500 companies.

    Usage::

        model = PredictiveModel()
        model.train(embeddings, historical_metrics)
        result = model.predict_revenue_growth("COMP001")

    Attributes:
        model: The fitted scikit-learn estimator (set after train() is called).
        company_ids: Ordered list of company IDs seen during training.
        feature_matrix: Combined feature matrix used for training (embeddings +
            historical metric features), stored to support prediction look-ups.
        target_values: Revenue-growth target values used during training.
    """

    def __init__(self) -> None:
        self.model = None  # fitted sklearn estimator
        self.company_ids: List[str] = []
        self.feature_matrix: Optional[np.ndarray] = None
        self.target_values: Optional[np.ndarray] = None
        self._company_index: Dict[str, int] = {}  # company_id -> row index

    # ------------------------------------------------------------------
    # Training (Requirement 8.1)
    # ------------------------------------------------------------------

    def train(
        self,
        embeddings: np.ndarray,
        historical_metrics: pd.DataFrame,
    ) -> None:
        """
        Train ML model on graph embeddings and historical metrics.

        The method combines the embedding vectors with numeric columns from
        *historical_metrics* into a single feature matrix, then fits a
        GradientBoostingRegressor (with RandomForestRegressor as fallback on
        convergence failure) to predict the ``revenue_growth`` column.

        Args:
            embeddings: 2-D numpy array of shape (N, embedding_dim) containing
                graph embeddings from Neo4j GDS.  Row *i* corresponds to the
                company identified by ``historical_metrics.index[i]`` (or the
                ``company_id`` column when present).
            historical_metrics: DataFrame with at least a ``revenue_growth``
                column used as the prediction target.  All other numeric columns
                are used as additional features.  The index (or ``company_id``
                column) identifies each company.

        Raises:
            ValueError: When the number of training samples N < 100.
            ValueError: When embeddings and historical_metrics have different
                numbers of rows.
            ValueError: When ``revenue_growth`` column is missing from
                historical_metrics.

        Validates: Requirement 8.1
        """
        # ---- Validate inputs ------------------------------------------------
        n_samples = embeddings.shape[0]

        if n_samples != len(historical_metrics):
            raise ValueError(
                f"embeddings has {n_samples} rows but historical_metrics has "
                f"{len(historical_metrics)} rows – they must match."
            )

        if "revenue_growth" not in historical_metrics.columns:
            raise ValueError(
                "historical_metrics must contain a 'revenue_growth' column "
                "to use as the prediction target."
            )

        # Minimum sample size check (must happen before any training)
        if n_samples < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Insufficient training data: {n_samples} samples provided, "
                f"but a minimum of {MIN_TRAINING_SAMPLES} samples is required. "
                "Please provide more historical data before training."
            )

        # ---- Extract company IDs -------------------------------------------
        if "company_id" in historical_metrics.columns:
            company_ids = list(historical_metrics["company_id"].astype(str))
        else:
            company_ids = [str(idx) for idx in historical_metrics.index]

        # ---- Build feature matrix ------------------------------------------
        # Use all numeric columns except the target as additional features.
        exclude_cols = {"revenue_growth", "company_id"}
        numeric_cols = [
            c
            for c in historical_metrics.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

        if numeric_cols:
            metric_features = historical_metrics[numeric_cols].to_numpy(dtype=float)
            X = np.hstack([embeddings, metric_features])
        else:
            X = embeddings.copy()

        y = historical_metrics["revenue_growth"].to_numpy(dtype=float)

        # ---- Fit model ------------------------------------------------------
        fitted_model = self._fit_model(X, y)

        # ---- Persist training artefacts ------------------------------------
        self.model = fitted_model
        self.company_ids = company_ids
        self.feature_matrix = X
        self.target_values = y
        self._company_index = {cid: i for i, cid in enumerate(company_ids)}

        logger.info(
            "PredictiveModel trained on %d samples with %d features.",
            n_samples,
            X.shape[1],
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """
        Fit a GradientBoostingRegressor; fall back to RandomForestRegressor on
        convergence failure.

        Args:
            X: Feature matrix (N, F).
            y: Target vector (N,).

        Returns:
            A fitted scikit-learn estimator.
        """
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

        try:
            gbr = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
            )
            gbr.fit(X, y)
            logger.debug("GradientBoostingRegressor fitted successfully.")
            return gbr
        except Exception as exc:  # pragma: no cover – fallback path
            logger.warning(
                "GradientBoostingRegressor failed (%s); falling back to "
                "RandomForestRegressor.",
                exc,
            )
            rfr = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
            )
            rfr.fit(X, y)
            logger.debug("RandomForestRegressor fitted as fallback.")
            return rfr

    # ------------------------------------------------------------------
    # Prediction (Requirements 8.2, 8.5)
    # ------------------------------------------------------------------

    def predict_revenue_growth(self, company_id: str) -> PredictionResult:
        """
        Predict next fiscal year revenue growth for a single company.

        The company must have been present in the training dataset.

        Args:
            company_id: Unique identifier for the company.

        Returns:
            PredictionResult with predicted_growth and confidence_score.

        Raises:
            RuntimeError: When the model has not been trained yet.
            KeyError: When company_id was not seen during training.

        Validates: Requirements 8.2, 8.5
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been trained. Call train() before predict_revenue_growth()."
            )

        if company_id not in self._company_index:
            raise KeyError(
                f"company_id '{company_id}' was not present in the training dataset."
            )

        idx = self._company_index[company_id]
        x = self.feature_matrix[idx : idx + 1]  # shape (1, F)

        predicted_growth = float(self.model.predict(x)[0])
        confidence_score = self._compute_confidence(x)
        is_high_confidence = confidence_score > 0.80

        return PredictionResult(
            company_id=company_id,
            predicted_growth=predicted_growth,
            confidence_score=confidence_score,
            is_high_confidence=is_high_confidence,
        )

    def _compute_confidence(self, x: np.ndarray) -> float:
        """
        Estimate prediction confidence using tree-based variance.

        For ensemble models that expose individual estimators (GBR / RF), we
        compute the standard deviation of per-tree predictions and convert it
        to a confidence score in [0, 1] via an exponential decay:

            confidence = exp(-std / scale)

        where *scale* is the standard deviation of the training targets.

        Falls back to a fixed confidence of 0.5 when individual estimators are
        not available.

        Args:
            x: Feature row, shape (1, F).

        Returns:
            Confidence score in [0.0, 1.0].
        """
        scores = self._compute_confidence_batch(x)
        return float(scores[0])

    def _compute_confidence_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch confidence computation for multiple rows.

        Args:
            X: Feature matrix, shape (N, F).

        Returns:
            Confidence scores array of shape (N,) with values in [0.0, 1.0].
        """
        import math

        n = X.shape[0]
        estimators = getattr(self.model, "estimators_", None)
        if estimators is None:
            return np.full(n, 0.5)

        # GradientBoostingRegressor: estimators_ is a 2-D array of
        # DecisionTreeRegressor objects; we need to flatten it.
        if hasattr(estimators[0], "__len__"):
            # GBR: shape (n_estimators, 1)
            flat_estimators = [est[0] for est in estimators]
        else:
            # RF: shape (n_estimators,)
            flat_estimators = list(estimators)

        # Batch predict: shape (n_estimators, N)
        all_preds = np.array([est.predict(X) for est in flat_estimators])
        # std across estimators for each sample: shape (N,)
        stds = np.std(all_preds, axis=0)

        # Scale by training target std to make the metric relative.
        target_std = float(np.std(self.target_values)) if self.target_values is not None else 1.0
        scale = max(target_std, 1e-8)

        confidences = np.exp(-stds / scale)
        return np.clip(confidences, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Validation (Requirements 8.3, 8.4)
    # ------------------------------------------------------------------

    def validate_predictions(
        self, actual_outcomes: pd.DataFrame
    ) -> ValidationMetrics:
        """
        Compare predictions to actual outcomes and compute accuracy metrics.

        Args:
            actual_outcomes: DataFrame with columns ``company_id`` and
                ``actual_revenue_growth``.

        Returns:
            ValidationMetrics with accuracy, RMSE, MAE, and sample_size.

        Raises:
            RuntimeError: When the model has not been trained yet.
            ValueError: When required columns are missing.

        Validates: Requirements 8.3, 8.4
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been trained. Call train() before validate_predictions()."
            )

        required = {"company_id", "actual_revenue_growth"}
        missing = required - set(actual_outcomes.columns)
        if missing:
            raise ValueError(
                f"actual_outcomes is missing required columns: {missing}"
            )

        predictions: List[float] = []
        actuals: List[float] = []

        for _, row in actual_outcomes.iterrows():
            cid = str(row["company_id"])
            if cid not in self._company_index:
                continue
            try:
                result = self.predict_revenue_growth(cid)
                predictions.append(result.predicted_growth)
                actuals.append(float(row["actual_revenue_growth"]))
            except (KeyError, RuntimeError):
                continue

        if not predictions:
            return ValidationMetrics(accuracy=0.0, rmse=0.0, mae=0.0, sample_size=0)

        n = len(predictions)
        pred_arr = np.array(predictions)
        actual_arr = np.array(actuals)

        errors = pred_arr - actual_arr
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mean_actual = float(np.mean(np.abs(actual_arr)))

        # accuracy = 1 - (MAE / mean_actual), clamped to [0, 1]
        if mean_actual > 0:
            accuracy = float(np.clip(1.0 - mae / mean_actual, 0.0, 1.0))
        else:
            accuracy = 0.0

        logger.info(
            "Prediction validation: n=%d accuracy=%.4f RMSE=%.4f MAE=%.4f",
            n,
            accuracy,
            rmse,
            mae,
        )
        return ValidationMetrics(accuracy=accuracy, rmse=rmse, mae=mae, sample_size=n)

    # ------------------------------------------------------------------
    # High-growth / low-rank identification (Requirement 8.3)
    # ------------------------------------------------------------------

    def identify_high_growth_low_rank(
        self,
        company_rank: Dict[str, int],
        percentile_threshold: float = 75.0,
    ) -> List[str]:
        """
        Identify companies with high predicted growth that currently rank below
        the top quartile (i.e. rank > 75th percentile of rank values).

        A company qualifies when:
        - Its predicted revenue growth is above the median predicted growth.
        - Its current Fortune 500 rank is below the 75th percentile rank
          (i.e. it is NOT already in the top quartile by rank).

        Args:
            company_rank: Mapping of company_id -> Fortune 500 rank (lower = better).
            percentile_threshold: Rank percentile above which a company is
                considered "low rank" (default 75 → top-quartile boundary).

        Returns:
            List of company_ids that meet both criteria.

        Raises:
            RuntimeError: When the model has not been trained yet.

        Validates: Requirement 8.3
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been trained. Call train() before identify_high_growth_low_rank()."
            )

        # Generate predictions for all known companies
        all_predictions: Dict[str, float] = {}
        for cid in self.company_ids:
            try:
                result = self.predict_revenue_growth(cid)
                all_predictions[cid] = result.predicted_growth
            except (KeyError, RuntimeError):
                continue

        if not all_predictions:
            return []

        growth_values = list(all_predictions.values())
        median_growth = float(np.median(growth_values))

        # Determine rank threshold
        rank_values = [
            company_rank[cid] for cid in all_predictions if cid in company_rank
        ]
        if not rank_values:
            return []

        rank_threshold = float(np.percentile(rank_values, percentile_threshold))

        candidates = [
            cid
            for cid, growth in all_predictions.items()
            if growth > median_growth
            and cid in company_rank
            and company_rank[cid] > rank_threshold
        ]

        logger.info(
            "Identified %d high-growth low-rank companies "
            "(median_growth=%.4f, rank_threshold=%.1f).",
            len(candidates),
            median_growth,
            rank_threshold,
        )
        return candidates

    # ------------------------------------------------------------------
    # Bulk prediction (Requirements 8.2, 8.5)
    # ------------------------------------------------------------------

    def predict_all(self) -> List[PredictionRecord]:
        """
        Generate revenue-growth predictions for ALL companies in the training
        dataset and return them as PredictionRecord objects.

        Returns:
            List of PredictionRecord objects (one per company in
            ``self.company_ids``) with ``prediction_type='revenue_growth'``,
            ``actual_value=None``, ``prediction_date`` set to now, and
            ``target_date`` set to one year from now.

        Raises:
            RuntimeError: When the model has not been trained yet.

        Validates: Requirements 8.2, 8.5
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been trained. Call train() before predict_all()."
            )

        now = datetime.now()
        target = now + timedelta(days=365)

        # Batch predict for all companies at once
        X = self.feature_matrix  # shape (N, F)
        predicted_growths = self.model.predict(X)  # shape (N,)
        confidence_scores = self._compute_confidence_batch(X)  # shape (N,)

        records: List[PredictionRecord] = []
        for i, company_id in enumerate(self.company_ids):
            records.append(
                PredictionRecord(
                    company_id=company_id,
                    prediction_type="revenue_growth",
                    predicted_value=float(predicted_growths[i]),
                    confidence_score=float(confidence_scores[i]),
                    prediction_date=now,
                    target_date=target,
                    actual_value=None,
                )
            )

        logger.info(
            "predict_all() generated %d PredictionRecord objects.", len(records)
        )
        return records

    def get_high_confidence_predictions(self) -> List[PredictionRecord]:
        """
        Return only the high-confidence predictions (confidence_score > 0.80).

        Calls predict_all() internally and filters the results.

        Returns:
            List of PredictionRecord objects where confidence_score > 0.80.

        Raises:
            RuntimeError: When the model has not been trained yet.

        Validates: Requirements 8.5
        """
        all_records = self.predict_all()
        high_confidence = [r for r in all_records if r.confidence_score > 0.80]
        logger.info(
            "get_high_confidence_predictions() returned %d / %d records.",
            len(high_confidence),
            len(all_records),
        )
        return high_confidence

