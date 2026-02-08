"""Data validation for quality checks and anomaly detection."""

from datetime import date, timedelta
from typing import Dict, List, Tuple
import logging

import pandas as pd
from pandas import DataFrame
import numpy as np

from wikipedia_health.models.data_models import ValidationReport, Anomaly
from wikipedia_health.config.config import load_config


logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for data quality checks and anomaly detection.
    
    Performs completeness checks, anomaly detection using z-score method,
    missing value flagging, and schema validation.
    """
    
    def __init__(self, config = None):
        """Initialize data validator with configuration.
        
        Args:
            config: Optional Config object. If None, loads default config
        """
        from wikipedia_health.config.config import Config
        
        if config is None:
            config = load_config()
        
        self.config = config
        self.validation_config = self.config.validation
        self.statistical_config = self.config.statistical
        
        self.max_missing_percentage = self.validation_config.max_missing_percentage
        self.max_gap_days = self.validation_config.max_gap_days
        self.outlier_threshold = self.statistical_config.outlier_threshold
    
    def check_completeness(
        self,
        data: DataFrame,
        expected_date_range: Tuple[date, date]
    ) -> ValidationReport:
        """Check data completeness for expected date range.
        
        Args:
            data: DataFrame with 'date' column
            expected_date_range: Tuple of (start_date, end_date)
            
        Returns:
            ValidationReport with completeness analysis
        """
        start_date, end_date = expected_date_range
        
        if data.empty:
            return ValidationReport(
                is_valid=False,
                completeness_score=0.0,
                missing_dates=[],
                anomalies=[],
                quality_metrics={'total_records': 0},
                recommendations=['No data available in the dataset']
            )
        
        # Ensure date column is datetime
        if 'date' not in data.columns:
            return ValidationReport(
                is_valid=False,
                completeness_score=0.0,
                missing_dates=[],
                anomalies=[],
                quality_metrics={},
                recommendations=['Data missing required "date" column']
            )
        
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # Generate expected date range
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        actual_dates = pd.to_datetime(data['date'].unique())
        
        # Find missing dates
        missing_dates = set(expected_dates) - set(actual_dates)
        missing_dates_list = sorted([d.date() for d in missing_dates])
        
        # Calculate completeness score
        total_expected = len(expected_dates)
        total_actual = len(actual_dates)
        completeness_score = total_actual / total_expected if total_expected > 0 else 0.0
        
        # Check if completeness meets threshold
        missing_percentage = 1.0 - completeness_score
        is_valid = missing_percentage <= self.max_missing_percentage
        
        # Quality metrics
        quality_metrics = {
            'total_expected_dates': total_expected,
            'total_actual_dates': total_actual,
            'missing_dates_count': len(missing_dates_list),
            'missing_percentage': missing_percentage,
            'total_records': len(data)
        }
        
        # Generate recommendations
        recommendations = []
        if not is_valid:
            recommendations.append(
                f"Data completeness ({completeness_score:.1%}) below threshold "
                f"({1 - self.max_missing_percentage:.1%})"
            )
        
        if len(missing_dates_list) > 0:
            # Check for large gaps
            if len(missing_dates_list) > 1:
                missing_dates_sorted = sorted(missing_dates_list)
                max_gap = 0
                for i in range(1, len(missing_dates_sorted)):
                    gap = (missing_dates_sorted[i] - missing_dates_sorted[i-1]).days
                    max_gap = max(max_gap, gap)
                
                if max_gap > self.max_gap_days:
                    recommendations.append(
                        f"Large gap detected: {max_gap} consecutive missing days "
                        f"(threshold: {self.max_gap_days})"
                    )
            
            if len(missing_dates_list) <= 10:
                recommendations.append(f"Missing dates: {missing_dates_list}")
            else:
                recommendations.append(
                    f"Missing {len(missing_dates_list)} dates. "
                    f"First: {missing_dates_list[0]}, Last: {missing_dates_list[-1]}"
                )
        
        logger.info(
            f"Completeness check: {completeness_score:.1%} "
            f"({total_actual}/{total_expected} dates present)"
        )
        
        return ValidationReport(
            is_valid=is_valid,
            completeness_score=completeness_score,
            missing_dates=missing_dates_list,
            anomalies=[],
            quality_metrics=quality_metrics,
            recommendations=recommendations
        )
    
    def detect_anomalies(
        self,
        data: DataFrame,
        value_column: str = 'values',
        threshold: float = None
    ) -> List[Anomaly]:
        """Detect anomalies using z-score method.
        
        Args:
            data: DataFrame with date and value columns
            value_column: Name of column containing values to check
            threshold: Z-score threshold (default from config)
            
        Returns:
            List of detected anomalies
        """
        if threshold is None:
            threshold = self.outlier_threshold
        
        if data.empty or value_column not in data.columns:
            logger.warning(f"Cannot detect anomalies: empty data or missing column '{value_column}'")
            return []
        
        data = data.copy()
        
        # Ensure date column exists
        if 'date' not in data.columns:
            logger.warning("Data missing 'date' column for anomaly detection")
            return []
        
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate z-scores
        values = data[value_column].values
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            logger.info("Zero standard deviation - no anomalies detected")
            return []
        
        z_scores = (values - mean) / std
        
        # Find anomalies
        anomalies = []
        for idx, z_score in enumerate(z_scores):
            if abs(z_score) > threshold:
                row = data.iloc[idx]
                anomaly_date = row['date'].date()
                anomaly_value = row[value_column]
                
                anomaly = Anomaly(
                    date=anomaly_date,
                    value=float(anomaly_value),
                    expected_value=float(mean),
                    z_score=float(z_score),
                    description=f"Value {anomaly_value:.2f} deviates {abs(z_score):.2f} "
                                f"standard deviations from mean {mean:.2f}"
                )
                anomalies.append(anomaly)
        
        logger.info(f"Detected {len(anomalies)} anomalies using z-score threshold {threshold}")
        
        return anomalies
    
    def flag_missing_values(self, data: DataFrame) -> DataFrame:
        """Flag rows with missing values.
        
        Args:
            data: DataFrame to check for missing values
            
        Returns:
            DataFrame with additional 'has_missing' boolean column
        """
        data = data.copy()
        data['has_missing'] = data.isnull().any(axis=1)
        
        missing_count = data['has_missing'].sum()
        total_count = len(data)
        
        if missing_count > 0:
            logger.warning(
                f"Found {missing_count} rows with missing values "
                f"({missing_count/total_count:.1%} of total)"
            )
        else:
            logger.info("No missing values detected")
        
        return data
    
    def validate_schema(
        self,
        data: DataFrame,
        expected_schema: Dict[str, type]
    ) -> bool:
        """Validate DataFrame schema against expected structure.
        
        Args:
            data: DataFrame to validate
            expected_schema: Dict mapping column names to expected types
            
        Returns:
            True if schema matches, False otherwise
        """
        if data.empty:
            logger.warning("Cannot validate schema of empty DataFrame")
            return False
        
        # Check for missing columns
        expected_columns = set(expected_schema.keys())
        actual_columns = set(data.columns)
        
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check column types
        type_mismatches = []
        for col, expected_type in expected_schema.items():
            actual_type = data[col].dtype
            
            # Handle pandas dtype mapping
            if expected_type == int and not pd.api.types.is_integer_dtype(actual_type):
                type_mismatches.append(f"{col}: expected int, got {actual_type}")
            elif expected_type == float and not pd.api.types.is_float_dtype(actual_type):
                type_mismatches.append(f"{col}: expected float, got {actual_type}")
            elif expected_type == str and not pd.api.types.is_string_dtype(actual_type) and actual_type != object:
                type_mismatches.append(f"{col}: expected str, got {actual_type}")
            elif expected_type == date and not pd.api.types.is_datetime64_any_dtype(actual_type):
                type_mismatches.append(f"{col}: expected datetime, got {actual_type}")
        
        if type_mismatches:
            logger.error(f"Schema type mismatches: {type_mismatches}")
            return False
        
        logger.info("Schema validation passed")
        return True
