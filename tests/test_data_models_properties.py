"""Property-based tests for core data models.

Feature: wikipedia-product-health-analysis
"""

import pytest
import pandas as pd
import pickle
import json
from datetime import date, datetime, timedelta
from hypothesis import given, settings, strategies as st
from wikipedia_health.models import (
    TimeSeriesData,
    TestResult,
    CausalEffect,
    ForecastResult,
    DecompositionResult,
    ValidationReport,
    Changepoint,
    Finding,
    Anomaly,
)


# Custom strategies for generating test data
@st.composite
def time_series_data_strategy(draw):
    """Generate random TimeSeriesData instances."""
    n_points = draw(st.integers(min_value=10, max_value=100))
    start_date = draw(st.dates(min_value=date(2015, 1, 1), max_value=date(2024, 12, 31)))
    
    dates = pd.Series(pd.date_range(start=start_date, periods=n_points, freq='D'))
    values = pd.Series(draw(st.lists(
        st.floats(min_value=0, max_value=1000000, allow_nan=False, allow_infinity=False),
        min_size=n_points,
        max_size=n_points
    )))
    
    platform = draw(st.sampled_from(['desktop', 'mobile-web', 'mobile-app', 'all']))
    metric_type = draw(st.sampled_from(['pageviews', 'editors', 'edits']))
    
    metadata = {
        'source': draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        'acquisition_date': datetime.now().isoformat(),
        'filters': draw(st.text(min_size=0, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P'))))
    }
    
    return TimeSeriesData(
        date=dates,
        values=values,
        platform=platform,
        metric_type=metric_type,
        metadata=metadata
    )


@settings(max_examples=100)
@given(ts_data=time_series_data_strategy())
def test_property_4_data_persistence_round_trip_pickle(ts_data):
    """
    Feature: wikipedia-product-health-analysis
    Property 4: Data Persistence Round-Trip
    
    For any acquired dataset, storing the data with metadata and then retrieving it
    should produce an equivalent dataset with all metadata intact (acquisition timestamp,
    source parameters, filters applied).
    
    This test validates round-trip serialization using pickle.
    """
    # Serialize
    serialized = pickle.dumps(ts_data)
    
    # Deserialize
    deserialized = pickle.loads(serialized)
    
    # Assert equivalence
    assert deserialized.platform == ts_data.platform
    assert deserialized.metric_type == ts_data.metric_type
    assert deserialized.metadata == ts_data.metadata
    
    # Check data integrity
    pd.testing.assert_series_equal(deserialized.date, ts_data.date)
    pd.testing.assert_series_equal(deserialized.values, ts_data.values)


@settings(max_examples=100)
@given(ts_data=time_series_data_strategy())
def test_property_4_data_persistence_round_trip_dataframe(ts_data):
    """
    Feature: wikipedia-product-health-analysis
    Property 4: Data Persistence Round-Trip (DataFrame conversion)
    
    Converting to DataFrame and back should preserve data integrity.
    """
    # Convert to DataFrame
    df = ts_data.to_dataframe()
    
    # Reconstruct from DataFrame
    reconstructed = TimeSeriesData(
        date=df['date'],
        values=df['values'],
        platform=df['platform'].iloc[0],
        metric_type=df['metric_type'].iloc[0],
        metadata=ts_data.metadata
    )
    
    # Assert equivalence
    assert reconstructed.platform == ts_data.platform
    assert reconstructed.metric_type == ts_data.metric_type
    assert reconstructed.metadata == ts_data.metadata
    
    # Check data integrity (reset index for comparison, check_names=False to ignore series names)
    pd.testing.assert_series_equal(
        reconstructed.date.reset_index(drop=True),
        ts_data.date.reset_index(drop=True),
        check_names=False
    )
    pd.testing.assert_series_equal(
        reconstructed.values.reset_index(drop=True),
        ts_data.values.reset_index(drop=True),
        check_names=False
    )


@st.composite
def _test_result_strategy(draw):
    """Generate random TestResult instances."""
    effect_size = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
    ci_lower = effect_size - draw(st.floats(min_value=0.1, max_value=2, allow_nan=False, allow_infinity=False))
    ci_upper = effect_size + draw(st.floats(min_value=0.1, max_value=2, allow_nan=False, allow_infinity=False))
    
    return TestResult(
        test_name=draw(st.sampled_from(['t-test', 'ANOVA', 'Mann-Whitney', 'Kruskal-Wallis'])),
        statistic=draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
        p_value=draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)),
        effect_size=effect_size,
        confidence_interval=(ci_lower, ci_upper),
        is_significant=draw(st.booleans()),
        alpha=draw(st.sampled_from([0.01, 0.05, 0.10])),
        interpretation=draw(st.text(min_size=10, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Zs'))))
    )


@settings(max_examples=100)
@given(test_result=_test_result_strategy())
def test_property_4_test_result_round_trip(test_result):
    """
    Feature: wikipedia-product-health-analysis
    Property 4: Data Persistence Round-Trip (TestResult)
    
    TestResult objects should survive serialization round-trip.
    """
    # Serialize via dict
    result_dict = test_result.to_dict()
    
    # Reconstruct
    reconstructed = TestResult(
        test_name=result_dict['test_name'],
        statistic=result_dict['statistic'],
        p_value=result_dict['p_value'],
        effect_size=result_dict['effect_size'],
        confidence_interval=result_dict['confidence_interval'],
        is_significant=result_dict['is_significant'],
        alpha=result_dict['alpha'],
        interpretation=result_dict['interpretation']
    )
    
    # Assert equivalence
    assert reconstructed.test_name == test_result.test_name
    assert reconstructed.statistic == test_result.statistic
    assert reconstructed.p_value == test_result.p_value
    assert reconstructed.effect_size == test_result.effect_size
    assert reconstructed.confidence_interval == test_result.confidence_interval
    assert reconstructed.is_significant == test_result.is_significant
    assert reconstructed.alpha == test_result.alpha
    assert reconstructed.interpretation == test_result.interpretation


@st.composite
def _changepoint_strategy(draw):
    """Generate random Changepoint instances."""
    start_date = draw(st.dates(min_value=date(2015, 1, 1), max_value=date(2024, 12, 31)))
    pre_mean = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    post_mean = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    
    return Changepoint(
        date=start_date,
        index=draw(st.integers(min_value=0, max_value=3650)),
        confidence=draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)),
        magnitude=abs(post_mean - pre_mean),
        direction='increase' if post_mean > pre_mean else 'decrease',
        pre_mean=pre_mean,
        post_mean=post_mean
    )


@settings(max_examples=100)
@given(changepoint=_changepoint_strategy())
def test_property_4_changepoint_round_trip(changepoint):
    """
    Feature: wikipedia-product-health-analysis
    Property 4: Data Persistence Round-Trip (Changepoint)
    
    Changepoint objects should survive serialization round-trip.
    """
    # Serialize using pickle
    serialized = pickle.dumps(changepoint)
    
    # Deserialize
    deserialized = pickle.loads(serialized)
    
    # Assert equivalence
    assert deserialized.date == changepoint.date
    assert deserialized.index == changepoint.index
    assert deserialized.confidence == changepoint.confidence
    assert deserialized.magnitude == changepoint.magnitude
    assert deserialized.direction == changepoint.direction
    assert deserialized.pre_mean == changepoint.pre_mean
    assert deserialized.post_mean == changepoint.post_mean


@st.composite
def _finding_strategy(draw):
    """Generate random Finding instances."""
    n_evidence = draw(st.integers(min_value=0, max_value=5))
    evidence = [draw(_test_result_strategy()) for _ in range(n_evidence)]
    
    return Finding(
        finding_id=draw(st.text(min_size=3, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Nd')))),
        description=draw(st.text(min_size=20, max_size=200, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Zs')))),
        evidence=evidence,
        causal_effects=[],
        confidence_level=draw(st.sampled_from(['low', 'medium', 'high'])),
        requirements_validated=draw(st.lists(st.text(min_size=3, max_size=10, alphabet=st.characters(whitelist_categories=('Nd', 'P'))), min_size=0, max_size=5))
    )


@settings(max_examples=100)
@given(finding=_finding_strategy())
def test_property_4_finding_round_trip(finding):
    """
    Feature: wikipedia-product-health-analysis
    Property 4: Data Persistence Round-Trip (Finding)
    
    Finding objects should survive serialization round-trip.
    """
    # Serialize using pickle
    serialized = pickle.dumps(finding)
    
    # Deserialize
    deserialized = pickle.loads(serialized)
    
    # Assert equivalence
    assert deserialized.finding_id == finding.finding_id
    assert deserialized.description == finding.description
    assert deserialized.confidence_level == finding.confidence_level
    assert deserialized.requirements_validated == finding.requirements_validated
    assert len(deserialized.evidence) == len(finding.evidence)
    
    # Check evidence equivalence
    for orig, deser in zip(finding.evidence, deserialized.evidence):
        assert deser.test_name == orig.test_name
        assert deser.p_value == orig.p_value
        assert deser.effect_size == orig.effect_size
