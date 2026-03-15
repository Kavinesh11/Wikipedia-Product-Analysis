"""Property-based tests for metrics export functionality (Tasks 22.2, 22.4).

Properties:
- Property 81: CSV Export Structure Validation
- Property 82: JSON Export Validity
- Property 83: Tableau Integration Conditional Publishing
- Property 84: Power BI Export Format Compatibility
- Property 85: Export Metadata Inclusion

Validates: Requirements 17.1, 17.2, 17.3, 17.4, 17.5
"""

import csv
import io
import json
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import pytest

from fortune500_kg.metrics_exporter import MetricsExporter


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_company_id = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    min_size=1,
    max_size=8,
)

_metric_name = st.sampled_from([
    "innovation_score", "digital_maturity", "ecosystem_centrality",
    "revenue_growth", "pagerank",
])

_metric_value = st.floats(
    min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)


@st.composite
def company_metrics_dataset(draw, min_companies=1):
    """Generate a company_metrics dict."""
    n = draw(st.integers(min_value=min_companies, max_value=20))
    company_ids = draw(st.lists(_company_id, min_size=n, max_size=n, unique=True))
    metric_names = draw(st.lists(_metric_name, min_size=1, max_size=4, unique=True))
    return {
        cid: {m: draw(_metric_value) for m in metric_names}
        for cid in company_ids
    }


# ---------------------------------------------------------------------------
# Property 81: CSV Export Structure Validation
# Feature: fortune500-kg-analytics, Property 81
# Validates: Requirements 17.1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(company_metrics_dataset())
def test_property_81_csv_export_structure(company_metrics):
    """
    For any metrics export to CSV format, each row should contain a company
    identifier and all associated metric values with column headers.

    **Validates: Requirements 17.1**
    """
    exporter = MetricsExporter()
    csv_content = exporter.export_csv(company_metrics)

    # Filter out comment lines
    data_lines = [l for l in csv_content.splitlines() if not l.startswith("#")]
    assume(len(data_lines) >= 2)  # At least header + 1 data row

    reader = csv.DictReader(data_lines)
    rows = list(reader)

    # Must have one row per company
    assert len(rows) == len(company_metrics)

    # Each row must have company_id
    for row in rows:
        assert "company_id" in row
        assert row["company_id"] in company_metrics

    # All metric names must appear as columns
    all_metrics = set(
        m for metrics in company_metrics.values() for m in metrics.keys()
    )
    if rows:
        for metric in all_metrics:
            assert metric in rows[0], f"Metric '{metric}' missing from CSV header"


# ---------------------------------------------------------------------------
# Property 82: JSON Export Validity
# Feature: fortune500-kg-analytics, Property 82
# Validates: Requirements 17.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(company_metrics_dataset())
def test_property_82_json_export_validity(company_metrics):
    """
    For any metrics export to JSON format, the exported file should be valid
    JSON and parseable by standard JSON libraries.

    **Validates: Requirements 17.2**
    """
    exporter = MetricsExporter()
    json_content = exporter.export_json(company_metrics)

    # Must be parseable as JSON
    parsed = json.loads(json_content)
    assert isinstance(parsed, dict)

    # Must have data section
    assert "data" in parsed
    assert isinstance(parsed["data"], list)
    assert len(parsed["data"]) == len(company_metrics)

    # Each entry must have company_id
    for entry in parsed["data"]:
        assert "company_id" in entry
        assert entry["company_id"] in company_metrics


# ---------------------------------------------------------------------------
# Property 85: Export Metadata Inclusion
# Feature: fortune500-kg-analytics, Property 85
# Validates: Requirements 17.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(company_metrics_dataset())
def test_property_85_export_metadata_inclusion_json(company_metrics):
    """
    For any metrics export, the exported data should include metadata
    describing metric definitions and the timestamp when metrics were calculated.

    **Validates: Requirements 17.5**
    """
    metric_defs = {"innovation_score": "Stars + forks / employees"}
    exporter = MetricsExporter()
    json_content = exporter.export_json(company_metrics, metric_definitions=metric_defs)

    parsed = json.loads(json_content)
    assert "metadata" in parsed
    metadata = parsed["metadata"]

    # Must have export timestamp
    assert "export_timestamp" in metadata
    assert metadata["export_timestamp"]  # Non-empty

    # Must have metric definitions
    assert "metric_definitions" in metadata


@settings(max_examples=100)
@given(company_metrics_dataset())
def test_property_85_export_metadata_inclusion_csv(company_metrics):
    """
    CSV export should also include metadata (as comment lines).

    **Validates: Requirements 17.5**
    """
    exporter = MetricsExporter()
    csv_content = exporter.export_csv(company_metrics)

    # Metadata should appear as comment lines
    comment_lines = [l for l in csv_content.splitlines() if l.startswith("#")]
    assert len(comment_lines) >= 1

    # Must contain export timestamp
    combined = "\n".join(comment_lines)
    assert "export_timestamp" in combined


# ---------------------------------------------------------------------------
# Property 83: Tableau Integration Conditional Publishing
# Feature: fortune500-kg-analytics, Property 83
# Validates: Requirements 17.3
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(company_metrics_dataset())
def test_property_83_tableau_conditional_publishing_skipped(company_metrics):
    """
    When Tableau integration is NOT configured, publishing should be skipped.

    **Validates: Requirements 17.3**
    """
    exporter = MetricsExporter(tableau_config=None)
    result = exporter.publish_to_tableau(company_metrics)
    assert result["status"] == "skipped"


@settings(max_examples=50)
@given(company_metrics_dataset())
def test_property_83_tableau_conditional_publishing_active(company_metrics):
    """
    When Tableau integration IS configured, publishing should be attempted.

    **Validates: Requirements 17.3**
    """
    exporter = MetricsExporter(tableau_config={"server_url": "https://tableau.example.com"})
    result = exporter.publish_to_tableau(company_metrics)
    assert result["status"] in ("published", "failed")


# ---------------------------------------------------------------------------
# Property 84: Power BI Export Format Compatibility
# Feature: fortune500-kg-analytics, Property 84
# Validates: Requirements 17.4
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(company_metrics_dataset())
def test_property_84_power_bi_format_skipped_when_not_configured(company_metrics):
    """
    When Power BI integration is NOT configured, export should be skipped.

    **Validates: Requirements 17.4**
    """
    exporter = MetricsExporter(power_bi_config=None)
    result = exporter.export_to_power_bi(company_metrics)
    assert result["status"] == "skipped"


@settings(max_examples=50)
@given(company_metrics_dataset())
def test_property_84_power_bi_format_compatibility(company_metrics):
    """
    When Power BI integration IS configured, the exported format should be
    compatible with Power BI data import requirements (columns + rows structure).

    **Validates: Requirements 17.4**
    """
    exporter = MetricsExporter(power_bi_config={"workspace_id": "ws-123"})
    result = exporter.export_to_power_bi(company_metrics)

    assert result["status"] == "exported"
    data = result["data"]

    # Power BI compatible format requires columns and rows
    assert "columns" in data
    assert "rows" in data
    assert isinstance(data["columns"], list)
    assert isinstance(data["rows"], list)

    # company_id must be a column
    assert "company_id" in data["columns"]

    # Row count must match company count
    assert len(data["rows"]) == len(company_metrics)

    # Each row must have same number of values as columns
    for row in data["rows"]:
        assert len(row) == len(data["columns"])
