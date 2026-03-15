"""Unit tests for metrics export functionality (Tasks 22.1, 22.3).

Covers:
- MetricsExporter.export_csv (Requirements 17.1, 17.5)
- MetricsExporter.export_json (Requirements 17.2, 17.5)
- MetricsExporter.publish_to_tableau (Requirement 17.3)
- MetricsExporter.export_to_power_bi (Requirement 17.4)
"""

import csv
import io
import json
import pytest

from fortune500_kg.metrics_exporter import MetricsExporter


SAMPLE_METRICS = {
    "C1": {"innovation_score": 9.0, "digital_maturity": 7.5},
    "C2": {"innovation_score": 6.0, "digital_maturity": 5.0},
    "C3": {"innovation_score": 3.0, "digital_maturity": 2.5},
}

METRIC_DEFS = {
    "innovation_score": "Normalized (stars + forks) / employee_count",
    "digital_maturity": "(stars + forks + contributors) / revenue_rank",
}


@pytest.fixture
def exporter():
    return MetricsExporter()


@pytest.fixture
def tableau_exporter():
    return MetricsExporter(tableau_config={"server_url": "https://tableau.example.com"})


@pytest.fixture
def power_bi_exporter():
    return MetricsExporter(power_bi_config={"workspace_id": "ws-123"})


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------

class TestExportCsv:

    def test_returns_non_empty_string(self, exporter):
        result = exporter.export_csv(SAMPLE_METRICS)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_has_header_row(self, exporter):
        result = exporter.export_csv(SAMPLE_METRICS)
        lines = [l for l in result.splitlines() if not l.startswith("#")]
        assert lines[0].startswith("company_id")

    def test_all_companies_present(self, exporter):
        result = exporter.export_csv(SAMPLE_METRICS)
        for company_id in SAMPLE_METRICS:
            assert company_id in result

    def test_all_metrics_in_header(self, exporter):
        result = exporter.export_csv(SAMPLE_METRICS)
        lines = [l for l in result.splitlines() if not l.startswith("#")]
        header = lines[0]
        assert "innovation_score" in header
        assert "digital_maturity" in header

    def test_metadata_included_as_comments(self, exporter):
        result = exporter.export_csv(SAMPLE_METRICS, metric_definitions=METRIC_DEFS)
        assert "export_timestamp" in result
        assert "innovation_score" in result

    def test_empty_metrics_returns_empty_string(self, exporter):
        result = exporter.export_csv({})
        assert result == ""

    def test_parseable_as_csv(self, exporter):
        result = exporter.export_csv(SAMPLE_METRICS)
        # Filter out comment lines
        data_lines = [l for l in result.splitlines() if not l.startswith("#")]
        reader = csv.DictReader(data_lines)
        rows = list(reader)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:

    def test_returns_valid_json(self, exporter):
        result = exporter.export_json(SAMPLE_METRICS)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_has_metadata_section(self, exporter):
        result = exporter.export_json(SAMPLE_METRICS, metric_definitions=METRIC_DEFS)
        parsed = json.loads(result)
        assert "metadata" in parsed
        assert "export_timestamp" in parsed["metadata"]

    def test_has_data_section(self, exporter):
        result = exporter.export_json(SAMPLE_METRICS)
        parsed = json.loads(result)
        assert "data" in parsed
        assert len(parsed["data"]) == 3

    def test_each_entry_has_company_id(self, exporter):
        result = exporter.export_json(SAMPLE_METRICS)
        parsed = json.loads(result)
        for entry in parsed["data"]:
            assert "company_id" in entry

    def test_metric_definitions_in_metadata(self, exporter):
        result = exporter.export_json(SAMPLE_METRICS, metric_definitions=METRIC_DEFS)
        parsed = json.loads(result)
        assert "metric_definitions" in parsed["metadata"]


# ---------------------------------------------------------------------------
# publish_to_tableau
# ---------------------------------------------------------------------------

class TestPublishToTableau:

    def test_skipped_when_not_configured(self, exporter):
        result = exporter.publish_to_tableau(SAMPLE_METRICS)
        assert result["status"] == "skipped"

    def test_published_when_configured(self, tableau_exporter):
        result = tableau_exporter.publish_to_tableau(SAMPLE_METRICS)
        assert result["status"] == "published"

    def test_published_with_client(self, tableau_exporter):
        class MockClient:
            def publish(self, metrics):
                return {"rows_published": len(metrics)}

        result = tableau_exporter.publish_to_tableau(SAMPLE_METRICS, tableau_client=MockClient())
        assert result["status"] == "published"
        assert "published_at" in result


# ---------------------------------------------------------------------------
# export_to_power_bi
# ---------------------------------------------------------------------------

class TestExportToPowerBi:

    def test_skipped_when_not_configured(self, exporter):
        result = exporter.export_to_power_bi(SAMPLE_METRICS)
        assert result["status"] == "skipped"

    def test_exported_when_configured(self, power_bi_exporter):
        result = power_bi_exporter.export_to_power_bi(SAMPLE_METRICS)
        assert result["status"] == "exported"

    def test_power_bi_format_has_columns_and_rows(self, power_bi_exporter):
        result = power_bi_exporter.export_to_power_bi(SAMPLE_METRICS)
        data = result["data"]
        assert "columns" in data
        assert "rows" in data

    def test_power_bi_columns_include_company_id(self, power_bi_exporter):
        result = power_bi_exporter.export_to_power_bi(SAMPLE_METRICS)
        assert "company_id" in result["data"]["columns"]

    def test_power_bi_row_count_matches_companies(self, power_bi_exporter):
        result = power_bi_exporter.export_to_power_bi(SAMPLE_METRICS)
        assert len(result["data"]["rows"]) == len(SAMPLE_METRICS)
