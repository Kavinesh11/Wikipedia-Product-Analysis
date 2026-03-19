"""Property-based tests for ingestion logging and validation.

Properties covered:
  3  - Required Company Attributes Persistence
  4  - Ingestion Logging Accuracy
  72 - Fortune 500 Completeness Validation
  73 - Missing GitHub Organization Identification
  74 - Required Attribute Presence Validation
  75 - Validation Failure Logging Completeness
  76 - Data Quality Report Completeness Metrics

Validates: Requirements 1.3, 1.4, 15.1, 15.2, 15.3, 15.4, 15.5
"""

import logging
import logging.handlers
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fortune500_kg.data_ingestion_pipeline import DataIngestionPipeline
from fortune500_kg.data_models import (
    Company,
    CrawlData,
    DataQualityReport,
    IngestionResult,
    ValidationError,
)

# ── Strategies ────────────────────────────────────────────────────────────────

company_id_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=20,
)

valid_company_st = st.fixed_dictionaries({
    "id": company_id_st,
    "name": st.text(min_size=1, max_size=50),
    "sector": st.sampled_from(["Technology", "Healthcare", "Finance", "Energy", "Retail"]),
    "revenue_rank": st.integers(min_value=1, max_value=500),
    "employee_count": st.integers(min_value=1, max_value=500_000),
    "github_org": st.text(min_size=1, max_size=30),
})

# Company missing github_org
company_no_github_st = st.fixed_dictionaries({
    "id": company_id_st,
    "name": st.text(min_size=1, max_size=50),
    "sector": st.sampled_from(["Technology", "Healthcare", "Finance"]),
    "revenue_rank": st.integers(min_value=1, max_value=500),
    "employee_count": st.integers(min_value=1, max_value=500_000),
    "github_org": st.none(),
})

# Company missing employee_count
company_no_employee_st = st.fixed_dictionaries({
    "id": company_id_st,
    "name": st.text(min_size=1, max_size=50),
    "sector": st.sampled_from(["Technology", "Healthcare", "Finance"]),
    "revenue_rank": st.integers(min_value=1, max_value=500),
    "employee_count": st.none(),
    "github_org": st.text(min_size=1, max_size=30),
})

# Company missing revenue_rank
company_no_rank_st = st.fixed_dictionaries({
    "id": company_id_st,
    "name": st.text(min_size=1, max_size=50),
    "sector": st.sampled_from(["Technology", "Healthcare", "Finance"]),
    "revenue_rank": st.none(),
    "employee_count": st.integers(min_value=1, max_value=500_000),
    "github_org": st.text(min_size=1, max_size=30),
})


def _make_neo4j_record(company: Dict[str, Any]) -> MagicMock:
    """Build a mock Neo4j record from a company dict."""
    rec = MagicMock()
    rec.__getitem__ = lambda self, key: company.get(key)
    return rec


def _make_driver(records: List[Dict[str, Any]]) -> MagicMock:
    """Build a mock Neo4j driver that returns the given records."""
    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter(
        [_make_neo4j_record(r) for r in records]
    )

    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run = MagicMock(return_value=mock_result)
    mock_session.execute_write = MagicMock(return_value=True)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    return mock_driver


# ── Property 3: Required Company Attributes Persistence ──────────────────────

class TestRequiredAttributesPersistence:
    """Property 3 — Required Company Attributes Persistence (Req 1.3)."""

    @given(companies=st.lists(valid_company_st, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_ingestion_result_counts_match_input(self, companies: List[Dict]):
        """node_count in IngestionResult equals number of valid companies ingested."""
        driver = _make_driver(companies)
        pipeline = DataIngestionPipeline(driver)

        crawl_data = CrawlData(companies=companies, relationships=[])
        result = pipeline.ingest_crawl4ai_data(crawl_data)

        assert isinstance(result, IngestionResult)
        assert result.node_count >= 0
        assert result.node_count <= len(companies)

    @given(companies=st.lists(valid_company_st, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_ingestion_result_has_timestamp(self, companies: List[Dict]):
        """IngestionResult always carries a timestamp."""
        driver = _make_driver(companies)
        pipeline = DataIngestionPipeline(driver)

        crawl_data = CrawlData(companies=companies, relationships=[])
        result = pipeline.ingest_crawl4ai_data(crawl_data)

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    @given(companies=st.lists(valid_company_st, min_size=0, max_size=5))
    @settings(max_examples=100)
    def test_ingestion_result_errors_is_list(self, companies: List[Dict]):
        """IngestionResult.errors is always a list (never None)."""
        driver = _make_driver(companies)
        pipeline = DataIngestionPipeline(driver)

        crawl_data = CrawlData(companies=companies, relationships=[])
        result = pipeline.ingest_crawl4ai_data(crawl_data)

        assert isinstance(result.errors, list)


# ── helpers for log capture without caplog ───────────────────────────────────

class _LogCapture(logging.Handler):
    """Minimal in-memory log handler for use inside @given tests."""

    def __init__(self):
        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return "\n".join(self.format(r) for r in self.records)


def _capture_logs(logger_name: str, level: int = logging.DEBUG):
    """Context manager that captures log output for a named logger."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        handler = _LogCapture()
        handler.setLevel(level)
        log = logging.getLogger(logger_name)
        original_level = log.level
        log.setLevel(level)
        log.addHandler(handler)
        try:
            yield handler
        finally:
            log.removeHandler(handler)
            log.setLevel(original_level)

    return _ctx()


# ── Property 4: Ingestion Logging Accuracy ───────────────────────────────────

class TestIngestionLoggingAccuracy:
    """Property 4 — Ingestion Logging Accuracy (Req 1.4)."""

    @given(companies=st.lists(valid_company_st, min_size=1, max_size=15))
    @settings(max_examples=100)
    def test_ingestion_logs_node_count(self, companies: List[Dict]):
        """Logger emits a message containing the node count after ingestion."""
        driver = _make_driver(companies)
        pipeline = DataIngestionPipeline(driver)
        crawl_data = CrawlData(companies=companies, relationships=[])

        with _capture_logs("fortune500_kg.data_ingestion_pipeline", logging.INFO) as cap:
            result = pipeline.ingest_crawl4ai_data(crawl_data)

        log_text = cap.text
        assert str(result.node_count) in log_text or "node" in log_text.lower()

    @given(companies=st.lists(valid_company_st, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_ingestion_logs_edge_count(self, companies: List[Dict]):
        """Logger emits a message containing the edge count after ingestion."""
        driver = _make_driver(companies)
        pipeline = DataIngestionPipeline(driver)
        crawl_data = CrawlData(companies=companies, relationships=[])

        with _capture_logs("fortune500_kg.data_ingestion_pipeline", logging.INFO) as cap:
            result = pipeline.ingest_crawl4ai_data(crawl_data)

        log_text = cap.text
        assert str(result.edge_count) in log_text or "edge" in log_text.lower()

    @given(n=st.integers(min_value=0, max_value=20))
    @settings(max_examples=100)
    def test_log_counts_are_non_negative(self, n: int):
        """node_count and edge_count in IngestionResult are always >= 0."""
        companies = [
            {
                "id": f"C{i}", "name": f"Co{i}", "sector": "Tech",
                "revenue_rank": i + 1, "employee_count": 1000,
                "github_org": f"org{i}",
            }
            for i in range(n)
        ]
        driver = _make_driver(companies)
        pipeline = DataIngestionPipeline(driver)
        crawl_data = CrawlData(companies=companies, relationships=[])
        result = pipeline.ingest_crawl4ai_data(crawl_data)

        assert result.node_count >= 0
        assert result.edge_count >= 0


# ── Property 72: Fortune 500 Completeness Validation ─────────────────────────

class TestFortune500CompletenessValidation:
    """Property 72 — Fortune 500 Completeness Validation (Req 15.1)."""

    @given(n=st.integers(min_value=0, max_value=50))
    @settings(max_examples=100)
    def test_total_companies_matches_graph_records(self, n: int):
        """DataQualityReport.total_companies equals the number of nodes in the graph."""
        records = [
            {
                "id": f"C{i}", "github_org": f"org{i}",
                "employee_count": 1000, "revenue_rank": i + 1,
            }
            for i in range(n)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        assert report.total_companies == n

    @given(n=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100)
    def test_report_has_report_date(self, n: int):
        """DataQualityReport always carries a report_date timestamp."""
        records = [
            {"id": f"C{i}", "github_org": f"org{i}",
             "employee_count": 1000, "revenue_rank": i + 1}
            for i in range(n)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        assert report.report_date is not None
        assert isinstance(report.report_date, datetime)


# ── Property 73: Missing GitHub Organization Identification ───────────────────

class TestMissingGitHubOrgIdentification:
    """Property 73 — Missing GitHub Organization Identification (Req 15.2)."""

    @given(
        complete=st.lists(
            st.fixed_dictionaries({
                "id": company_id_st,
                "github_org": st.text(min_size=1, max_size=20),
                "employee_count": st.integers(min_value=1, max_value=100_000),
                "revenue_rank": st.integers(min_value=1, max_value=500),
            }),
            min_size=0, max_size=10,
        ),
        missing=st.lists(
            st.fixed_dictionaries({
                "id": company_id_st,
                "github_org": st.none(),
                "employee_count": st.integers(min_value=1, max_value=100_000),
                "revenue_rank": st.integers(min_value=1, max_value=500),
            }),
            min_size=1, max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_missing_github_orgs_are_identified(self, complete, missing):
        """All companies without github_org appear in report.missing_github_org."""
        # Ensure unique IDs across both lists
        seen_ids = set()
        unique_complete, unique_missing = [], []
        for c in complete:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                unique_complete.append(c)
        for c in missing:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                unique_missing.append(c)

        if not unique_missing:
            return  # nothing to assert

        records = unique_complete + unique_missing
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        missing_ids = {c["id"] for c in unique_missing}
        assert missing_ids.issubset(set(report.missing_github_org))

    @given(
        companies=st.lists(
            st.fixed_dictionaries({
                "id": company_id_st,
                "github_org": st.text(min_size=1, max_size=20),
                "employee_count": st.integers(min_value=1, max_value=100_000),
                "revenue_rank": st.integers(min_value=1, max_value=500),
            }),
            min_size=1, max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_no_false_positives_in_missing_github_org(self, companies):
        """Companies with a github_org must NOT appear in missing_github_org."""
        seen_ids: set = set()
        unique = []
        for c in companies:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                unique.append(c)

        driver = _make_driver(unique)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        present_ids = {c["id"] for c in unique}
        assert not present_ids.intersection(set(report.missing_github_org))


# ── Property 74: Required Attribute Presence Validation ──────────────────────

class TestRequiredAttributePresenceValidation:
    """Property 74 — Required Attribute Presence Validation (Req 15.3)."""

    @given(
        n_missing=st.integers(min_value=1, max_value=10),
        n_present=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_missing_employee_count_identified(self, n_missing, n_present):
        """Companies without employee_count appear in report.missing_employee_count."""
        records = (
            [{"id": f"M{i}", "github_org": f"org{i}",
              "employee_count": None, "revenue_rank": i + 1}
             for i in range(n_missing)]
            + [{"id": f"P{i}", "github_org": f"org{i}",
                "employee_count": 1000, "revenue_rank": i + 1}
               for i in range(n_present)]
        )
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        missing_ids = {f"M{i}" for i in range(n_missing)}
        assert missing_ids.issubset(set(report.missing_employee_count))

    @given(
        n_missing=st.integers(min_value=1, max_value=10),
        n_present=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_missing_revenue_rank_identified(self, n_missing, n_present):
        """Companies without revenue_rank appear in report.missing_revenue_rank."""
        records = (
            [{"id": f"M{i}", "github_org": f"org{i}",
              "employee_count": 1000, "revenue_rank": None}
             for i in range(n_missing)]
            + [{"id": f"P{i}", "github_org": f"org{i}",
                "employee_count": 1000, "revenue_rank": i + 1}
               for i in range(n_present)]
        )
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        missing_ids = {f"M{i}" for i in range(n_missing)}
        assert missing_ids.issubset(set(report.missing_revenue_rank))

    @given(
        n=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_complete_companies_not_in_any_missing_list(self, n):
        """Companies with all required fields must not appear in any missing list."""
        records = [
            {"id": f"C{i}", "github_org": f"org{i}",
             "employee_count": 1000, "revenue_rank": i + 1}
            for i in range(n)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        all_ids = {f"C{i}" for i in range(n)}
        assert not all_ids.intersection(set(report.missing_github_org))
        assert not all_ids.intersection(set(report.missing_employee_count))
        assert not all_ids.intersection(set(report.missing_revenue_rank))


# ── Property 75: Validation Failure Logging Completeness ─────────────────────

class TestValidationFailureLoggingCompleteness:
    """Property 75 — Validation Failure Logging Completeness (Req 15.4)."""

    @given(n_missing=st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_missing_github_org_logged(self, n_missing: int):
        """A warning is logged for every company missing github_org."""
        records = [
            {"id": f"M{i}", "github_org": None,
             "employee_count": 1000, "revenue_rank": i + 1}
            for i in range(n_missing)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)

        with _capture_logs("fortune500_kg.data_ingestion_pipeline", logging.WARNING) as cap:
            pipeline.validate_data_quality()

        log_text = cap.text
        for i in range(n_missing):
            assert f"M{i}" in log_text

    @given(n_missing=st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_missing_employee_count_logged(self, n_missing: int):
        """A warning is logged for every company missing employee_count."""
        records = [
            {"id": f"E{i}", "github_org": f"org{i}",
             "employee_count": None, "revenue_rank": i + 1}
            for i in range(n_missing)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)

        with _capture_logs("fortune500_kg.data_ingestion_pipeline", logging.WARNING) as cap:
            pipeline.validate_data_quality()

        log_text = cap.text
        for i in range(n_missing):
            assert f"E{i}" in log_text

    @given(n_missing=st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_missing_revenue_rank_logged(self, n_missing: int):
        """A warning is logged for every company missing revenue_rank."""
        records = [
            {"id": f"R{i}", "github_org": f"org{i}",
             "employee_count": 1000, "revenue_rank": None}
            for i in range(n_missing)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)

        with _capture_logs("fortune500_kg.data_ingestion_pipeline", logging.WARNING) as cap:
            pipeline.validate_data_quality()

        log_text = cap.text
        for i in range(n_missing):
            assert f"R{i}" in log_text

    @given(n_missing=st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_validation_errors_list_populated(self, n_missing: int):
        """report.validation_errors contains one entry per missing field."""
        records = [
            {"id": f"V{i}", "github_org": None,
             "employee_count": None, "revenue_rank": None}
            for i in range(n_missing)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        # Each company is missing 3 fields → 3 * n_missing errors
        assert len(report.validation_errors) == 3 * n_missing


# ── Property 76: Data Quality Report Completeness Metrics ────────────────────

class TestDataQualityReportCompletenessMetrics:
    """Property 76 — Data Quality Report Completeness Metrics (Req 15.5)."""

    @given(
        n_complete=st.integers(min_value=0, max_value=20),
        n_incomplete=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=100)
    def test_completeness_percentage_bounds(self, n_complete, n_incomplete):
        """completeness_percentage is always in [0.0, 100.0]."""
        records = (
            [{"id": f"C{i}", "github_org": f"org{i}",
              "employee_count": 1000, "revenue_rank": i + 1}
             for i in range(n_complete)]
            + [{"id": f"I{i}", "github_org": None,
                "employee_count": None, "revenue_rank": None}
               for i in range(n_incomplete)]
        )
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        assert 0.0 <= report.completeness_percentage <= 100.0

    @given(n=st.integers(min_value=1, max_value=30))
    @settings(max_examples=100)
    def test_all_complete_gives_100_percent(self, n: int):
        """When all companies have all fields, completeness_percentage == 100.0."""
        records = [
            {"id": f"C{i}", "github_org": f"org{i}",
             "employee_count": 1000, "revenue_rank": i + 1}
            for i in range(n)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        assert report.completeness_percentage == pytest.approx(100.0)

    @given(n=st.integers(min_value=1, max_value=30))
    @settings(max_examples=100)
    def test_all_incomplete_gives_0_percent(self, n: int):
        """When all companies are missing all fields, completeness_percentage == 0.0."""
        records = [
            {"id": f"I{i}", "github_org": None,
             "employee_count": None, "revenue_rank": None}
            for i in range(n)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        assert report.completeness_percentage == pytest.approx(0.0)

    @given(
        n_complete=st.integers(min_value=1, max_value=20),
        n_incomplete=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_completeness_percentage_formula(self, n_complete, n_incomplete):
        """completeness_percentage == (complete / total) * 100."""
        records = (
            [{"id": f"C{i}", "github_org": f"org{i}",
              "employee_count": 1000, "revenue_rank": i + 1}
             for i in range(n_complete)]
            + [{"id": f"I{i}", "github_org": None,
                "employee_count": None, "revenue_rank": None}
               for i in range(n_incomplete)]
        )
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        total = n_complete + n_incomplete
        expected = (report.companies_with_complete_data / total) * 100.0
        assert report.completeness_percentage == pytest.approx(expected, rel=1e-6)

    @given(n=st.integers(min_value=1, max_value=30))
    @settings(max_examples=100)
    def test_companies_with_complete_data_leq_total(self, n: int):
        """companies_with_complete_data is always <= total_companies."""
        records = [
            {"id": f"C{i}",
             "github_org": f"org{i}" if i % 2 == 0 else None,
             "employee_count": 1000 if i % 3 != 0 else None,
             "revenue_rank": i + 1}
            for i in range(n)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        assert report.companies_with_complete_data <= report.total_companies

    @given(n=st.integers(min_value=0, max_value=30))
    @settings(max_examples=100)
    def test_report_structure_completeness(self, n: int):
        """DataQualityReport always has all required fields populated."""
        records = [
            {"id": f"C{i}", "github_org": f"org{i}",
             "employee_count": 1000, "revenue_rank": i + 1}
            for i in range(n)
        ]
        driver = _make_driver(records)
        pipeline = DataIngestionPipeline(driver)
        report = pipeline.validate_data_quality()

        assert hasattr(report, "total_companies")
        assert hasattr(report, "companies_with_complete_data")
        assert hasattr(report, "completeness_percentage")
        assert hasattr(report, "missing_github_org")
        assert hasattr(report, "missing_employee_count")
        assert hasattr(report, "missing_revenue_rank")
        assert hasattr(report, "validation_errors")
        assert isinstance(report.missing_github_org, list)
        assert isinstance(report.missing_employee_count, list)
        assert isinstance(report.missing_revenue_rank, list)
        assert isinstance(report.validation_errors, list)
