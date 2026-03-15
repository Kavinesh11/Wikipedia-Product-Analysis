"""Unit tests for custom Cypher query execution (Task 21.1).

Covers:
- AnalyticsEngine.validate_cypher_syntax (Requirement 16.2)
- AnalyticsEngine.execute_custom_query (Requirements 16.1-16.5)
"""

import pytest
from datetime import datetime

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository
from fortune500_kg.exceptions import QuerySyntaxError, QueryTimeoutError
from fortune500_kg.data_models import QueryResult


@pytest.fixture
def engine():
    return AnalyticsEngine(metrics_repo=MetricsRepository())


def mock_executor_ok(query):
    """Mock executor returning 2 rows."""
    return (["company_id", "score"], [{"company_id": "C1", "score": 9.0}, {"company_id": "C2", "score": 7.0}], 5.0)


def mock_executor_slow(query):
    """Mock executor simulating a slow query (>30s)."""
    return ([], [], 31000.0)  # 31 seconds in ms


# ---------------------------------------------------------------------------
# validate_cypher_syntax
# ---------------------------------------------------------------------------

class TestValidateCypherSyntax:

    def test_valid_match_query(self, engine):
        assert engine.validate_cypher_syntax("MATCH (n:Company) RETURN n") is True

    def test_valid_create_query(self, engine):
        assert engine.validate_cypher_syntax("CREATE (n:Company {name: 'Test'})") is True

    def test_valid_merge_query(self, engine):
        assert engine.validate_cypher_syntax("MERGE (n:Company {id: '1'})") is True

    def test_valid_return_query(self, engine):
        assert engine.validate_cypher_syntax("RETURN 1 AS result") is True

    def test_empty_query_raises_syntax_error(self, engine):
        with pytest.raises(QuerySyntaxError):
            engine.validate_cypher_syntax("")

    def test_whitespace_only_raises_syntax_error(self, engine):
        with pytest.raises(QuerySyntaxError):
            engine.validate_cypher_syntax("   ")

    def test_invalid_start_raises_syntax_error(self, engine):
        with pytest.raises(QuerySyntaxError):
            engine.validate_cypher_syntax("SELECT * FROM companies")

    def test_case_insensitive_validation(self, engine):
        assert engine.validate_cypher_syntax("match (n) return n") is True


# ---------------------------------------------------------------------------
# execute_custom_query
# ---------------------------------------------------------------------------

class TestExecuteCustomQuery:

    def test_returns_query_result(self, engine):
        result = engine.execute_custom_query(
            "MATCH (n) RETURN n",
            mock_executor=mock_executor_ok,
        )
        assert isinstance(result, QueryResult)

    def test_result_has_columns_and_rows(self, engine):
        result = engine.execute_custom_query(
            "MATCH (n) RETURN n",
            mock_executor=mock_executor_ok,
        )
        assert result.columns == ["company_id", "score"]
        assert len(result.rows) == 2

    def test_result_has_execution_time(self, engine):
        result = engine.execute_custom_query(
            "MATCH (n) RETURN n",
            mock_executor=mock_executor_ok,
        )
        assert result.execution_time_ms >= 0

    def test_invalid_syntax_raises_error(self, engine):
        with pytest.raises(QuerySyntaxError):
            engine.execute_custom_query("INVALID QUERY")

    def test_timeout_raises_error(self, engine):
        with pytest.raises(QueryTimeoutError):
            engine.execute_custom_query(
                "MATCH (n) RETURN n",
                timeout_seconds=30.0,
                mock_executor=mock_executor_slow,
            )

    def test_audit_log_populated(self, engine):
        engine.execute_custom_query(
            "MATCH (n) RETURN n",
            user_id="test_user",
            mock_executor=mock_executor_ok,
        )
        assert hasattr(engine, '_audit_log')
        assert len(engine._audit_log) >= 1
        last_entry = engine._audit_log[-1]
        assert last_entry["user_id"] == "test_user"
        assert last_entry["query"] == "MATCH (n) RETURN n"
        assert "timestamp" in last_entry

    def test_audit_log_has_timestamp(self, engine):
        engine.execute_custom_query(
            "MATCH (n) RETURN n",
            mock_executor=mock_executor_ok,
        )
        last_entry = engine._audit_log[-1]
        assert isinstance(last_entry["timestamp"], datetime)

    def test_result_stored_in_tabular_format(self, engine):
        result = engine.execute_custom_query(
            "MATCH (n) RETURN n",
            mock_executor=mock_executor_ok,
        )
        # Rows should be list of dicts (tabular format)
        for row in result.rows:
            assert isinstance(row, dict)
