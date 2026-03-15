"""Property-based tests for custom Cypher query execution (Task 21.2).

Properties:
- Property 77: Cypher Query Syntax Validation
- Property 78: Query Result Tabular Format
- Property 79: Query Execution Audit Logging
- Property 80: Query Timeout Enforcement

Validates: Requirements 16.2, 16.3, 16.4, 16.5
"""

from datetime import datetime
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import pytest

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository
from fortune500_kg.exceptions import QuerySyntaxError, QueryTimeoutError
from fortune500_kg.data_models import QueryResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_valid_cypher_keyword = st.sampled_from([
    "MATCH", "CREATE", "MERGE", "RETURN", "WITH", "CALL",
    "UNWIND", "DELETE", "SET", "REMOVE", "OPTIONAL",
    "match", "create", "merge", "return",  # lowercase variants
])

_invalid_start = st.sampled_from([
    "SELECT", "INSERT", "UPDATE", "DROP", "ALTER", "EXEC",
    "FROM", "WHERE", "JOIN", "HAVING",
])

_user_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=1,
    max_size=20,
)

_column_name = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_",
    min_size=1,
    max_size=15,
)


@st.composite
def mock_query_result(draw):
    """Generate a mock query result (columns, rows, exec_time_ms)."""
    columns = draw(st.lists(_column_name, min_size=1, max_size=5, unique=True))
    n_rows = draw(st.integers(min_value=0, max_value=20))
    rows = [
        {col: draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
         for col in columns}
        for _ in range(n_rows)
    ]
    exec_time_ms = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    return columns, rows, exec_time_ms


# ---------------------------------------------------------------------------
# Property 77: Cypher Query Syntax Validation
# Feature: fortune500-kg-analytics, Property 77
# Validates: Requirements 16.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(_invalid_start, st.text(min_size=0, max_size=50))
def test_property_77_cypher_syntax_validation_rejects_invalid(invalid_start, suffix):
    """
    For any submitted Cypher query, queries with invalid syntax should be
    rejected before execution with a syntax error message.

    **Validates: Requirements 16.2**
    """
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    invalid_query = f"{invalid_start} {suffix}"

    with pytest.raises(QuerySyntaxError):
        engine.validate_cypher_syntax(invalid_query)


@settings(max_examples=100)
@given(_valid_cypher_keyword, st.text(min_size=0, max_size=50))
def test_property_77_cypher_syntax_validation_accepts_valid(keyword, suffix):
    """
    Valid Cypher queries starting with recognized keywords should pass validation.

    **Validates: Requirements 16.2**
    """
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    valid_query = f"{keyword} {suffix}"

    # Should not raise
    result = engine.validate_cypher_syntax(valid_query)
    assert result is True


# ---------------------------------------------------------------------------
# Property 78: Query Result Tabular Format
# Feature: fortune500-kg-analytics, Property 78
# Validates: Requirements 16.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(mock_query_result())
def test_property_78_query_result_tabular_format(mock_result):
    """
    For any executed Cypher query that returns results, the results should be
    formatted as a table with column headers and row data.

    **Validates: Requirements 16.3**
    """
    columns, rows, exec_time_ms = mock_result
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())

    def executor(q):
        return columns, rows, exec_time_ms

    result = engine.execute_custom_query(
        "MATCH (n) RETURN n",
        mock_executor=executor,
    )

    assert isinstance(result, QueryResult)
    # Columns must be a list of strings
    assert isinstance(result.columns, list)
    for col in result.columns:
        assert isinstance(col, str)

    # Rows must be a list of dicts (tabular format)
    assert isinstance(result.rows, list)
    for row in result.rows:
        assert isinstance(row, dict)
        # Each row must have all column keys
        for col in result.columns:
            assert col in row


# ---------------------------------------------------------------------------
# Property 79: Query Execution Audit Logging
# Feature: fortune500-kg-analytics, Property 79
# Validates: Requirements 16.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(_user_id, mock_query_result())
def test_property_79_query_audit_logging(user_id, mock_result):
    """
    For any executed Cypher query, the audit log should contain an entry with
    the query text, timestamp, and user identifier.

    **Validates: Requirements 16.4**
    """
    columns, rows, exec_time_ms = mock_result
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())

    def executor(q):
        return columns, rows, exec_time_ms

    query_text = "MATCH (n) RETURN n"
    engine.execute_custom_query(
        query_text,
        user_id=user_id,
        mock_executor=executor,
    )

    assert hasattr(engine, '_audit_log')
    assert len(engine._audit_log) >= 1

    last_entry = engine._audit_log[-1]
    assert last_entry["query"] == query_text, "Audit log must contain query text"
    assert last_entry["user_id"] == user_id, "Audit log must contain user identifier"
    assert isinstance(last_entry["timestamp"], datetime), "Audit log must contain timestamp"


# ---------------------------------------------------------------------------
# Property 80: Query Timeout Enforcement
# Feature: fortune500-kg-analytics, Property 80
# Validates: Requirements 16.5
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    st.floats(min_value=0.001, max_value=29.9, allow_nan=False, allow_infinity=False),
)
def test_property_80_query_timeout_enforcement(timeout_seconds):
    """
    For any Cypher query with execution time exceeding the timeout, the query
    should be terminated and a timeout error should be returned.

    **Validates: Requirements 16.5**
    """
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())

    # Mock executor that always exceeds the timeout
    def slow_executor(q):
        # Return execution time that exceeds the timeout
        exec_time_ms = (timeout_seconds + 1.0) * 1000
        return ([], [], exec_time_ms)

    with pytest.raises(QueryTimeoutError):
        engine.execute_custom_query(
            "MATCH (n) RETURN n",
            timeout_seconds=timeout_seconds,
            mock_executor=slow_executor,
        )
