"""Custom exceptions for Fortune 500 Knowledge Graph Analytics."""


class QuerySyntaxError(Exception):
    """Raised when a Cypher query has invalid syntax."""
    pass


class QueryTimeoutError(Exception):
    """Raised when a Cypher query execution exceeds the timeout limit."""
    pass


class RateLimitError(Exception):
    """Raised when GitHub API rate limit is exceeded."""
    pass


class InsufficientDataError(Exception):
    """Raised when there is insufficient data for ML model training."""
    pass
