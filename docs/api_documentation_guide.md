# API Documentation Guide

This guide explains how to generate and maintain API documentation for the Wikipedia Product Health Analysis System.

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

Or install with dev dependencies:

```bash
pip install -e ".[dev]"
```

### Generate HTML Documentation

From the `docs/` directory:

```bash
cd docs
make html
```

On Windows:

```bash
cd docs
make.bat html
```

The generated documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in a browser.

### Generate PDF Documentation

```bash
cd docs
make latexpdf
```

Requires LaTeX installation.

### Clean Build Files

```bash
cd docs
make clean
```

## Documentation Structure

```
docs/
├── conf.py                    # Sphinx configuration
├── index.rst                  # Documentation home page
├── getting_started.rst        # Getting started guide
├── api_reference.rst          # Auto-generated API docs
├── user_guides.rst            # User guides and tutorials
├── methodology.rst            # Statistical methodology docs
├── cli-usage.md              # CLI usage guide
├── configuration.md          # Configuration guide
├── requirements.txt          # Documentation dependencies
├── Makefile                  # Unix build script
└── make.bat                  # Windows build script
```

## Docstring Format

All public classes and methods must have docstrings following Google style:

### Class Docstring Example

```python
class WikimediaAPIClient:
    """Client for fetching data from Wikimedia APIs.
    
    This client handles data acquisition from Wikimedia's REST APIs,
    including pageviews, editor counts, and edit volumes. It implements
    retry logic with exponential backoff and validates all responses.
    
    Args:
        config: API configuration containing endpoints, timeout, and retry settings.
        
    Attributes:
        config: The API configuration object.
        session: Requests session for connection pooling.
        
    Example:
        >>> from wikipedia_health.config import load_config
        >>> config = load_config()
        >>> client = WikimediaAPIClient(config.api)
        >>> data = client.fetch_pageviews(
        ...     start_date=date(2020, 1, 1),
        ...     end_date=date(2023, 12, 31),
        ...     platforms=['desktop', 'mobile-web']
        ... )
    """
```

### Method Docstring Example

```python
def fetch_pageviews(
    self,
    start_date: date,
    end_date: date,
    platforms: List[str],
    agent_type: str = 'user'
) -> pd.DataFrame:
    """Fetch pageview data from Wikimedia Pageviews API.
    
    Retrieves daily pageview counts for specified platforms and date range.
    Automatically filters bot traffic when agent_type='user'.
    
    Args:
        start_date: Start date for data retrieval (inclusive).
        end_date: End date for data retrieval (inclusive).
        platforms: List of platforms to fetch. Valid values: 'desktop',
            'mobile-web', 'mobile-app', 'all'.
        agent_type: Agent type filter. Use 'user' to exclude bots,
            'all-agents' to include all traffic. Defaults to 'user'.
            
    Returns:
        DataFrame with columns: date, platform, pageviews, metadata.
        
    Raises:
        DataAcquisitionError: If API request fails after all retries.
        ValueError: If date range is invalid or platforms list is empty.
        
    Example:
        >>> client = WikimediaAPIClient(config)
        >>> data = client.fetch_pageviews(
        ...     start_date=date(2020, 1, 1),
        ...     end_date=date(2020, 12, 31),
        ...     platforms=['desktop', 'mobile-web'],
        ...     agent_type='user'
        ... )
        >>> print(data.shape)
        (730, 4)  # 365 days * 2 platforms
    """
```

### Property Docstring Example

```python
@property
def is_valid(self) -> bool:
    """Check if validation passed.
    
    Returns:
        True if all validation checks passed, False otherwise.
    """
```

## Docstring Sections

### Required Sections

1. **Summary**: One-line description (first line)
2. **Extended Description**: Detailed explanation (optional but recommended)
3. **Args**: All parameters with types and descriptions
4. **Returns**: Return value type and description
5. **Raises**: All exceptions that may be raised

### Optional Sections

6. **Example**: Usage examples (highly recommended)
7. **Note**: Important notes or caveats
8. **Warning**: Warnings about potential issues
9. **See Also**: Related functions or classes
10. **References**: Academic references or external links

## Type Annotations

Always include type annotations in function signatures:

```python
from typing import List, Dict, Optional, Tuple
from datetime import date
import pandas as pd

def analyze_trend(
    data: pd.DataFrame,
    start_date: date,
    end_date: date,
    confidence_level: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """Analyze trend with confidence interval."""
    pass
```

## Documentation Best Practices

### 1. Be Concise but Complete

- First line: One-sentence summary
- Extended description: 2-3 sentences explaining purpose and behavior
- Keep examples short and focused

### 2. Document All Public APIs

- All public classes
- All public methods and functions
- All public attributes and properties
- Module-level functions

### 3. Don't Document Private APIs

- Skip methods starting with `_` (unless truly important)
- Skip internal helper functions
- Use `exclude-members` in Sphinx config

### 4. Include Examples

- Show typical usage
- Include expected output when helpful
- Use doctest format for testable examples

### 5. Document Exceptions

- List all exceptions that may be raised
- Explain when each exception occurs
- Include exception hierarchy if relevant

### 6. Cross-Reference

Use Sphinx cross-references:

```python
"""
See Also:
    :class:`WikimediaAPIClient`: For data acquisition.
    :func:`validate_data`: For data validation.
    :mod:`wikipedia_health.config`: For configuration options.
"""
```

### 7. Mathematical Notation

Use LaTeX for mathematical formulas:

```python
"""
Calculates Cohen's d effect size:
    
.. math::
    d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_{pooled}}
    
where :math:`s_{pooled}` is the pooled standard deviation.
"""
```

## Updating Documentation

### When Adding New Features

1. Add docstrings to all new classes and methods
2. Update relevant user guides
3. Add examples to documentation
4. Rebuild documentation to check for errors

### When Changing APIs

1. Update docstrings to reflect changes
2. Update examples if behavior changed
3. Add deprecation warnings if removing features
4. Update changelog

### Deprecation Example

```python
def old_method(self):
    """Old method (deprecated).
    
    .. deprecated:: 0.2.0
        Use :func:`new_method` instead.
    """
    warnings.warn(
        "old_method is deprecated, use new_method instead",
        DeprecationWarning,
        stacklevel=2
    )
```

## Checking Documentation

### Build Without Warnings

```bash
cd docs
make clean
make html SPHINXOPTS="-W"
```

The `-W` flag treats warnings as errors.

### Check Links

```bash
cd docs
make linkcheck
```

### Check Coverage

```bash
cd docs
make coverage
```

This generates a report of undocumented APIs.

## Publishing Documentation

### GitHub Pages

1. Build documentation:
   ```bash
   cd docs
   make html
   ```

2. Copy `_build/html/` to `gh-pages` branch

3. Push to GitHub

### Read the Docs

1. Create `.readthedocs.yaml`:
   ```yaml
   version: 2
   
   build:
     os: ubuntu-22.04
     tools:
       python: "3.9"
   
   sphinx:
     configuration: docs/conf.py
   
   python:
     install:
       - requirements: docs/requirements.txt
       - method: pip
         path: .
   ```

2. Connect repository to Read the Docs

3. Documentation builds automatically on push

## Troubleshooting

### Import Errors

If Sphinx can't import modules:

1. Check `sys.path` in `conf.py`
2. Ensure package is installed: `pip install -e .`
3. Check for circular imports

### Missing Docstrings

If docstrings don't appear:

1. Check indentation (must be first statement in function/class)
2. Verify `autodoc` extension is enabled
3. Check `autodoc_default_options` in `conf.py`

### Formatting Issues

If formatting looks wrong:

1. Check docstring format (Google vs NumPy)
2. Verify `napoleon` extension is enabled
3. Check indentation in docstrings

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [NumPy Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Read the Docs](https://docs.readthedocs.io/)
