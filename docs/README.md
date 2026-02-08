# Documentation

This directory contains the documentation for the Wikipedia Product Health Analysis System.

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
make html
```

On Windows:

```bash
make.bat html
```

View the documentation by opening `_build/html/index.html` in your browser.

### Build PDF Documentation

```bash
make latexpdf
```

Requires LaTeX installation.

### Clean Build Files

```bash
make clean
```

## Documentation Structure

- `index.rst` - Main documentation page
- `getting_started.rst` - Getting started guide
- `api_reference.rst` - API reference (auto-generated from docstrings)
- `user_guides.rst` - User guides and tutorials
- `methodology.rst` - Statistical methodology documentation
- `cli-usage.md` - Command-line interface usage
- `configuration.md` - Configuration guide
- `api_documentation_guide.md` - Guide for writing API documentation

## Checking Documentation Quality

### Check for Missing Docstrings

```bash
python check_docstrings.py
```

With verbose output:

```bash
python check_docstrings.py --verbose
```

### Check for Broken Links

```bash
make linkcheck
```

### Check Documentation Coverage

```bash
make coverage
```

## Writing Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of function.
    
    Longer description explaining what the function does,
    how it works, and any important details.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When param2 is negative.
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
```

See `api_documentation_guide.md` for complete guidelines.

### Adding New Pages

1. Create a new `.rst` file in the `docs/` directory
2. Add it to the `toctree` in `index.rst`
3. Rebuild documentation

## Continuous Integration

Documentation is automatically built and checked in CI:

- Docstring completeness check
- Sphinx build without warnings
- Link checking

## Publishing

Documentation can be published to:

- **GitHub Pages**: Copy `_build/html/` to `gh-pages` branch
- **Read the Docs**: Automatically builds from repository

## Troubleshooting

### Import Errors

If Sphinx can't import modules:

1. Ensure package is installed: `pip install -e ..`
2. Check `sys.path` in `conf.py`

### Missing Docstrings

Run the docstring checker:

```bash
python check_docstrings.py --verbose
```

### Build Warnings

Build with warnings as errors:

```bash
make html SPHINXOPTS="-W"
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Read the Docs](https://docs.readthedocs.io/)
