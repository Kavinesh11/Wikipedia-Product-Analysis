#!/usr/bin/env python
"""Simple test runner for clustering property tests"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

try:
    import pytest
    
    # Run the clustering property tests
    exit_code = pytest.main([
        'tests/property/test_clustering_properties.py',
        '-v',
        '--tb=short',
        '-m', 'property'
    ])
    
    sys.exit(exit_code)
    
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
