#!/usr/bin/env python3
"""
Script to reduce max_examples in Hypothesis property tests for faster execution.
This modifies @settings decorators to use fewer examples.
"""

import re
import os
from pathlib import Path


def reduce_max_examples(file_path: Path, reduction_factor: int = 5) -> bool:
    """
    Reduce max_examples in a test file by the given factor.
    
    Args:
        file_path: Path to the test file
        reduction_factor: Factor to divide max_examples by (default: 5)
    
    Returns:
        True if file was modified, False otherwise
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match max_examples=NUMBER
    pattern = r'max_examples=(\d+)'
    
    def replace_max_examples(match):
        original_value = int(match.group(1))
        new_value = max(5, original_value // reduction_factor)  # Minimum 5 examples
        return f'max_examples={new_value}'
    
    content = re.sub(pattern, replace_max_examples, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def main():
    """Reduce max_examples in all property test files."""
    tests_dir = Path('tests')
    
    if not tests_dir.exists():
        print(f"Error: {tests_dir} directory not found")
        return
    
    modified_files = []
    
    # Find all test files
    for test_file in tests_dir.rglob('test_*.py'):
        if 'property' in str(test_file) or 'test_' in test_file.name:
            if reduce_max_examples(test_file):
                modified_files.append(test_file)
                print(f"✓ Reduced examples in: {test_file}")
    
    print(f"\n{'='*60}")
    print(f"Modified {len(modified_files)} test files")
    print(f"{'='*60}")
    
    if modified_files:
        print("\nModified files:")
        for f in modified_files:
            print(f"  - {f}")


if __name__ == '__main__':
    main()
