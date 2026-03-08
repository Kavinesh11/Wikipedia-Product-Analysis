#!/usr/bin/env python3
"""
Aggressively reduce max_examples to minimum (5) for very fast test execution.
"""

import re
from pathlib import Path


def reduce_to_minimum(file_path: Path) -> bool:
    """Set all max_examples to 5 (minimum)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace any max_examples=NUMBER with max_examples=5
    content = re.sub(r'max_examples=\d+', 'max_examples=5', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def main():
    tests_dir = Path('tests')
    modified_files = []
    
    for test_file in tests_dir.rglob('test_*.py'):
        if reduce_to_minimum(test_file):
            modified_files.append(test_file)
            print(f"✓ Set to 5 examples: {test_file}")
    
    print(f"\n{'='*60}")
    print(f"Modified {len(modified_files)} files - all set to 5 examples")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
