#!/usr/bin/env python3
"""Script to check for missing or incomplete docstrings in the codebase.

This script analyzes Python files in the wikipedia_health package and reports:
- Classes without docstrings
- Public methods without docstrings
- Functions without docstrings
- Docstrings missing required sections (Args, Returns, Raises)

Usage:
    python docs/check_docstrings.py
    python docs/check_docstrings.py --verbose
    python docs/check_docstrings.py --fix-missing
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class DocstringChecker(ast.NodeVisitor):
    """AST visitor to check for missing or incomplete docstrings."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Dict[str, str]] = []
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check class docstrings."""
        self.current_class = node.name
        
        if not ast.get_docstring(node):
            self.issues.append({
                'type': 'missing_class_docstring',
                'name': node.name,
                'line': node.lineno,
                'message': f'Class {node.name} is missing a docstring'
            })
        else:
            docstring = ast.get_docstring(node)
            self._check_docstring_quality(node.name, docstring, 'class', node.lineno)
            
        self.generic_visit(node)
        self.current_class = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function/method docstrings."""
        # Skip private methods (except __init__)
        if node.name.startswith('_') and node.name != '__init__':
            return
            
        # Determine if this is a method or function
        context = f'method of {self.current_class}' if self.current_class else 'function'
        full_name = f'{self.current_class}.{node.name}' if self.current_class else node.name
        
        if not ast.get_docstring(node):
            self.issues.append({
                'type': 'missing_function_docstring',
                'name': full_name,
                'line': node.lineno,
                'message': f'{context.capitalize()} {full_name} is missing a docstring'
            })
        else:
            docstring = ast.get_docstring(node)
            self._check_docstring_quality(full_name, docstring, context, node.lineno, node)
            
        self.generic_visit(node)
        
    def _check_docstring_quality(
        self, 
        name: str, 
        docstring: str, 
        context: str,
        lineno: int,
        node: ast.AST = None
    ) -> None:
        """Check if docstring has required sections."""
        issues = []
        
        # Check for Args section if function has parameters
        if isinstance(node, ast.FunctionDef):
            # Get parameters (excluding self, cls)
            params = [arg.arg for arg in node.args.args 
                     if arg.arg not in ('self', 'cls')]
            
            if params and 'Args:' not in docstring and 'Parameters:' not in docstring:
                issues.append('missing Args section')
                
            # Check for Returns section if function returns something
            has_return = any(isinstance(n, ast.Return) and n.value is not None 
                           for n in ast.walk(node))
            if has_return and 'Returns:' not in docstring:
                issues.append('missing Returns section')
                
            # Check for Raises section if function raises exceptions
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
            if has_raise and 'Raises:' not in docstring:
                issues.append('missing Raises section')
        
        # Check for example
        if 'Example:' not in docstring and 'Examples:' not in docstring:
            issues.append('missing Example section (recommended)')
            
        if issues:
            self.issues.append({
                'type': 'incomplete_docstring',
                'name': name,
                'line': lineno,
                'message': f'{context.capitalize()} {name} docstring is incomplete: {", ".join(issues)}'
            })


def check_file(filepath: Path) -> List[Dict[str, str]]:
    """Check a single Python file for docstring issues.
    
    Args:
        filepath: Path to Python file to check.
        
    Returns:
        List of issues found in the file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))
            
        checker = DocstringChecker(str(filepath))
        checker.visit(tree)
        return checker.issues
    except SyntaxError as e:
        return [{
            'type': 'syntax_error',
            'name': str(filepath),
            'line': e.lineno or 0,
            'message': f'Syntax error: {e.msg}'
        }]
    except Exception as e:
        return [{
            'type': 'error',
            'name': str(filepath),
            'line': 0,
            'message': f'Error processing file: {str(e)}'
        }]


def check_package(package_dir: Path, verbose: bool = False) -> Tuple[int, int]:
    """Check all Python files in a package directory.
    
    Args:
        package_dir: Path to package directory.
        verbose: If True, print detailed information.
        
    Returns:
        Tuple of (total_files_checked, total_issues_found).
    """
    all_issues = []
    files_checked = 0
    
    # Find all Python files
    for filepath in package_dir.rglob('*.py'):
        # Skip __pycache__ and test files
        if '__pycache__' in str(filepath) or 'test_' in filepath.name:
            continue
            
        if verbose:
            print(f'Checking {filepath}...')
            
        issues = check_file(filepath)
        if issues:
            all_issues.extend([{**issue, 'file': str(filepath)} for issue in issues])
        files_checked += 1
    
    # Print summary
    print(f'\nDocstring Check Summary')
    print(f'=' * 60)
    print(f'Files checked: {files_checked}')
    print(f'Issues found: {len(all_issues)}')
    print()
    
    if all_issues:
        # Group by type
        by_type = {}
        for issue in all_issues:
            issue_type = issue['type']
            if issue_type not in by_type:
                by_type[issue_type] = []
            by_type[issue_type].append(issue)
        
        # Print by type
        for issue_type, issues in sorted(by_type.items()):
            print(f'\n{issue_type.replace("_", " ").title()} ({len(issues)}):')
            print('-' * 60)
            for issue in issues:
                print(f'  {issue["file"]}:{issue["line"]}')
                print(f'    {issue["message"]}')
                print()
    else:
        print('✓ All public APIs have docstrings!')
    
    return files_checked, len(all_issues)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Check for missing or incomplete docstrings'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )
    parser.add_argument(
        '--package-dir',
        type=Path,
        default=Path('wikipedia_health'),
        help='Package directory to check (default: wikipedia_health)'
    )
    
    args = parser.parse_args()
    
    if not args.package_dir.exists():
        print(f'Error: Package directory {args.package_dir} does not exist')
        sys.exit(1)
    
    files_checked, issues_found = check_package(args.package_dir, args.verbose)
    
    # Exit with error code if issues found
    if issues_found > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
