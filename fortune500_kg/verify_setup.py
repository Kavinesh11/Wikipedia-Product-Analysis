"""Verify Fortune 500 KG Analytics setup."""

import os
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} missing: {filepath}")
        return False


def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    if Path(dirpath).exists() and Path(dirpath).is_dir():
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description} missing: {dirpath}")
        return False


def main():
    """Run setup verification checks."""
    print("=" * 60)
    print("Fortune 500 Knowledge Graph - Setup Verification")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    print("\n1. Checking project structure...")
    
    # Check directories
    directories = [
        ("infrastructure", "Infrastructure directory"),
        ("infrastructure/schema", "Schema directory"),
        ("tests", "Tests directory"),
    ]
    
    for dirpath, description in directories:
        checks_total += 1
        if check_directory_exists(dirpath, description):
            checks_passed += 1
    
    print("\n2. Checking configuration files...")
    
    # Check configuration files
    config_files = [
        ("requirements.txt", "Requirements file"),
        ("docker-compose.yml", "Docker Compose configuration"),
        (".env.example", "Environment example file"),
        ("pytest.ini", "Pytest configuration"),
    ]
    
    for filepath, description in config_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    print("\n3. Checking schema files...")
    
    # Check schema files
    schema_files = [
        ("infrastructure/schema/01_create_constraints.cypher", "Constraints script"),
        ("infrastructure/schema/02_create_indexes.cypher", "Indexes script"),
        ("infrastructure/schema/03_schema_documentation.cypher", "Schema documentation"),
    ]
    
    for filepath, description in schema_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    print("\n4. Checking Python modules...")
    
    # Check Python modules
    python_files = [
        ("infrastructure/__init__.py", "Infrastructure package"),
        ("infrastructure/connection.py", "Connection module"),
        ("infrastructure/schema_manager.py", "Schema manager module"),
        ("infrastructure/init_schema.py", "Schema initialization script"),
        ("tests/conftest.py", "Test configuration"),
        ("tests/test_infrastructure.py", "Infrastructure tests"),
        ("tests/test_schema_properties.py", "Property-based tests"),
    ]
    
    for filepath, description in python_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    print("\n5. Checking documentation...")
    
    # Check documentation
    doc_files = [
        ("README.md", "Main README"),
        ("INSTALLATION.md", "Installation guide"),
        ("infrastructure/README.md", "Infrastructure README"),
    ]
    
    for filepath, description in doc_files:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    print("\n6. Checking environment configuration...")
    
    checks_total += 1
    if Path(".env").exists():
        print("✓ .env file exists")
        checks_passed += 1
    else:
        print("⚠ .env file not found (copy from .env.example)")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Verification Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("\n✓ All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start Neo4j: docker-compose up -d")
        print("3. Initialize schema: cd infrastructure && python init_schema.py")
        print("4. Run tests: pytest tests/")
        return 0
    else:
        print(f"\n✗ {checks_total - checks_passed} checks failed.")
        print("Please review the missing files/directories above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
