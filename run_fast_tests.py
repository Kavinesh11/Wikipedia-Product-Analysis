#!/usr/bin/env python3
"""
Run property tests with reduced examples and generate a summary report.
"""

import subprocess
import sys
from datetime import datetime


def run_tests():
    """Run pytest and capture results."""
    print("="*70)
    print("RUNNING PROPERTY TESTS WITH REDUCED EXAMPLES (5 per test)")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run pytest with summary output
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/property/",
        "-v",
        "--tb=line",  # Shorter traceback
        "--maxfail=10",  # Stop after 10 failures
        "-x",  # Stop on first failure for faster feedback
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print("\n" + "="*70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("\n❌ Tests timed out after 3 minutes")
        return 1
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
