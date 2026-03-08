#!/usr/bin/env python3
"""
Property Test Suite Runner for Wikipedia Intelligence System
Executes all 71 property tests and generates a comprehensive report
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Property test files and their expected property counts
PROPERTY_TEST_FILES = {
    "tests/property/test_config_properties.py": [67, 68, 69],
    "tests/property/test_storage_properties.py": [18, 19],
    "tests/property/test_rate_limiting_properties.py": [57, 58, 59, 60, 61, 62],
    "tests/property/test_edit_history_properties.py": [5, 6, 7, 8, 9, 10],
    "tests/property/test_crawl4ai_properties.py": [11, 12, 13, 15, 16, 17],
    "tests/property/test_etl_properties.py": [50, 51, 52, 53, 56],
    "tests/property/test_forecasting_properties.py": [20, 21, 22, 23, 24],
    "tests/property/test_reputation_properties.py": [25, 26, 27, 28],
    "tests/property/test_clustering_properties.py": [29, 30, 31, 32, 33],
    "tests/property/test_dashboard_properties.py": [34, 35, 36, 37, 38, 39],
    "tests/property/test_hype_properties.py": [40, 41, 42, 43, 44],
    "tests/property/test_knowledge_graph_properties.py": [45, 46, 47, 48, 49],
    "tests/property/test_alert_properties.py": [54],
    "tests/property/test_logging_properties.py": [63, 64, 65, 66],
}

def run_test_file(test_file):
    """Run a single test file and return results"""
    print(f"\n{'='*80}")
    print(f"Running: {test_file}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per file
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "file": test_file,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "elapsed_time": elapsed_time,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        return {
            "file": test_file,
            "returncode": -1,
            "stdout": "",
            "stderr": "Test file timed out after 300 seconds",
            "elapsed_time": elapsed_time,
            "success": False,
            "timeout": True
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "file": test_file,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "elapsed_time": elapsed_time,
            "success": False,
            "error": True
        }

def parse_test_results(stdout):
    """Parse pytest output to extract test counts"""
    passed = failed = skipped = 0
    
    # Look for the summary line like "4 passed in 5.63s"
    for line in stdout.split('\n'):
        if 'passed' in line.lower() or 'failed' in line.lower():
            parts = line.split()
            for i, part in enumerate(parts):
                if 'passed' in part.lower() and i > 0:
                    try:
                        passed = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass
                if 'failed' in part.lower() and i > 0:
                    try:
                        failed = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass
                if 'skipped' in part.lower() and i > 0:
                    try:
                        skipped = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass
    
    return passed, failed, skipped

def main():
    """Run all property tests and generate report"""
    print("="*80)
    print("Wikipedia Intelligence System - Property Test Suite")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Property Test Files: {len(PROPERTY_TEST_FILES)}")
    print(f"Expected Total Properties: 71")
    print("="*80)
    
    all_results = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_time = 0
    
    # Run each test file
    for test_file, properties in PROPERTY_TEST_FILES.items():
        result = run_test_file(test_file)
        all_results.append(result)
        
        if result["success"]:
            passed, failed, skipped = parse_test_results(result["stdout"])
            total_passed += passed
            total_failed += failed
            total_skipped += skipped
            print(f"✓ PASSED - {passed} tests passed, {failed} failed, {skipped} skipped in {result['elapsed_time']:.2f}s")
        else:
            if result.get("timeout"):
                print(f"✗ TIMEOUT - Test file exceeded 300 second limit")
            else:
                print(f"✗ FAILED - Test file failed to execute")
                if result["stderr"]:
                    print(f"  Error: {result['stderr'][:200]}")
        
        total_time += result["elapsed_time"]
    
    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(f"Total Test Files: {len(all_results)}")
    print(f"Successful Files: {sum(1 for r in all_results if r['success'])}")
    print(f"Failed Files: {sum(1 for r in all_results if not r['success'])}")
    print(f"\nTotal Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Total Tests Skipped: {total_skipped}")
    print(f"\nTotal Execution Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # List failed files
    failed_files = [r for r in all_results if not r["success"]]
    if failed_files:
        print("\nFAILED TEST FILES:")
        for result in failed_files:
            print(f"  - {result['file']}")
            if result.get("timeout"):
                print(f"    Reason: Timeout (>300s)")
            elif result["stderr"]:
                print(f"    Error: {result['stderr'][:100]}")
    
    # Save detailed report to file
    report_file = "property_test_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files": len(all_results),
                "successful_files": sum(1 for r in all_results if r['success']),
                "failed_files": sum(1 for r in all_results if not r['success']),
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_time": total_time
            },
            "results": all_results
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Return exit code based on results
    if total_failed > 0 or any(not r["success"] for r in all_results):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
