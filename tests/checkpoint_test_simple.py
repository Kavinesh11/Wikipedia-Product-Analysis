"""
Checkpoint Test: Data Collection Components (Simplified)

This script tests the data collection components without making real API calls.
It verifies:
1. Data structures are correctly defined
2. Components can be instantiated
3. Basic functionality works with mock data
4. Error handling is implemented

Task 8: Checkpoint - Ensure data collection works
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.storage.dto import (
    RevisionRecord, VandalismMetrics, EditMetrics,
    ArticleContent
)


def test_revision_record_structure():
    """Test RevisionRecord data structure."""
    print("\n" + "="*80)
    print("TEST 1: RevisionRecord Data Structure")
    print("="*80)
    
    try:
        # Create a sample revision record
        revision = RevisionRecord(
            article="Python_(programming_language)",
            revision_id=123456789,
            timestamp=datetime.now(),
            editor_type="registered",
            editor_id="TestUser",
            is_reverted=False,
            bytes_changed=100,
            edit_summary="Updated documentation"
        )
        
        print(f"\n✓ RevisionRecord created successfully")
        print(f"  - Article: {revision.article}")
        print(f"  - Revision ID: {revision.revision_id}")
        print(f"  - Editor Type: {revision.editor_type}")
        print(f"  - Timestamp: {revision.timestamp}")
        
        # Test validation
        assert revision.editor_type in ["anonymous", "registered"]
        print(f"\n✓ Validation works correctly")
        
        # Test invalid editor type
        try:
            invalid_revision = RevisionRecord(
                article="Test",
                revision_id=1,
                timestamp=datetime.now(),
                editor_type="invalid",  # Invalid type
                editor_id="Test",
                is_reverted=False,
                bytes_changed=0,
                edit_summary=""
            )
            print(f"\n✗ Validation failed - should have rejected invalid editor_type")
            return False
        except ValueError as e:
            print(f"\n✓ Validation correctly rejected invalid editor_type: {e}")
        
        print("\n" + "="*80)
        print("✓ RevisionRecord: PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ RevisionRecord: FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_article_content_structure():
    """Test ArticleContent data structure."""
    print("\n" + "="*80)
    print("TEST 2: ArticleContent Data Structure")
    print("="*80)
    
    try:
        # Create a sample article content
        article = ArticleContent(
            title="Python (programming language)",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            summary="Python is a high-level programming language...",
            infobox={"Paradigm": "Multi-paradigm", "Designed by": "Guido van Rossum"},
            tables=[],
            categories=["Programming languages", "Python (programming language)"],
            internal_links=["/wiki/Programming_language", "/wiki/Guido_van_Rossum"],
            crawl_timestamp=datetime.now()
        )
        
        print(f"\n✓ ArticleContent created successfully")
        print(f"  - Title: {article.title}")
        print(f"  - URL: {article.url}")
        print(f"  - Summary length: {len(article.summary)} characters")
        print(f"  - Infobox fields: {len(article.infobox)}")
        print(f"  - Categories: {len(article.categories)}")
        print(f"  - Internal links: {len(article.internal_links)}")
        
        # Test validation
        assert article.url.startswith("http")
        print(f"\n✓ URL validation works correctly")
        
        # Test invalid URL
        try:
            invalid_article = ArticleContent(
                title="Test",
                url="invalid-url",  # Invalid URL
                summary="",
                infobox={},
                tables=[],
                categories=[],
                internal_links=[],
                crawl_timestamp=datetime.now()
            )
            print(f"\n✗ Validation failed - should have rejected invalid URL")
            return False
        except ValueError as e:
            print(f"\n✓ Validation correctly rejected invalid URL: {e}")
        
        print("\n" + "="*80)
        print("✓ ArticleContent: PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ ArticleContent: FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edit_history_scraper_instantiation():
    """Test that EditHistoryScraper can be instantiated."""
    print("\n" + "="*80)
    print("TEST 3: EditHistoryScraper Instantiation")
    print("="*80)
    
    try:
        from src.data_ingestion.edit_history_scraper import EditHistoryScraper
        
        # Test static methods without instantiation
        print(f"\n✓ Testing editor classification (static method)")
        assert EditHistoryScraper._classify_editor("192.168.1.1") == "anonymous"
        assert EditHistoryScraper._classify_editor("TestUser") == "registered"
        assert EditHistoryScraper._classify_editor("2001:0db8:85a3::8a2e:0370:7334") == "anonymous"  # IPv6
        print(f"  - IP addresses classified as anonymous")
        print(f"  - Usernames classified as registered")
        
        # Create mock scraper instance (without API client to avoid event loop issues)
        print(f"\n✓ Testing scraper methods with mock data")
        
        # Test edit velocity calculation with mock data
        revisions = [
            RevisionRecord(
                article="Test",
                revision_id=i,
                timestamp=datetime.now() - timedelta(hours=i),
                editor_type="registered",
                editor_id="User",
                is_reverted=False,
                bytes_changed=10,
                edit_summary="Edit"
            )
            for i in range(10)
        ]
        
        # Create a minimal scraper instance for testing (will fail on API calls but methods work)
        scraper = EditHistoryScraper.__new__(EditHistoryScraper)
        
        velocity = scraper.calculate_edit_velocity(revisions, window_hours=24)
        print(f"  - Edit velocity calculation: {velocity:.2f} edits/hour")
        assert velocity >= 0, "Velocity should be non-negative"
        
        # Test vandalism detection with reverts
        revisions_with_reverts = revisions.copy()
        revisions_with_reverts.append(
            RevisionRecord(
                article="Test",
                revision_id=100,
                timestamp=datetime.now(),
                editor_type="registered",
                editor_id="Admin",
                is_reverted=False,
                bytes_changed=-100,
                edit_summary="Reverted vandalism by User"
            )
        )
        
        vandalism_metrics = scraper.detect_vandalism_signals(revisions_with_reverts)
        print(f"  - Vandalism detection:")
        print(f"    Total edits: {vandalism_metrics.total_edits}")
        print(f"    Reverted edits: {vandalism_metrics.reverted_edits}")
        print(f"    Vandalism percentage: {vandalism_metrics.vandalism_percentage:.1f}%")
        
        assert vandalism_metrics.total_edits == len(revisions_with_reverts)
        assert 0 <= vandalism_metrics.vandalism_percentage <= 100
        assert vandalism_metrics.reverted_edits > 0, "Should detect revert keyword"
        
        # Test rolling window metrics
        metrics_by_window = scraper.calculate_rolling_window_metrics(revisions)
        print(f"  - Rolling window metrics: {len(metrics_by_window)} windows calculated")
        assert len(metrics_by_window) > 0
        
        print("\n" + "="*80)
        print("✓ EditHistoryScraper: PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ EditHistoryScraper: FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crawl4ai_pipeline_instantiation():
    """Test that Crawl4AIPipeline can be instantiated."""
    print("\n" + "="*80)
    print("TEST 4: Crawl4AIPipeline Instantiation")
    print("="*80)
    
    try:
        from src.data_ingestion.crawl4ai_pipeline import Crawl4AIPipeline, ExtractionConfig
        
        # Create pipeline instance
        pipeline = Crawl4AIPipeline(
            max_concurrent=5,
            timeout=30.0,
            crawl_delay=2.0
        )
        
        print(f"\n✓ Crawl4AIPipeline instantiated successfully")
        print(f"  - Max concurrent: {pipeline.max_concurrent}")
        print(f"  - Timeout: {pipeline.timeout}s")
        print(f"  - Crawl delay: {pipeline.crawl_delay}s")
        
        # Test extraction config
        config = ExtractionConfig(
            extract_summary=True,
            extract_infobox=True,
            extract_tables=True
        )
        print(f"\n✓ ExtractionConfig created successfully")
        
        # Test infobox extraction with mock HTML
        html = """
        <html>
            <body>
                <table class="infobox">
                    <tr><th>Field1</th><td>Value1</td></tr>
                    <tr><th>Field2</th><td>Value2</td></tr>
                </table>
            </body>
        </html>
        """
        infobox = pipeline.extract_infobox(html)
        print(f"\n✓ Infobox extraction works: {len(infobox)} fields extracted")
        assert isinstance(infobox, dict)
        assert len(infobox) > 0
        
        # Test table extraction with mock HTML
        html_table = """
        <html>
            <body>
                <table class="wikitable">
                    <tr><th>Col1</th><th>Col2</th></tr>
                    <tr><td>A</td><td>B</td></tr>
                    <tr><td>C</td><td>D</td></tr>
                </table>
            </body>
        </html>
        """
        tables = pipeline.extract_tables(html_table)
        print(f"\n✓ Table extraction works: {len(tables)} tables extracted")
        assert isinstance(tables, list)
        
        # Test internal link extraction with mock HTML
        html_links = """
        <html>
            <body>
                <div class="mw-parser-output">
                    <a href="/wiki/Python">Python</a>
                    <a href="/wiki/Programming">Programming</a>
                    <a href="/wiki/File:Test.jpg">File</a>
                    <a href="https://external.com">External</a>
                </div>
            </body>
        </html>
        """
        links = pipeline.extract_internal_links(html_links)
        print(f"\n✓ Internal link extraction works: {len(links)} links extracted")
        assert isinstance(links, list)
        # Should filter out File: links and external links
        assert all(link.startswith("/wiki/") for link in links)
        assert not any("File:" in link for link in links)
        
        print("\n" + "="*80)
        print("✓ Crawl4AIPipeline: PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ Crawl4AIPipeline: FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checkpoint tests."""
    print("\n" + "="*80)
    print("CHECKPOINT TEST: Data Collection Components (Simplified)")
    print("="*80)
    print("\nThis test validates data structures and basic functionality")
    print("without making real API calls.")
    
    results = []
    
    # Test 1: RevisionRecord structure
    result1 = test_revision_record_structure()
    results.append(("RevisionRecord Structure", result1))
    
    # Test 2: ArticleContent structure
    result2 = test_article_content_structure()
    results.append(("ArticleContent Structure", result2))
    
    # Test 3: EditHistoryScraper
    result3 = test_edit_history_scraper_instantiation()
    results.append(("EditHistoryScraper", result3))
    
    # Test 4: Crawl4AIPipeline
    result4 = test_crawl4ai_pipeline_instantiation()
    results.append(("Crawl4AIPipeline", result4))
    
    # Summary
    print("\n" + "="*80)
    print("CHECKPOINT TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Data collection components are correctly structured!")
        print("\nNOTE: Real API testing requires:")
        print("  1. Proper User-Agent configuration for Wikipedia API")
        print("  2. Playwright browsers installed (run: playwright install)")
        print("  3. Network connectivity to Wikipedia")
    else:
        print("✗ SOME TESTS FAILED - Please review the errors above")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
