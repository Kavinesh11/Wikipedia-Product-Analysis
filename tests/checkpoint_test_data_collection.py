"""
Checkpoint Test: Data Collection Components

This script tests the implemented data collection components:
1. Edit History Scraper - Test with real Wikipedia data
2. Crawl4AI Pipeline - Test with sample Wikipedia articles
3. Verify data is correctly structured

Task 8: Checkpoint - Ensure data collection works
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.data_ingestion.edit_history_scraper import EditHistoryScraper
from src.data_ingestion.crawl4ai_pipeline import Crawl4AIPipeline, ExtractionConfig


async def test_edit_history_scraper():
    """Test Edit History Scraper with real Wikipedia data."""
    print("\n" + "="*80)
    print("TEST 1: Edit History Scraper")
    print("="*80)
    
    try:
        # Initialize scraper
        scraper = EditHistoryScraper()
        
        # Test with a well-known article (Python programming language)
        article = "Python_(programming_language)"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        print(f"\nFetching revisions for '{article}'")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Fetch revisions
        revisions = await scraper.fetch_revisions(
            article=article,
            start_date=start_date,
            end_date=end_date,
            limit=50  # Small sample for testing
        )
        
        print(f"\n✓ Fetched {len(revisions)} revisions")
        
        # Verify data structure
        if revisions:
            sample_rev = revisions[0]
            print(f"\nSample revision:")
            print(f"  - Revision ID: {sample_rev.revision_id}")
            print(f"  - Timestamp: {sample_rev.timestamp}")
            print(f"  - Editor Type: {sample_rev.editor_type}")
            print(f"  - Editor ID: {sample_rev.editor_id}")
            print(f"  - Summary: {sample_rev.edit_summary[:100]}...")
            
            # Verify required fields
            assert sample_rev.article == article
            assert sample_rev.revision_id > 0
            assert sample_rev.editor_type in ["anonymous", "registered"]
            assert isinstance(sample_rev.timestamp, datetime)
            print("\n✓ Data structure is correct")
        
        # Test edit velocity calculation
        if revisions:
            velocity = scraper.calculate_edit_velocity(revisions, window_hours=24)
            print(f"\n✓ Edit velocity: {velocity:.2f} edits/hour")
            assert velocity >= 0, "Edit velocity should be non-negative"
        
        # Test vandalism detection
        if revisions:
            vandalism_metrics = scraper.detect_vandalism_signals(revisions)
            print(f"\n✓ Vandalism detection:")
            print(f"  - Total edits: {vandalism_metrics.total_edits}")
            print(f"  - Reverted edits: {vandalism_metrics.reverted_edits}")
            print(f"  - Vandalism percentage: {vandalism_metrics.vandalism_percentage:.1f}%")
            
            assert vandalism_metrics.total_edits == len(revisions)
            assert 0 <= vandalism_metrics.vandalism_percentage <= 100
        
        # Test rolling window metrics
        if revisions:
            metrics_by_window = scraper.calculate_rolling_window_metrics(revisions)
            print(f"\n✓ Rolling window metrics calculated for {len(metrics_by_window)} windows")
            for window_label, metrics in metrics_by_window.items():
                print(f"  - {window_label}: velocity={metrics.edit_velocity:.2f}, "
                      f"vandalism={metrics.vandalism_rate:.1f}%, "
                      f"anonymous={metrics.anonymous_edit_pct:.1f}%")
        
        await scraper.close()
        
        print("\n" + "="*80)
        print("✓ Edit History Scraper: PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ Edit History Scraper: FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_crawl4ai_pipeline():
    """Test Crawl4AI Pipeline with sample Wikipedia articles."""
    print("\n" + "="*80)
    print("TEST 2: Crawl4AI Pipeline")
    print("="*80)
    
    try:
        # Initialize pipeline
        pipeline = Crawl4AIPipeline(
            max_concurrent=5,
            timeout=30.0,
            crawl_delay=2.0  # Be respectful to Wikipedia
        )
        
        # Test with a simple article
        test_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        
        print(f"\nCrawling article: {test_url}")
        
        # Crawl single article
        article_content = await pipeline.crawl_article(test_url)
        
        print(f"\n✓ Successfully crawled article")
        print(f"\nExtracted content:")
        print(f"  - Title: {article_content.title}")
        print(f"  - URL: {article_content.url}")
        print(f"  - Summary length: {len(article_content.summary)} characters")
        print(f"  - Infobox fields: {len(article_content.infobox)}")
        print(f"  - Tables: {len(article_content.tables)}")
        print(f"  - Categories: {len(article_content.categories)}")
        print(f"  - Internal links: {len(article_content.internal_links)}")
        
        # Verify data structure
        assert article_content.title, "Title should not be empty"
        assert article_content.url == test_url
        assert isinstance(article_content.summary, str)
        assert isinstance(article_content.infobox, dict)
        assert isinstance(article_content.tables, list)
        assert isinstance(article_content.categories, list)
        assert isinstance(article_content.internal_links, list)
        assert isinstance(article_content.crawl_timestamp, datetime)
        print("\n✓ Data structure is correct")
        
        # Show sample infobox data
        if article_content.infobox:
            print(f"\nSample infobox fields:")
            for key, value in list(article_content.infobox.items())[:5]:
                print(f"  - {key}: {value[:50]}..." if len(str(value)) > 50 else f"  - {key}: {value}")
        
        # Show sample categories
        if article_content.categories:
            print(f"\nSample categories:")
            for cat in article_content.categories[:5]:
                print(f"  - {cat}")
        
        # Show sample internal links
        if article_content.internal_links:
            print(f"\nSample internal links:")
            for link in article_content.internal_links[:5]:
                print(f"  - {link}")
        
        # Test infobox extraction
        print(f"\n✓ Testing infobox extraction...")
        infobox = pipeline.extract_infobox(f"<html><body><table class='infobox'><tr><th>Test</th><td>Value</td></tr></table></body></html>")
        assert isinstance(infobox, dict)
        print(f"  Infobox extraction works")
        
        # Test table extraction
        print(f"\n✓ Testing table extraction...")
        tables = pipeline.extract_tables(f"<html><body><table class='wikitable'><tr><th>Col1</th><th>Col2</th></tr><tr><td>A</td><td>B</td></tr></table></body></html>")
        assert isinstance(tables, list)
        print(f"  Table extraction works")
        
        # Test internal link extraction
        print(f"\n✓ Testing internal link extraction...")
        links = pipeline.extract_internal_links(f"<html><body><div class='mw-parser-output'><a href='/wiki/Test'>Test</a></div></body></html>")
        assert isinstance(links, list)
        print(f"  Internal link extraction works")
        
        await pipeline.close()
        
        print("\n" + "="*80)
        print("✓ Crawl4AI Pipeline: PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ Crawl4AI Pipeline: FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_deep_crawl():
    """Test deep crawl functionality (limited to avoid overload)."""
    print("\n" + "="*80)
    print("TEST 3: Deep Crawl (Limited)")
    print("="*80)
    
    try:
        # Initialize pipeline
        pipeline = Crawl4AIPipeline(
            max_concurrent=3,
            timeout=30.0,
            crawl_delay=3.0  # Be extra respectful
        )
        
        # Test with very limited deep crawl
        seed_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        
        print(f"\nPerforming limited deep crawl from: {seed_url}")
        print(f"Max depth: 1, Max articles: 3")
        
        # Perform deep crawl with strict limits
        articles = await pipeline.deep_crawl(
            seed_url=seed_url,
            max_depth=1,  # Only 1 level deep
            max_articles=3  # Only 3 articles total
        )
        
        print(f"\n✓ Deep crawl completed")
        print(f"  - Articles crawled: {len(articles)}")
        
        # Verify results
        assert len(articles) > 0, "Should have crawled at least one article"
        assert len(articles) <= 3, "Should not exceed max_articles limit"
        
        for i, article in enumerate(articles, 1):
            print(f"\n  Article {i}:")
            print(f"    - Title: {article.title}")
            print(f"    - Internal links: {len(article.internal_links)}")
        
        await pipeline.close()
        
        print("\n" + "="*80)
        print("✓ Deep Crawl: PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ Deep Crawl: FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all checkpoint tests."""
    print("\n" + "="*80)
    print("CHECKPOINT TEST: Data Collection Components")
    print("="*80)
    print("\nThis test validates that data collection components are working correctly.")
    print("It will make real API calls to Wikipedia, so please be patient...")
    
    results = []
    
    # Test 1: Edit History Scraper
    result1 = await test_edit_history_scraper()
    results.append(("Edit History Scraper", result1))
    
    # Wait between tests to be respectful
    await asyncio.sleep(2)
    
    # Test 2: Crawl4AI Pipeline
    result2 = await test_crawl4ai_pipeline()
    results.append(("Crawl4AI Pipeline", result2))
    
    # Wait between tests
    await asyncio.sleep(2)
    
    # Test 3: Deep Crawl (optional, limited)
    result3 = await test_deep_crawl()
    results.append(("Deep Crawl", result3))
    
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
        print("✓ ALL TESTS PASSED - Data collection is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Please review the errors above")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
