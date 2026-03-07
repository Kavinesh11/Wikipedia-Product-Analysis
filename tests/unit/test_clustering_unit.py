"""Unit tests for Topic Clustering Engine

Tests clustering functionality with sample article sets, CAGR calculations,
and normalization logic.

Feature: wikipedia-intelligence-system
Requirements: 7.1, 7.2, 7.3, 7.4
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analytics.clustering import TopicClusteringEngine
from src.storage.dto import ArticleContent, GrowthMetrics


class TestTopicClusteringEngine:
    """Unit tests for TopicClusteringEngine"""
    
    def test_cluster_sample_articles(self):
        """Test clustering with sample article sets
        
        Verifies that articles with similar content are grouped together.
        Requirements: 7.1
        """
        # Create sample articles with distinct topics
        tech_articles = [
            ArticleContent(
                title="Python Programming",
                url="https://en.wikipedia.org/wiki/Python",
                summary="Python is a high-level programming language for general-purpose programming",
                infobox={},
                tables=[],
                categories=["Programming languages", "Software development"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="Java Programming",
                url="https://en.wikipedia.org/wiki/Java",
                summary="Java is a class-based object-oriented programming language",
                infobox={},
                tables=[],
                categories=["Programming languages", "Object-oriented programming"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="JavaScript",
                url="https://en.wikipedia.org/wiki/JavaScript",
                summary="JavaScript is a programming language for web development",
                infobox={},
                tables=[],
                categories=["Programming languages", "Web development"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="Apple Inc",
                url="https://en.wikipedia.org/wiki/Apple_Inc",
                summary="Apple Inc is an American technology company that designs consumer electronics",
                infobox={},
                tables=[],
                categories=["Technology companies", "Consumer electronics"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="Microsoft",
                url="https://en.wikipedia.org/wiki/Microsoft",
                summary="Microsoft Corporation is an American technology company that develops software",
                infobox={},
                tables=[],
                categories=["Technology companies", "Software companies"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
        ]
        
        # Create engine with 2 clusters (programming languages vs companies)
        engine = TopicClusteringEngine(n_clusters=2)
        result = engine.cluster_articles(tech_articles)
        
        # Verify clustering result structure
        assert result.n_clusters == 2
        assert len(result.cluster_assignments) == 5
        assert len(result.confidence_scores) == 5
        assert len(result.cluster_labels) == 2
        
        # Verify all articles are assigned
        for article in tech_articles:
            assert article.title in result.cluster_assignments
            assert article.title in result.confidence_scores
        
        # Verify confidence scores are in valid range
        for title, confidence in result.confidence_scores.items():
            assert 0.0 <= confidence <= 1.0
        
        # Verify similar articles tend to cluster together
        # Programming languages should be in same cluster
        python_cluster = result.cluster_assignments["Python Programming"]
        java_cluster = result.cluster_assignments["Java Programming"]
        js_cluster = result.cluster_assignments["JavaScript"]
        
        # At least 2 of the 3 programming languages should be in same cluster
        prog_clusters = [python_cluster, java_cluster, js_cluster]
        most_common_cluster = max(set(prog_clusters), key=prog_clusters.count)
        assert prog_clusters.count(most_common_cluster) >= 2
    
    def test_cagr_calculation_with_known_values(self):
        """Test CAGR calculation with known values
        
        Verifies CAGR formula: ((end_value / start_value) ^ (1 / years)) - 1
        Requirements: 7.4
        """
        # Create sample articles
        articles = [
            ArticleContent(
                title=f"Article_{i}",
                url=f"https://en.wikipedia.org/wiki/Article_{i}",
                summary=f"This is article {i} about technology and innovation",
                infobox={},
                tables=[],
                categories=["Technology"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            )
            for i in range(5)
        ]
        
        # Cluster articles
        engine = TopicClusteringEngine(n_clusters=2)
        result = engine.cluster_articles(articles)
        
        # Create pageview data with known growth pattern
        # Start: 1000 views/day, End: 2000 views/day over 365 days
        # Expected CAGR: ((2000/1000)^(1/1)) - 1 = 1.0 = 100%
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        pageview_data = []
        for i, date in enumerate(dates):
            # Linear growth from 1000 to 2000
            views = 1000 + (1000 * i / (len(dates) - 1))
            for article in articles:
                pageview_data.append({
                    'article': article.title,
                    'date': date,
                    'views': int(views)
                })
        
        pageviews_df = pd.DataFrame(pageview_data)
        
        # Calculate CAGR for cluster 0
        cluster_id = result.cluster_assignments[articles[0].title]
        
        # Calculate growth metrics directly
        metrics = engine.calculate_cluster_growth(cluster_id, pageviews_df)
        
        # Growth rate should be approximately 100% (from 1000 to 2000)
        # Allow for some variance due to clustering and aggregation
        assert 80 <= metrics.growth_rate <= 120, f"Expected growth ~100%, got {metrics.growth_rate}%"
    
    def test_cagr_with_zero_growth(self):
        """Test CAGR calculation with zero growth
        
        Requirements: 7.4
        """
        articles = [
            ArticleContent(
                title="Stable Article",
                url="https://en.wikipedia.org/wiki/Stable",
                summary="This article has stable traffic",
                infobox={},
                tables=[],
                categories=["Stable"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="Another Stable",
                url="https://en.wikipedia.org/wiki/Another",
                summary="Another stable article",
                infobox={},
                tables=[],
                categories=["Stable"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="Third Stable",
                url="https://en.wikipedia.org/wiki/Third",
                summary="Third stable article",
                infobox={},
                tables=[],
                categories=["Stable"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
        ]
        
        engine = TopicClusteringEngine(n_clusters=2)
        result = engine.cluster_articles(articles)
        
        # Create flat pageview data (no growth)
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(365)]
        
        pageview_data = []
        for date in dates:
            for article in articles:
                pageview_data.append({
                    'article': article.title,
                    'date': date,
                    'views': 1000  # Constant views
                })
        
        pageviews_df = pd.DataFrame(pageview_data)
        
        # Get the cluster ID for the first article
        cluster_id = result.cluster_assignments[articles[0].title]
        cagr = engine.calculate_topic_cagr(cluster_id, pageviews_df, years=1)
        
        # CAGR should be approximately 0%
        assert -5 <= cagr <= 5, f"Expected CAGR ~0%, got {cagr}%"
    
    def test_normalization_logic(self):
        """Test baseline normalization in industry comparison
        
        Verifies that normalization adjusts growth rates by baseline traffic.
        Requirements: 7.3
        """
        # Create articles for two clusters with different baseline traffic
        high_traffic_articles = [
            ArticleContent(
                title=f"Popular_{i}",
                url=f"https://en.wikipedia.org/wiki/Popular_{i}",
                summary="Popular article with high baseline traffic and lots of views",
                infobox={},
                tables=[],
                categories=["Popular", "High Traffic"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            )
            for i in range(3)
        ]
        
        low_traffic_articles = [
            ArticleContent(
                title=f"Niche_{i}",
                url=f"https://en.wikipedia.org/wiki/Niche_{i}",
                summary="Niche article with low baseline traffic and fewer views",
                infobox={},
                tables=[],
                categories=["Niche", "Low Traffic"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            )
            for i in range(3)
        ]
        
        all_articles = high_traffic_articles + low_traffic_articles
        
        # Cluster into 2 groups
        engine = TopicClusteringEngine(n_clusters=2)
        result = engine.cluster_articles(all_articles)
        
        # Create pageview data with different baselines
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(30)]
        
        pageview_data = []
        for i, date in enumerate(dates):
            # High traffic cluster: baseline 10000, grows to 15000 (50% growth)
            for article in high_traffic_articles:
                views = 10000 + (5000 * i / 29)
                pageview_data.append({
                    'article': article.title,
                    'date': date,
                    'views': int(views)
                })
            
            # Low traffic cluster: baseline 1000, grows to 1500 (50% growth)
            for article in low_traffic_articles:
                views = 1000 + (500 * i / 29)
                pageview_data.append({
                    'article': article.title,
                    'date': date,
                    'views': int(views)
                })
        
        pageviews_df = pd.DataFrame(pageview_data)
        
        # Get cluster IDs
        high_cluster = result.cluster_assignments[high_traffic_articles[0].title]
        low_cluster = result.cluster_assignments[low_traffic_articles[0].title]
        
        # Compare without normalization
        comparison_no_norm = engine.compare_industries(
            [high_cluster, low_cluster],
            pageviews_df,
            normalize_baseline=False
        )
        
        # Both should have similar growth rates (around 50%)
        high_metrics_no_norm = next(m for m in comparison_no_norm.clusters if m.cluster_id == high_cluster)
        low_metrics_no_norm = next(m for m in comparison_no_norm.clusters if m.cluster_id == low_cluster)
        
        # Without normalization, growth rates should be similar (both ~50%)
        assert abs(high_metrics_no_norm.growth_rate - low_metrics_no_norm.growth_rate) < 10
        
        # Compare with normalization
        comparison_norm = engine.compare_industries(
            [high_cluster, low_cluster],
            pageviews_df,
            normalize_baseline=True
        )
        
        # Verify baseline_normalized flag is set correctly
        assert comparison_norm.baseline_normalized is True
        assert comparison_no_norm.baseline_normalized is False
        
        # With normalization, values should be different from non-normalized
        high_metrics_norm = next(m for m in comparison_norm.clusters if m.cluster_id == high_cluster)
        low_metrics_norm = next(m for m in comparison_norm.clusters if m.cluster_id == low_cluster)
        
        # Normalized values should differ from non-normalized
        # The normalization formula is: (growth_rate * total_views) / baseline
        # This should produce different values for different baselines
        assert high_metrics_norm.growth_rate != high_metrics_no_norm.growth_rate or \
               low_metrics_norm.growth_rate != low_metrics_no_norm.growth_rate
    
    def test_empty_cluster_handling(self):
        """Test handling of clusters with no articles
        
        Requirements: 7.1, 7.2
        """
        articles = [
            ArticleContent(
                title=f"Article_{i}",
                url=f"https://en.wikipedia.org/wiki/Article_{i}",
                summary="Sample article content",
                infobox={},
                tables=[],
                categories=["Category"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            )
            for i in range(3)
        ]
        
        engine = TopicClusteringEngine(n_clusters=2)
        result = engine.cluster_articles(articles)
        
        # Create pageviews for only some articles
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(10)]
        
        pageview_data = []
        for date in dates:
            # Only add pageviews for first article
            pageview_data.append({
                'article': articles[0].title,
                'date': date,
                'views': 1000
            })
        
        pageviews_df = pd.DataFrame(pageview_data)
        
        # Calculate growth for both clusters
        for cluster_id in range(2):
            metrics = engine.calculate_cluster_growth(cluster_id, pageviews_df)
            
            # Should not raise error, should return valid metrics
            assert isinstance(metrics, GrowthMetrics)
            assert metrics.cluster_id == cluster_id
            assert metrics.growth_rate >= 0 or metrics.growth_rate == 0.0
    
    def test_insufficient_data_for_cagr(self):
        """Test CAGR calculation with insufficient data
        
        Requirements: 7.4
        """
        articles = [
            ArticleContent(
                title="Article",
                url="https://en.wikipedia.org/wiki/Article",
                summary="Sample article",
                infobox={},
                tables=[],
                categories=["Category"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="Article2",
                url="https://en.wikipedia.org/wiki/Article2",
                summary="Another sample",
                infobox={},
                tables=[],
                categories=["Category"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="Article3",
                url="https://en.wikipedia.org/wiki/Article3",
                summary="Third sample",
                infobox={},
                tables=[],
                categories=["Category"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
        ]
        
        engine = TopicClusteringEngine(n_clusters=2)
        result = engine.cluster_articles(articles)
        
        # Create pageviews for only 30 days (less than 1 year)
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(30)]
        
        pageview_data = []
        for date in dates:
            for article in articles:
                pageview_data.append({
                    'article': article.title,
                    'date': date,
                    'views': 1000
                })
        
        pageviews_df = pd.DataFrame(pageview_data)
        
        # Get cluster ID for first article
        cluster_id = result.cluster_assignments[articles[0].title]
        
        # Should handle gracefully and return CAGR based on available data
        cagr = engine.calculate_topic_cagr(cluster_id, pageviews_df, years=1)
        
        # Should return a valid number (not raise error)
        assert isinstance(cagr, (int, float))
        assert not np.isnan(cagr)
    
    def test_cluster_growth_with_single_datapoint(self):
        """Test growth calculation with only one data point
        
        Requirements: 7.2
        """
        articles = [
            ArticleContent(
                title="Single Point Article",
                url="https://en.wikipedia.org/wiki/Single",
                summary="Article with single data point",
                infobox={},
                tables=[],
                categories=["Test"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="Another Single",
                url="https://en.wikipedia.org/wiki/Another",
                summary="Another test article",
                infobox={},
                tables=[],
                categories=["Test"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
            ArticleContent(
                title="Third Single",
                url="https://en.wikipedia.org/wiki/Third",
                summary="Third test article",
                infobox={},
                tables=[],
                categories=["Test"],
                internal_links=[],
                crawl_timestamp=datetime.now()
            ),
        ]
        
        engine = TopicClusteringEngine(n_clusters=2)
        result = engine.cluster_articles(articles)
        
        # Create pageviews for only one date
        pageview_data = []
        date = datetime(2023, 1, 1)
        for article in articles:
            pageview_data.append({
                'article': article.title,
                'date': date,
                'views': 1000
            })
        
        pageviews_df = pd.DataFrame(pageview_data)
        
        # Get cluster ID for first article
        cluster_id = result.cluster_assignments[articles[0].title]
        metrics = engine.calculate_cluster_growth(cluster_id, pageviews_df)
        
        # Should return zero growth (can't calculate growth with 1 point)
        assert metrics.growth_rate == 0.0
        assert metrics.cagr == 0.0
        assert metrics.total_views > 0  # Should have some views
        assert metrics.article_count > 0
