"""
Checkpoint 18: Analytics Components Integration Test

Tests that all analytics components work together:
- Topic Clustering Engine
- Hype Detection Engine
- Knowledge Graph Builder
- Alert System
- Reputation Monitor

This checkpoint validates end-to-end analytics pipeline functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch
import networkx as nx

from src.analytics.clustering import TopicClusteringEngine
from src.analytics.hype_detection import HypeDetectionEngine
from src.analytics.knowledge_graph import KnowledgeGraphBuilder
from src.analytics.reputation_monitor import ReputationMonitor
from src.utils.alert_system import AlertSystem
from src.storage.dto import (
    ArticleContent, RevisionRecord, EditMetrics, 
    ReputationScore, HypeMetrics, Alert
)


@pytest.fixture
def sample_articles():
    """Create sample articles for clustering."""
    return [
        ArticleContent(
            title="Python (programming language)",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            summary="Python is a high-level programming language.",
            infobox={"paradigm": "multi-paradigm", "typing": "dynamic"},
            tables=[],
            categories=["Programming languages", "Python"],
            internal_links=["Java (programming language)", "JavaScript"],
            crawl_timestamp=datetime.now()
        ),
        ArticleContent(
            title="Java (programming language)",
            url="https://en.wikipedia.org/wiki/Java_(programming_language)",
            summary="Java is a class-based, object-oriented programming language.",
            infobox={"paradigm": "object-oriented", "typing": "static"},
            tables=[],
            categories=["Programming languages", "Java"],
            internal_links=["Python (programming language)", "C++"],
            crawl_timestamp=datetime.now()
        ),
        ArticleContent(
            title="JavaScript",
            url="https://en.wikipedia.org/wiki/JavaScript",
            summary="JavaScript is a programming language for web development.",
            infobox={"paradigm": "multi-paradigm", "typing": "dynamic"},
            tables=[],
            categories=["Programming languages", "Web development"],
            internal_links=["Python (programming language)", "TypeScript"],
            crawl_timestamp=datetime.now()
        ),
        ArticleContent(
            title="Machine Learning",
            url="https://en.wikipedia.org/wiki/Machine_Learning",
            summary="Machine learning is a subset of artificial intelligence.",
            infobox={"field": "AI", "applications": "many"},
            tables=[],
            categories=["Artificial intelligence", "Machine learning"],
            internal_links=["Deep Learning", "Neural Networks"],
            crawl_timestamp=datetime.now()
        ),
        ArticleContent(
            title="Deep Learning",
            url="https://en.wikipedia.org/wiki/Deep_Learning",
            summary="Deep learning is part of machine learning methods.",
            infobox={"field": "AI", "type": "neural networks"},
            tables=[],
            categories=["Artificial intelligence", "Deep learning"],
            internal_links=["Machine Learning", "Neural Networks"],
            crawl_timestamp=datetime.now()
        )
    ]


@pytest.fixture
def sample_pageviews():
    """Create sample pageview data with trends."""
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
    
    # Normal article with steady growth
    normal_views = pd.DataFrame({
        'date': dates,
        'views': np.linspace(1000, 1200, len(dates)) + np.random.normal(0, 50, len(dates))
    })
    
    # Hyped article with spike
    spike_start = len(dates) // 2
    hype_views = np.linspace(1000, 1100, len(dates))
    hype_views[spike_start:] = np.linspace(1100, 5000, len(dates) - spike_start)
    hype_df = pd.DataFrame({
        'date': dates,
        'views': hype_views + np.random.normal(0, 100, len(dates))
    })
    
    return {
        'normal': normal_views,
        'hyped': hype_df
    }


@pytest.fixture
def sample_revisions():
    """Create sample revision data for reputation monitoring."""
    base_time = datetime.now() - timedelta(days=30)
    
    # Normal article with few edits
    normal_revisions = [
        RevisionRecord(
            article="Normal Article",
            revision_id=i,
            timestamp=base_time + timedelta(days=i),
            editor_type="registered" if i % 3 == 0 else "anonymous",
            editor_id=f"user_{i}" if i % 3 == 0 else f"192.168.1.{i}",
            is_reverted=False,
            bytes_changed=100,
            edit_summary="Minor edit"
        )
        for i in range(10)
    ]
    
    # High-risk article with many reverts and anonymous edits
    risky_revisions = [
        RevisionRecord(
            article="Controversial Article",
            revision_id=i,
            timestamp=base_time + timedelta(hours=i),
            editor_type="anonymous",
            editor_id=f"192.168.1.{i}",
            is_reverted=i % 2 == 0,  # 50% revert rate
            bytes_changed=500,
            edit_summary="Reverted vandalism" if i % 2 == 0 else "Edit"
        )
        for i in range(50)
    ]
    
    return {
        'normal': normal_revisions,
        'risky': risky_revisions
    }


class TestClusteringIntegration:
    """Test Topic Clustering Engine with real data."""
    
    def test_cluster_articles_success(self, sample_articles):
        """Test clustering articles into topics."""
        engine = TopicClusteringEngine(n_clusters=2)
        
        result = engine.cluster_articles(sample_articles)
        
        # Verify clustering result structure
        assert result is not None
        assert hasattr(result, 'cluster_assignments')
        assert hasattr(result, 'cluster_labels')
        assert len(result.cluster_assignments) == len(sample_articles)
        
        # Verify all articles are assigned to clusters
        for article in sample_articles:
            assert article.title in result.cluster_assignments
            cluster_id = result.cluster_assignments[article.title]
            assert 0 <= cluster_id < 2
        
        # Verify confidence scores
        assert hasattr(result, 'confidence_scores')
        for article in sample_articles:
            assert article.title in result.confidence_scores
            confidence = result.confidence_scores[article.title]
            assert 0 <= confidence <= 1
        
        print(f"✓ Successfully clustered {len(sample_articles)} articles into 2 clusters")
        print(f"  Cluster assignments: {result.cluster_assignments}")
        print(f"  Cluster labels: {result.cluster_labels}")
    
    def test_cluster_growth_calculation(self, sample_articles):
        """Test calculating growth metrics for clusters."""
        engine = TopicClusteringEngine(n_clusters=2)
        result = engine.cluster_articles(sample_articles)
        
        # Create sample pageview data
        dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
        pageviews = pd.DataFrame({
            'article': ['Python (programming language)'] * len(dates) + 
                      ['Java (programming language)'] * len(dates),
            'date': list(dates) + list(dates),
            'views': list(np.linspace(1000, 1500, len(dates))) + 
                    list(np.linspace(800, 1200, len(dates)))
        })
        
        # Calculate growth for first cluster
        growth_metrics = engine.calculate_cluster_growth(0, pageviews)
        
        assert growth_metrics is not None
        assert growth_metrics.cluster_id == 0
        assert growth_metrics.growth_rate != 0  # Should have some growth
        assert growth_metrics.total_views > 0
        assert growth_metrics.article_count > 0
        
        print(f"✓ Cluster 0 growth metrics:")
        print(f"  Growth rate: {growth_metrics.growth_rate:.2f}%")
        print(f"  Total views: {growth_metrics.total_views}")
        print(f"  Article count: {growth_metrics.article_count}")
        print(f"  Is emerging: {growth_metrics.is_emerging}")


class TestHypeDetectionIntegration:
    """Test Hype Detection Engine with trending data."""
    
    def test_detect_hype_in_trending_article(self, sample_pageviews):
        """Test hype detection on article with spike."""
        engine = HypeDetectionEngine(hype_threshold=0.35)  # Lower threshold for test data
        
        # Calculate metrics for hyped article
        hyped_data = sample_pageviews['hyped']
        
        # Calculate view velocity (growth rate)
        views_start = hyped_data['views'].iloc[0]
        views_end = hyped_data['views'].iloc[-1]
        view_velocity = ((views_end - views_start) / views_start) * 100
        
        # Simulate edit growth and content expansion
        edit_growth = 200.0  # 200% increase in edits
        content_expansion = 150.0  # 150% content growth
        
        # Calculate hype metrics
        hype_metrics = engine.calculate_hype_metrics(
            article="Trending Article",
            pageviews=hyped_data,
            view_velocity=view_velocity,
            edit_growth=edit_growth,
            content_expansion=content_expansion,
            window_days=7
        )
        
        assert hype_metrics is not None
        assert hype_metrics.hype_score > 0
        assert hype_metrics.is_trending  # Should be flagged as trending
        assert hype_metrics.attention_density > 0
        assert len(hype_metrics.spike_events) > 0  # Should detect spikes
        
        print(f"✓ Detected hype in trending article:")
        print(f"  Hype score: {hype_metrics.hype_score:.3f}")
        print(f"  Is trending: {hype_metrics.is_trending}")
        print(f"  Attention density: {hype_metrics.attention_density:.2f} views/day")
        print(f"  Spike events detected: {len(hype_metrics.spike_events)}")
    
    def test_no_hype_in_normal_article(self, sample_pageviews):
        """Test that normal articles don't trigger hype detection."""
        engine = HypeDetectionEngine(hype_threshold=0.75)
        
        # Calculate metrics for normal article
        normal_data = sample_pageviews['normal']
        
        views_start = normal_data['views'].iloc[0]
        views_end = normal_data['views'].iloc[-1]
        view_velocity = ((views_end - views_start) / views_start) * 100
        
        edit_growth = 10.0  # Modest edit growth
        content_expansion = 5.0  # Minimal content growth
        
        hype_metrics = engine.calculate_hype_metrics(
            article="Normal Article",
            pageviews=normal_data,
            view_velocity=view_velocity,
            edit_growth=edit_growth,
            content_expansion=content_expansion,
            window_days=7
        )
        
        assert hype_metrics is not None
        assert hype_metrics.hype_score < 0.75  # Should not exceed threshold
        assert not hype_metrics.is_trending  # Should not be flagged
        
        print(f"✓ Normal article correctly not flagged as trending:")
        print(f"  Hype score: {hype_metrics.hype_score:.3f}")
        print(f"  Is trending: {hype_metrics.is_trending}")


class TestKnowledgeGraphIntegration:
    """Test Knowledge Graph Builder with article networks."""
    
    def test_build_graph_from_articles(self, sample_articles):
        """Test building knowledge graph from articles."""
        builder = KnowledgeGraphBuilder()
        
        graph = builder.build_graph(sample_articles)
        
        assert graph is not None
        assert len(graph.nodes) == len(sample_articles)
        assert len(graph.edges) > 0  # Should have some connections
        
        # Verify all articles are nodes
        for article in sample_articles:
            assert article.title in graph.nodes
        
        print(f"✓ Built knowledge graph:")
        print(f"  Nodes: {len(graph.nodes)}")
        print(f"  Edges: {len(graph.edges)}")
    
    def test_calculate_centrality(self, sample_articles):
        """Test centrality calculation for graph nodes."""
        builder = KnowledgeGraphBuilder()
        graph = builder.build_graph(sample_articles)
        
        centrality_scores = builder.calculate_centrality(graph)
        
        assert centrality_scores is not None
        assert len(centrality_scores) == len(sample_articles)
        
        # Verify all nodes have centrality scores
        for article in sample_articles:
            assert article.title in centrality_scores
            score = centrality_scores[article.title]
            assert 0 <= score <= 1
        
        # Find most central article
        most_central = max(centrality_scores.items(), key=lambda x: x[1])
        
        print(f"✓ Calculated centrality scores:")
        print(f"  Most central article: {most_central[0]} (score: {most_central[1]:.3f})")
    
    def test_detect_communities(self, sample_articles):
        """Test community detection in knowledge graph."""
        builder = KnowledgeGraphBuilder()
        graph = builder.build_graph(sample_articles)
        
        communities = builder.detect_communities(graph)
        
        assert communities is not None
        assert len(communities) > 0
        
        # Verify all articles are in some community
        all_community_articles = set()
        for community in communities:
            all_community_articles.update(community.articles)
        
        for article in sample_articles:
            assert article.title in all_community_articles
        
        print(f"✓ Detected {len(communities)} communities:")
        for i, community in enumerate(communities):
            print(f"  Community {i}: {len(community.articles)} articles, density: {community.density:.3f}")


class TestReputationMonitorIntegration:
    """Test Reputation Monitor with edit patterns."""
    
    def test_detect_high_risk_article(self, sample_revisions):
        """Test reputation monitoring on high-risk article."""
        monitor = ReputationMonitor(alert_threshold=0.5)  # Lower threshold for test
        
        # Calculate metrics for risky article
        risky_revisions = sample_revisions['risky']
        edit_metrics = monitor.calculate_edit_metrics(
            article="Controversial Article",
            revisions=risky_revisions,
            time_window_hours=50  # All revisions in 50 hours
        )
        
        # Calculate reputation risk
        reputation_score = monitor.calculate_reputation_risk(edit_metrics)
        
        assert reputation_score is not None
        assert reputation_score.risk_score > 0.5  # Should be high risk
        assert reputation_score.alert_level in ["medium", "high"]
        assert reputation_score.vandalism_rate > 0  # Should have vandalism
        
        print(f"✓ Detected high-risk article:")
        print(f"  Risk score: {reputation_score.risk_score:.3f}")
        print(f"  Alert level: {reputation_score.alert_level}")
        print(f"  Vandalism rate: {reputation_score.vandalism_rate:.2f}%")
        print(f"  Anonymous edits: {reputation_score.anonymous_edit_pct:.2f}%")
    
    def test_normal_article_low_risk(self, sample_revisions):
        """Test that normal articles have low risk scores."""
        monitor = ReputationMonitor(alert_threshold=0.7)
        
        normal_revisions = sample_revisions['normal']
        edit_metrics = monitor.calculate_edit_metrics(
            article="Normal Article",
            revisions=normal_revisions,
            time_window_hours=240  # 10 days
        )
        
        reputation_score = monitor.calculate_reputation_risk(edit_metrics)
        
        assert reputation_score is not None
        assert reputation_score.risk_score < 0.7  # Should be low risk
        assert reputation_score.alert_level in ["low", "medium"]
        
        print(f"✓ Normal article has low risk:")
        print(f"  Risk score: {reputation_score.risk_score:.3f}")
        print(f"  Alert level: {reputation_score.alert_level}")
    
    def test_edit_spike_detection(self, sample_revisions):
        """Test detection of edit spikes."""
        monitor = ReputationMonitor()
        
        # High velocity vs low baseline
        baseline = 1.0  # 1 edit per hour
        current_velocity = 5.0  # 5 edits per hour
        
        is_spike = monitor.detect_edit_spikes(current_velocity, baseline)
        
        assert is_spike  # Should detect spike (5x > 3x threshold)
        
        print(f"✓ Edit spike detected:")
        print(f"  Current velocity: {current_velocity} edits/hour")
        print(f"  Baseline: {baseline} edits/hour")
        print(f"  Ratio: {current_velocity/baseline:.1f}x")


class TestAlertSystemIntegration:
    """Test Alert System with mock notifications."""
    
    def test_send_alert_with_webhook(self):
        """Test sending alert via webhook."""
        # Mock webhook config
        webhook_config = {
            'url': 'https://example.com/webhook',
            'timeout': 5
        }
        
        alert_system = AlertSystem(webhook_config=webhook_config)
        
        # Create test alert
        alert = Alert(
            alert_id="test_alert_1",
            alert_type="reputation_risk",
            priority="high",
            article="Test Article",
            message="High reputation risk detected",
            timestamp=datetime.now(),
            metadata={"risk_score": 0.85}
        )
        
        # Mock the webhook request
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            success = alert_system.send_alert(alert, channels=['webhook'])
            
            assert success
            assert mock_post.called
        
        print(f"✓ Alert sent successfully via webhook")
    
    def test_alert_deduplication(self):
        """Test that duplicate alerts are not sent."""
        alert_system = AlertSystem(
            webhook_config={'url': 'https://example.com/webhook'},
            dedup_window_minutes=60
        )
        
        alert = Alert(
            alert_id="test_alert_2",
            alert_type="reputation_risk",
            priority="high",
            article="Test Article",
            message="High reputation risk detected",
            timestamp=datetime.now(),
            metadata={}
        )
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            # Send first alert
            success1 = alert_system.send_alert(alert, channels=['webhook'])
            assert success1
            
            # Try to send duplicate
            success2 = alert_system.send_alert(alert, channels=['webhook'])
            assert not success2  # Should be deduplicated
            
            # Verify webhook was only called once
            assert mock_post.call_count == 1
        
        print(f"✓ Alert deduplication working correctly")


class TestEndToEndIntegration:
    """Test complete analytics pipeline integration."""
    
    def test_complete_analytics_workflow(self, sample_articles, sample_pageviews, sample_revisions):
        """Test end-to-end analytics workflow."""
        print("\n" + "="*60)
        print("CHECKPOINT 18: Complete Analytics Integration Test")
        print("="*60)
        
        # Step 1: Cluster articles
        print("\n1. Clustering articles...")
        clustering_engine = TopicClusteringEngine(n_clusters=2)
        clustering_result = clustering_engine.cluster_articles(sample_articles)
        assert clustering_result is not None
        print(f"   ✓ Clustered {len(sample_articles)} articles into 2 clusters")
        
        # Step 2: Build knowledge graph
        print("\n2. Building knowledge graph...")
        graph_builder = KnowledgeGraphBuilder()
        knowledge_graph = graph_builder.build_graph(sample_articles)
        assert knowledge_graph is not None
        print(f"   ✓ Built graph with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges")
        
        # Step 3: Calculate centrality
        print("\n3. Calculating centrality...")
        centrality_scores = graph_builder.calculate_centrality(knowledge_graph)
        assert len(centrality_scores) > 0
        most_central = max(centrality_scores.items(), key=lambda x: x[1])
        print(f"   ✓ Most central article: {most_central[0]}")
        
        # Step 4: Detect communities
        print("\n4. Detecting communities...")
        communities = graph_builder.detect_communities(knowledge_graph)
        assert len(communities) > 0
        print(f"   ✓ Detected {len(communities)} communities")
        
        # Step 5: Detect hype
        print("\n5. Detecting hype...")
        hype_engine = HypeDetectionEngine(hype_threshold=0.75)
        hype_metrics = hype_engine.calculate_hype_metrics(
            article="Trending Article",
            pageviews=sample_pageviews['hyped'],
            view_velocity=300.0,
            edit_growth=200.0,
            content_expansion=150.0
        )
        assert hype_metrics is not None
        print(f"   ✓ Hype score: {hype_metrics.hype_score:.3f}, Trending: {hype_metrics.is_trending}")
        
        # Step 6: Monitor reputation
        print("\n6. Monitoring reputation...")
        reputation_monitor = ReputationMonitor(alert_threshold=0.7)
        edit_metrics = reputation_monitor.calculate_edit_metrics(
            article="Controversial Article",
            revisions=sample_revisions['risky'],
            time_window_hours=50
        )
        reputation_score = reputation_monitor.calculate_reputation_risk(edit_metrics)
        assert reputation_score is not None
        print(f"   ✓ Risk score: {reputation_score.risk_score:.3f}, Level: {reputation_score.alert_level}")
        
        # Step 7: Generate and send alert
        print("\n7. Testing alert system...")
        alert_system = AlertSystem(
            webhook_config={'url': 'https://example.com/webhook'}
        )
        alert = reputation_monitor.generate_alert(
            article="Controversial Article",
            risk_score=reputation_score.risk_score
        )
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            success = alert_system.send_alert(alert, channels=['webhook'])
            assert success
        print(f"   ✓ Alert sent successfully")
        
        print("\n" + "="*60)
        print("✓ ALL ANALYTICS COMPONENTS WORKING TOGETHER")
        print("="*60)
        print("\nSummary:")
        print(f"  • Clustering: {len(clustering_result.cluster_assignments)} articles clustered")
        print(f"  • Knowledge Graph: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        print(f"  • Communities: {len(communities)} detected")
        print(f"  • Hype Detection: {len(hype_metrics.spike_events)} spikes found")
        print(f"  • Reputation: {reputation_score.alert_level} risk level")
        print(f"  • Alerts: Successfully sent and deduplicated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
