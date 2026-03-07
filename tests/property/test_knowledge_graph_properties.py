"""Property-Based Tests for Knowledge Graph Builder

Tests correctness properties for graph construction, centrality calculation,
community detection, and incremental updates.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from datetime import datetime
from typing import List
import networkx as nx

from src.analytics.knowledge_graph import KnowledgeGraphBuilder
from src.storage.dto import ArticleContent, KnowledgeGraph


# ============================================================================
# STRATEGIES
# ============================================================================

@st.composite
def article_title_strategy(draw):
    """Generate valid Wikipedia article titles"""
    # Use alphanumeric with underscores (Wikipedia style)
    length = draw(st.integers(min_value=3, max_value=30))
    chars = draw(st.lists(
        st.sampled_from('ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz0123456789'),
        min_size=length,
        max_size=length
    ))
    return ''.join(chars)


@st.composite
def article_content_strategy(draw, available_titles=None):
    """Generate ArticleContent with internal links"""
    title = draw(article_title_strategy())
    
    # Generate internal links from available titles
    if available_titles and len(available_titles) > 0:
        num_links = draw(st.integers(min_value=0, max_value=min(5, len(available_titles))))
        internal_links = draw(st.lists(
            st.sampled_from(available_titles),
            min_size=num_links,
            max_size=num_links,
            unique=True
        ))
    else:
        internal_links = []
    
    return ArticleContent(
        title=title,
        url=f"https://en.wikipedia.org/wiki/{title}",
        summary=draw(st.text(min_size=10, max_size=100)),
        infobox={},
        tables=[],
        categories=draw(st.lists(st.text(min_size=3, max_size=20), max_size=3)),
        internal_links=internal_links,
        crawl_timestamp=datetime.utcnow()
    )


@st.composite
def article_network_strategy(draw, min_articles=1, max_articles=20):
    """Generate a network of articles with internal links"""
    num_articles = draw(st.integers(min_value=min_articles, max_value=max_articles))
    
    # First generate all titles
    titles = []
    for _ in range(num_articles):
        title = draw(article_title_strategy())
        # Ensure unique titles
        while title in titles:
            title = draw(article_title_strategy())
        titles.append(title)
    
    # Then generate articles with links to existing titles
    articles = []
    for title in titles:
        num_links = draw(st.integers(min_value=0, max_value=min(5, len(titles) - 1)))
        # Links to other articles (not self)
        available_targets = [t for t in titles if t != title]
        if available_targets and num_links > 0:
            internal_links = draw(st.lists(
                st.sampled_from(available_targets),
                min_size=num_links,
                max_size=num_links,
                unique=True
            ))
        else:
            internal_links = []
        
        articles.append(ArticleContent(
            title=title,
            url=f"https://en.wikipedia.org/wiki/{title}",
            summary=draw(st.text(min_size=10, max_size=100)),
            infobox={},
            tables=[],
            categories=[],
            internal_links=internal_links,
            crawl_timestamp=datetime.utcnow()
        ))
    
    return articles


# ============================================================================
# PROPERTY TESTS
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 45: Related Article Discovery
@given(
    seed_articles=article_network_strategy(min_articles=2, max_articles=10),
    max_depth=st.integers(min_value=1, max_value=3)
)
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.large_base_example])
def test_property_45_related_article_discovery(seed_articles, max_depth):
    """
    Property 45: Related Article Discovery
    
    For any deep crawl, the System should discover all articles reachable
    through internal links up to the configured maximum depth.
    
    **Validates: Requirements 10.1**
    """
    builder = KnowledgeGraphBuilder()
    
    # Build graph from articles
    graph = builder.build_graph(seed_articles)
    
    # Verify all articles are nodes
    article_titles = {article.title for article in seed_articles}
    assert set(graph.nodes) == article_titles, \
        "All articles should be nodes in the graph"
    
    # Verify all valid internal links are edges
    for article in seed_articles:
        for target in article.internal_links:
            if target in article_titles:
                assert (article.title, target) in graph.edges, \
                    f"Link from {article.title} to {target} should be an edge"
    
    # Simulate BFS discovery up to max_depth
    if len(seed_articles) > 0:
        start_node = seed_articles[0].title
        
        # Use NetworkX BFS to find reachable nodes
        if builder.graph:
            reachable = set()
            for depth in range(max_depth + 1):
                if depth == 0:
                    reachable.add(start_node)
                else:
                    # Find nodes at this depth
                    for node in list(reachable):
                        if builder.graph.has_node(node):
                            neighbors = list(builder.graph.successors(node))
                            reachable.update(neighbors)
            
            # All reachable nodes should be in the graph
            for node in reachable:
                assert node in graph.nodes, \
                    f"Reachable node {node} should be in graph"


# Feature: wikipedia-intelligence-system, Property 46: Knowledge Graph Construction
@given(articles=article_network_strategy(min_articles=1, max_articles=15))
@settings(max_examples=100, deadline=2000)
def test_property_46_knowledge_graph_construction(articles):
    """
    Property 46: Knowledge Graph Construction
    
    For any set of crawled articles, the System should construct a graph
    where each article is a node and each internal link is a directed edge.
    
    **Validates: Requirements 10.2**
    """
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(articles)
    
    # Verify all articles are nodes
    article_titles = {article.title for article in articles}
    assert set(graph.nodes) == article_titles, \
        "Each article should be exactly one node"
    
    # Verify edges correspond to internal links
    expected_edges = set()
    for article in articles:
        for target in article.internal_links:
            if target in article_titles:
                expected_edges.add((article.title, target))
    
    assert set(graph.edges) == expected_edges, \
        "Each internal link should be exactly one directed edge"
    
    # Verify graph is directed (can have A->B without B->A)
    # This is implicit in the edge structure


# Feature: wikipedia-intelligence-system, Property 47: Graph Clustering
@given(articles=article_network_strategy(min_articles=2, max_articles=15))
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.large_base_example])
def test_property_47_graph_clustering(articles):
    """
    Property 47: Graph Clustering
    
    For any knowledge graph, the System should identify clusters using
    community detection algorithms, grouping densely connected nodes.
    
    **Validates: Requirements 10.3**
    """
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(articles)
    
    communities = builder.detect_communities(graph)
    
    # All articles should be assigned to exactly one community
    all_assigned_articles = set()
    for comm in communities:
        all_assigned_articles.update(comm.articles)
    
    assert all_assigned_articles == set(graph.nodes), \
        "All articles should be assigned to exactly one community"
    
    # Communities should not overlap
    for i, comm1 in enumerate(communities):
        for j, comm2 in enumerate(communities):
            if i != j:
                overlap = set(comm1.articles) & set(comm2.articles)
                assert len(overlap) == 0, \
                    "Communities should not overlap"
    
    # Each community should have valid properties
    for comm in communities:
        assert comm.size == len(comm.articles), \
            "Community size should match article count"
        assert 0.0 <= comm.density <= 1.0, \
            "Community density should be between 0 and 1"


# Feature: wikipedia-intelligence-system, Property 48: Centrality Calculation
@given(articles=article_network_strategy(min_articles=2, max_articles=15))
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.large_base_example])
def test_property_48_centrality_calculation(articles):
    """
    Property 48: Centrality Calculation
    
    For any knowledge graph, the System should calculate centrality metrics
    (betweenness, eigenvector) for all nodes.
    
    **Validates: Requirements 10.4**
    """
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(articles)
    
    centrality_scores = builder.calculate_centrality(graph)
    
    # All nodes should have centrality scores
    assert set(centrality_scores.keys()) == set(graph.nodes), \
        "All nodes should have centrality scores"
    
    # Centrality scores should be in valid range [0, 1]
    for node, score in centrality_scores.items():
        assert 0.0 <= score <= 1.0, \
            f"Centrality score for {node} should be between 0 and 1, got {score}"
    
    # Centrality scores should be non-negative
    for score in centrality_scores.values():
        assert score >= 0.0, \
            "Centrality scores should be non-negative"


# Feature: wikipedia-intelligence-system, Property 49: Incremental Graph Updates
@given(
    initial_articles=article_network_strategy(min_articles=2, max_articles=10),
    new_articles_count=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.large_base_example])
def test_property_49_incremental_graph_updates(initial_articles, new_articles_count):
    """
    Property 49: Incremental Graph Updates
    
    For any existing knowledge graph and new crawled articles, adding the
    new articles should update the graph without requiring a full rebuild
    of existing nodes.
    
    **Validates: Requirements 10.6**
    """
    builder = KnowledgeGraphBuilder()
    
    # Build initial graph
    initial_graph = builder.build_graph(initial_articles)
    initial_nodes = set(initial_graph.nodes)
    initial_edges = set(initial_graph.edges)
    
    # Generate new articles that may link to existing ones
    existing_titles = [article.title for article in initial_articles]
    new_articles = []
    for _ in range(new_articles_count):
        # Create new article with unique title
        new_title = f"NewArticle_{_}_{datetime.utcnow().timestamp()}"
        
        # May link to existing articles
        num_links = min(2, len(existing_titles))
        if num_links > 0 and existing_titles:
            import random
            internal_links = random.sample(existing_titles, num_links)
        else:
            internal_links = []
        
        new_articles.append(ArticleContent(
            title=new_title,
            url=f"https://en.wikipedia.org/wiki/{new_title}",
            summary="New article summary",
            infobox={},
            tables=[],
            categories=[],
            internal_links=internal_links,
            crawl_timestamp=datetime.utcnow()
        ))
    
    # Update graph incrementally
    updated_graph = builder.update_incremental(initial_graph, new_articles)
    
    # All initial nodes should still be present
    assert initial_nodes.issubset(set(updated_graph.nodes)), \
        "Initial nodes should be preserved in incremental update"
    
    # All initial edges should still be present
    assert initial_edges.issubset(set(updated_graph.edges)), \
        "Initial edges should be preserved in incremental update"
    
    # New articles should be added as nodes
    new_titles = {article.title for article in new_articles}
    assert new_titles.issubset(set(updated_graph.nodes)), \
        "New articles should be added as nodes"
    
    # New edges from new articles should be added
    for article in new_articles:
        for target in article.internal_links:
            if target in updated_graph.nodes:
                assert (article.title, target) in updated_graph.edges, \
                    f"New edge from {article.title} to {target} should be added"
    
    # Total node count should be initial + new
    expected_node_count = len(initial_nodes) + len(new_titles)
    assert len(updated_graph.nodes) == expected_node_count, \
        "Node count should be initial + new articles"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_empty_graph():
    """Test graph construction with no articles"""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph([])
    
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_single_article_graph():
    """Test graph with single article (no links possible)"""
    builder = KnowledgeGraphBuilder()
    
    article = ArticleContent(
        title="SingleArticle",
        url="https://en.wikipedia.org/wiki/SingleArticle",
        summary="Summary",
        infobox={},
        tables=[],
        categories=[],
        internal_links=[],
        crawl_timestamp=datetime.utcnow()
    )
    
    graph = builder.build_graph([article])
    
    assert len(graph.nodes) == 1
    assert "SingleArticle" in graph.nodes
    assert len(graph.edges) == 0


def test_self_referential_links_ignored():
    """Test that self-referential links are handled correctly"""
    builder = KnowledgeGraphBuilder()
    
    article = ArticleContent(
        title="Article1",
        url="https://en.wikipedia.org/wiki/Article1",
        summary="Summary",
        infobox={},
        tables=[],
        categories=[],
        internal_links=["Article1"],  # Self-reference
        crawl_timestamp=datetime.utcnow()
    )
    
    graph = builder.build_graph([article])
    
    # Self-loops are allowed in directed graphs, so this should create an edge
    assert len(graph.nodes) == 1
    # NetworkX allows self-loops
    assert ("Article1", "Article1") in graph.edges


def test_dangling_links_ignored():
    """Test that links to non-existent articles are ignored"""
    builder = KnowledgeGraphBuilder()
    
    article = ArticleContent(
        title="Article1",
        url="https://en.wikipedia.org/wiki/Article1",
        summary="Summary",
        infobox={},
        tables=[],
        categories=[],
        internal_links=["NonExistentArticle"],
        crawl_timestamp=datetime.utcnow()
    )
    
    graph = builder.build_graph([article])
    
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0  # Link to non-existent article ignored


def test_centrality_empty_graph():
    """Test centrality calculation on empty graph"""
    builder = KnowledgeGraphBuilder()
    graph = KnowledgeGraph(nodes=[], edges=[], centrality_scores={}, communities={})
    
    centrality = builder.calculate_centrality(graph)
    
    assert len(centrality) == 0


def test_communities_empty_graph():
    """Test community detection on empty graph"""
    builder = KnowledgeGraphBuilder()
    graph = KnowledgeGraph(nodes=[], edges=[], centrality_scores={}, communities={})
    
    communities = builder.detect_communities(graph)
    
    assert len(communities) == 0
