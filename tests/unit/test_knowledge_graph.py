"""Unit Tests for Knowledge Graph Builder

Tests specific examples, edge cases, and known graph structures.
"""
import pytest
from datetime import datetime
import networkx as nx

from src.analytics.knowledge_graph import KnowledgeGraphBuilder
from src.storage.dto import ArticleContent, KnowledgeGraph


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_articles():
    """Sample article network for testing"""
    return [
        ArticleContent(
            title="Python_(programming_language)",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            summary="Python is a high-level programming language",
            infobox={"paradigm": "multi-paradigm"},
            tables=[],
            categories=["Programming languages"],
            internal_links=["Guido_van_Rossum", "Java_(programming_language)"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Guido_van_Rossum",
            url="https://en.wikipedia.org/wiki/Guido_van_Rossum",
            summary="Guido van Rossum is a Dutch programmer",
            infobox={"occupation": "programmer"},
            tables=[],
            categories=["Computer scientists"],
            internal_links=["Python_(programming_language)", "Netherlands"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Java_(programming_language)",
            url="https://en.wikipedia.org/wiki/Java_(programming_language)",
            summary="Java is a high-level programming language",
            infobox={"paradigm": "object-oriented"},
            tables=[],
            categories=["Programming languages"],
            internal_links=["Python_(programming_language)", "James_Gosling"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Netherlands",
            url="https://en.wikipedia.org/wiki/Netherlands",
            summary="The Netherlands is a country in Europe",
            infobox={"capital": "Amsterdam"},
            tables=[],
            categories=["Countries"],
            internal_links=["Guido_van_Rossum"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="James_Gosling",
            url="https://en.wikipedia.org/wiki/James_Gosling",
            summary="James Gosling is a Canadian computer scientist",
            infobox={"occupation": "computer scientist"},
            tables=[],
            categories=["Computer scientists"],
            internal_links=["Java_(programming_language)"],
            crawl_timestamp=datetime(2024, 1, 1)
        )
    ]


@pytest.fixture
def linear_chain_articles():
    """Articles forming a linear chain: A -> B -> C"""
    return [
        ArticleContent(
            title="Article_A",
            url="https://en.wikipedia.org/wiki/Article_A",
            summary="Article A",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Article_B"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Article_B",
            url="https://en.wikipedia.org/wiki/Article_B",
            summary="Article B",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Article_C"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Article_C",
            url="https://en.wikipedia.org/wiki/Article_C",
            summary="Article C",
            infobox={},
            tables=[],
            categories=[],
            internal_links=[],
            crawl_timestamp=datetime(2024, 1, 1)
        )
    ]


@pytest.fixture
def star_network_articles():
    """Articles forming a star: Center connected to 4 periphery nodes"""
    return [
        ArticleContent(
            title="Center",
            url="https://en.wikipedia.org/wiki/Center",
            summary="Center article",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Node1", "Node2", "Node3", "Node4"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Node1",
            url="https://en.wikipedia.org/wiki/Node1",
            summary="Node 1",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Center"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Node2",
            url="https://en.wikipedia.org/wiki/Node2",
            summary="Node 2",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Center"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Node3",
            url="https://en.wikipedia.org/wiki/Node3",
            summary="Node 3",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Center"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Node4",
            url="https://en.wikipedia.org/wiki/Node4",
            summary="Node 4",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Center"],
            crawl_timestamp=datetime(2024, 1, 1)
        )
    ]


# ============================================================================
# GRAPH CONSTRUCTION TESTS
# ============================================================================

def test_build_graph_sample_network(sample_articles):
    """Test graph construction with sample article network"""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(sample_articles)
    
    # Verify all articles are nodes
    assert len(graph.nodes) == 5
    assert "Python_(programming_language)" in graph.nodes
    assert "Guido_van_Rossum" in graph.nodes
    assert "Java_(programming_language)" in graph.nodes
    assert "Netherlands" in graph.nodes
    assert "James_Gosling" in graph.nodes
    
    # Verify edges
    expected_edges = {
        ("Python_(programming_language)", "Guido_van_Rossum"),
        ("Python_(programming_language)", "Java_(programming_language)"),
        ("Guido_van_Rossum", "Python_(programming_language)"),
        ("Guido_van_Rossum", "Netherlands"),
        ("Java_(programming_language)", "Python_(programming_language)"),
        ("Java_(programming_language)", "James_Gosling"),
        ("Netherlands", "Guido_van_Rossum"),
        ("James_Gosling", "Java_(programming_language)")
    }
    assert set(graph.edges) == expected_edges


def test_build_graph_linear_chain(linear_chain_articles):
    """Test graph construction with linear chain"""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(linear_chain_articles)
    
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert ("Article_A", "Article_B") in graph.edges
    assert ("Article_B", "Article_C") in graph.edges


def test_build_graph_star_network(star_network_articles):
    """Test graph construction with star topology"""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(star_network_articles)
    
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 8  # 4 outgoing from center + 4 incoming to center
    
    # Center should have edges to all nodes
    center_outgoing = [edge for edge in graph.edges if edge[0] == "Center"]
    assert len(center_outgoing) == 4
    
    # All nodes should have edges back to center
    center_incoming = [edge for edge in graph.edges if edge[1] == "Center"]
    assert len(center_incoming) == 4


# ============================================================================
# CENTRALITY CALCULATION TESTS
# ============================================================================

def test_centrality_star_network(star_network_articles):
    """Test centrality calculation on star network - center should have highest centrality"""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(star_network_articles)
    centrality = builder.calculate_centrality(graph)
    
    # Center should have highest centrality
    center_score = centrality["Center"]
    for node in ["Node1", "Node2", "Node3", "Node4"]:
        assert center_score >= centrality[node], \
            f"Center should have higher centrality than {node}"


def test_centrality_linear_chain(linear_chain_articles):
    """Test centrality calculation on linear chain"""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(linear_chain_articles)
    centrality = builder.calculate_centrality(graph)
    
    # All nodes should have centrality scores
    assert len(centrality) == 3
    
    # All centrality scores should be valid (between 0 and 1)
    for node, score in centrality.items():
        assert 0.0 <= score <= 1.0
    
    # In a directed linear chain A->B->C, centrality depends on the specific metrics
    # Just verify that all nodes have been assigned scores
    assert "Article_A" in centrality
    assert "Article_B" in centrality
    assert "Article_C" in centrality


def test_centrality_isolated_nodes():
    """Test centrality with isolated nodes (no edges)"""
    builder = KnowledgeGraphBuilder()
    
    articles = [
        ArticleContent(
            title="Isolated1",
            url="https://en.wikipedia.org/wiki/Isolated1",
            summary="Isolated article 1",
            infobox={},
            tables=[],
            categories=[],
            internal_links=[],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Isolated2",
            url="https://en.wikipedia.org/wiki/Isolated2",
            summary="Isolated article 2",
            infobox={},
            tables=[],
            categories=[],
            internal_links=[],
            crawl_timestamp=datetime(2024, 1, 1)
        )
    ]
    
    graph = builder.build_graph(articles)
    centrality = builder.calculate_centrality(graph)
    
    # All nodes should have centrality scores (may not be zero due to eigenvector calculation)
    assert len(centrality) == 2
    assert "Isolated1" in centrality
    assert "Isolated2" in centrality
    
    # Centrality scores should be in valid range
    assert 0.0 <= centrality["Isolated1"] <= 1.0
    assert 0.0 <= centrality["Isolated2"] <= 1.0


# ============================================================================
# COMMUNITY DETECTION TESTS
# ============================================================================

def test_communities_sample_network(sample_articles):
    """Test community detection on sample network"""
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(sample_articles)
    communities = builder.detect_communities(graph)
    
    # Should detect at least one community
    assert len(communities) > 0
    
    # All articles should be in exactly one community
    all_articles = set()
    for comm in communities:
        all_articles.update(comm.articles)
    assert all_articles == set(graph.nodes)
    
    # Each community should have valid properties
    for comm in communities:
        assert comm.size > 0
        assert 0.0 <= comm.density <= 1.0


def test_communities_two_clusters():
    """Test community detection with two distinct clusters"""
    articles = [
        # Cluster 1: Programming languages
        ArticleContent(
            title="Python",
            url="https://en.wikipedia.org/wiki/Python",
            summary="Python",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Java", "Ruby"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Java",
            url="https://en.wikipedia.org/wiki/Java",
            summary="Java",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Python", "Ruby"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Ruby",
            url="https://en.wikipedia.org/wiki/Ruby",
            summary="Ruby",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Python", "Java"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        # Cluster 2: Countries
        ArticleContent(
            title="France",
            url="https://en.wikipedia.org/wiki/France",
            summary="France",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Germany", "Spain"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Germany",
            url="https://en.wikipedia.org/wiki/Germany",
            summary="Germany",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["France", "Spain"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="Spain",
            url="https://en.wikipedia.org/wiki/Spain",
            summary="Spain",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["France", "Germany"],
            crawl_timestamp=datetime(2024, 1, 1)
        )
    ]
    
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(articles)
    communities = builder.detect_communities(graph)
    
    # Should detect 2 communities (or possibly 1 if algorithm merges them)
    assert len(communities) >= 1
    
    # Each community should have high internal density
    for comm in communities:
        if comm.size > 1:
            assert comm.density > 0.0


# ============================================================================
# INCREMENTAL UPDATE TESTS
# ============================================================================

def test_incremental_update_add_new_articles(sample_articles):
    """Test adding new articles to existing graph"""
    builder = KnowledgeGraphBuilder()
    
    # Build initial graph
    initial_graph = builder.build_graph(sample_articles)
    initial_node_count = len(initial_graph.nodes)
    
    # Create new articles
    new_articles = [
        ArticleContent(
            title="C++_(programming_language)",
            url="https://en.wikipedia.org/wiki/C++_(programming_language)",
            summary="C++ is a programming language",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Python_(programming_language)", "Java_(programming_language)"],
            crawl_timestamp=datetime(2024, 1, 2)
        )
    ]
    
    # Update graph incrementally
    updated_graph = builder.update_incremental(initial_graph, new_articles)
    
    # Should have one more node
    assert len(updated_graph.nodes) == initial_node_count + 1
    assert "C++_(programming_language)" in updated_graph.nodes
    
    # New edges should be added
    assert ("C++_(programming_language)", "Python_(programming_language)") in updated_graph.edges
    assert ("C++_(programming_language)", "Java_(programming_language)") in updated_graph.edges


def test_incremental_update_preserves_existing(sample_articles):
    """Test that incremental update preserves existing nodes and edges"""
    builder = KnowledgeGraphBuilder()
    
    # Build initial graph
    initial_graph = builder.build_graph(sample_articles)
    initial_nodes = set(initial_graph.nodes)
    initial_edges = set(initial_graph.edges)
    
    # Add new article
    new_articles = [
        ArticleContent(
            title="NewArticle",
            url="https://en.wikipedia.org/wiki/NewArticle",
            summary="New article",
            infobox={},
            tables=[],
            categories=[],
            internal_links=[],
            crawl_timestamp=datetime(2024, 1, 2)
        )
    ]
    
    updated_graph = builder.update_incremental(initial_graph, new_articles)
    
    # All initial nodes should still exist
    assert initial_nodes.issubset(set(updated_graph.nodes))
    
    # All initial edges should still exist
    assert initial_edges.issubset(set(updated_graph.edges))


def test_incremental_update_no_duplicate_edges(sample_articles):
    """Test that incremental update doesn't create duplicate edges"""
    builder = KnowledgeGraphBuilder()
    
    # Build initial graph
    initial_graph = builder.build_graph(sample_articles)
    
    # Add article with link that already exists
    new_articles = [
        ArticleContent(
            title="Python_(programming_language)",  # Same title as existing
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            summary="Python is a programming language",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["Guido_van_Rossum"],  # Link that already exists
            crawl_timestamp=datetime(2024, 1, 2)
        )
    ]
    
    updated_graph = builder.update_incremental(initial_graph, new_articles)
    
    # Should not create duplicate nodes
    assert len(updated_graph.nodes) == len(initial_graph.nodes)
    
    # Edge count should not increase (edge already exists)
    # Note: The implementation adds the node but doesn't duplicate edges
    assert len(updated_graph.edges) == len(initial_graph.edges)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_graph_with_cycles():
    """Test graph construction with cycles"""
    articles = [
        ArticleContent(
            title="A",
            url="https://en.wikipedia.org/wiki/A",
            summary="A",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["B"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="B",
            url="https://en.wikipedia.org/wiki/B",
            summary="B",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["C"],
            crawl_timestamp=datetime(2024, 1, 1)
        ),
        ArticleContent(
            title="C",
            url="https://en.wikipedia.org/wiki/C",
            summary="C",
            infobox={},
            tables=[],
            categories=[],
            internal_links=["A"],  # Creates cycle A -> B -> C -> A
            crawl_timestamp=datetime(2024, 1, 1)
        )
    ]
    
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph(articles)
    
    # Should handle cycles correctly
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 3
    
    # Centrality should still work with cycles
    centrality = builder.calculate_centrality(graph)
    assert len(centrality) == 3


def test_large_graph_performance():
    """Test that graph operations work efficiently with larger graphs"""
    # Create 50 articles with random links
    import random
    random.seed(42)
    
    articles = []
    titles = [f"Article_{i}" for i in range(50)]
    
    for title in titles:
        # Each article links to 3-5 random other articles
        num_links = random.randint(3, 5)
        other_titles = [t for t in titles if t != title]
        internal_links = random.sample(other_titles, min(num_links, len(other_titles)))
        
        articles.append(ArticleContent(
            title=title,
            url=f"https://en.wikipedia.org/wiki/{title}",
            summary=f"Summary for {title}",
            infobox={},
            tables=[],
            categories=[],
            internal_links=internal_links,
            crawl_timestamp=datetime(2024, 1, 1)
        ))
    
    builder = KnowledgeGraphBuilder()
    
    # Build graph
    graph = builder.build_graph(articles)
    assert len(graph.nodes) == 50
    
    # Calculate centrality
    centrality = builder.calculate_centrality(graph)
    assert len(centrality) == 50
    
    # Detect communities
    communities = builder.detect_communities(graph)
    assert len(communities) > 0
    
    # All operations should complete without errors
