"""Knowledge Graph Builder

Constructs and analyzes domain knowledge graphs from article relationships.
Uses NetworkX for graph operations and community detection algorithms.
"""
from typing import List, Dict, Optional
import networkx as nx
from networkx.algorithms import community
from src.storage.dto import ArticleContent, KnowledgeGraph, Community


class KnowledgeGraphBuilder:
    """Builds and analyzes knowledge graphs from Wikipedia articles
    
    Creates graphs where articles are nodes and internal links are edges.
    Provides centrality analysis and community detection.
    """
    
    def __init__(self):
        """Initialize the knowledge graph builder"""
        self.graph: Optional[nx.DiGraph] = None
    
    def build_graph(self, articles: List[ArticleContent]) -> KnowledgeGraph:
        """Construct graph with articles as nodes, links as edges
        
        Creates a directed graph where:
        - Each article is a node (identified by title)
        - Each internal link is a directed edge (source -> target)
        
        Args:
            articles: List of crawled article content with internal links
            
        Returns:
            KnowledgeGraph with nodes, edges, and initial structure
            
        Validates: Requirements 10.1, 10.2
        """
        # Create directed graph
        self.graph = nx.DiGraph()
        
        # Add all articles as nodes first
        for article in articles:
            self.graph.add_node(article.title)
        
        # Add edges from internal links
        for article in articles:
            source = article.title
            for target in article.internal_links:
                # Only add edge if target is also in our article set
                if target in self.graph.nodes:
                    self.graph.add_edge(source, target)
        
        # Extract nodes and edges for return
        nodes = list(self.graph.nodes())
        edges = list(self.graph.edges())
        
        return KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            centrality_scores={},
            communities={}
        )
    
    def calculate_centrality(self, graph: KnowledgeGraph) -> Dict[str, float]:
        """Calculate centrality metrics for all nodes
        
        Computes both betweenness and eigenvector centrality.
        Returns the average of both metrics for each node.
        
        Args:
            graph: Knowledge graph to analyze
            
        Returns:
            Dictionary mapping article titles to centrality scores (0-1)
            
        Validates: Requirements 10.4
        """
        if self.graph is None:
            # Reconstruct graph from KnowledgeGraph object
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(graph.nodes)
            self.graph.add_edges_from(graph.edges)
        
        centrality_scores = {}
        
        # Calculate betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(self.graph)
        except:
            # If graph is empty or has issues, return zeros
            betweenness = {node: 0.0 for node in self.graph.nodes()}
        
        # Calculate eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)
        except:
            # If convergence fails or graph has issues, return zeros
            eigenvector = {node: 0.0 for node in self.graph.nodes()}
        
        # Average the two centrality measures
        for node in self.graph.nodes():
            betweenness_score = betweenness.get(node, 0.0)
            eigenvector_score = eigenvector.get(node, 0.0)
            centrality_scores[node] = (betweenness_score + eigenvector_score) / 2.0
        
        return centrality_scores
    
    def detect_communities(self, graph: KnowledgeGraph) -> List[Community]:
        """Identify clusters representing industries/competitors
        
        Uses the Louvain algorithm for community detection.
        Communities represent groups of densely connected articles.
        
        Args:
            graph: Knowledge graph to analyze
            
        Returns:
            List of detected communities with member articles
            
        Validates: Requirements 10.3
        """
        if self.graph is None:
            # Reconstruct graph from KnowledgeGraph object
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(graph.nodes)
            self.graph.add_edges_from(graph.edges)
        
        # Convert to undirected for community detection
        undirected_graph = self.graph.to_undirected()
        
        # Handle empty or single-node graphs
        if len(undirected_graph.nodes()) == 0:
            return []
        
        if len(undirected_graph.nodes()) == 1:
            node = list(undirected_graph.nodes())[0]
            return [Community(
                community_id=0,
                articles=[node],
                size=1,
                density=0.0
            )]
        
        # Apply Louvain community detection
        try:
            communities_generator = community.louvain_communities(undirected_graph)
            detected_communities = list(communities_generator)
        except:
            # If algorithm fails, treat each node as its own community
            detected_communities = [{node} for node in undirected_graph.nodes()]
        
        # Convert to Community objects
        result = []
        for idx, comm_set in enumerate(detected_communities):
            articles = list(comm_set)
            size = len(articles)
            
            # Calculate density (edges within community / possible edges)
            subgraph = undirected_graph.subgraph(articles)
            num_edges = subgraph.number_of_edges()
            possible_edges = size * (size - 1) / 2 if size > 1 else 0
            density = num_edges / possible_edges if possible_edges > 0 else 0.0
            
            result.append(Community(
                community_id=idx,
                articles=articles,
                size=size,
                density=density
            ))
        
        return result
    
    def update_incremental(
        self, 
        graph: KnowledgeGraph, 
        new_articles: List[ArticleContent]
    ) -> KnowledgeGraph:
        """Update graph with newly crawled articles
        
        Adds new articles and their links without rebuilding existing structure.
        This is more efficient than rebuilding the entire graph.
        
        Args:
            graph: Existing knowledge graph
            new_articles: Newly crawled articles to add
            
        Returns:
            Updated knowledge graph with new articles integrated
            
        Validates: Requirements 10.6
        """
        if self.graph is None:
            # Reconstruct graph from KnowledgeGraph object
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(graph.nodes)
            self.graph.add_edges_from(graph.edges)
        
        # Add new articles as nodes
        for article in new_articles:
            if article.title not in self.graph.nodes:
                self.graph.add_node(article.title)
        
        # Add edges from new articles
        for article in new_articles:
            source = article.title
            for target in article.internal_links:
                # Add edge if target exists in graph (old or new)
                if target in self.graph.nodes:
                    # Only add if edge doesn't already exist
                    if not self.graph.has_edge(source, target):
                        self.graph.add_edge(source, target)
        
        # Also check if new articles are targets of existing articles' links
        # This handles the case where an existing article links to a newly added article
        for article in new_articles:
            new_title = article.title
            # Check all existing nodes for links to this new article
            for existing_node in list(self.graph.nodes()):
                if existing_node != new_title:
                    # We need to check if existing_node should link to new_title
                    # This requires the original article data, which we don't have here
                    # So we only handle forward links from new articles
                    pass
        
        # Extract updated nodes and edges
        nodes = list(self.graph.nodes())
        edges = list(self.graph.edges())
        
        return KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            centrality_scores=graph.centrality_scores,  # Preserve existing scores
            communities=graph.communities  # Preserve existing communities
        )
