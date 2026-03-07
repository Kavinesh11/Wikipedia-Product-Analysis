"""Topic Clustering Engine

Clusters related articles by industry and calculates comparative metrics.
Uses TF-IDF vectorization and K-means clustering to group articles by topic.
"""
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import logging

from src.storage.dto import (
    ArticleContent,
    ClusteringResult,
    GrowthMetrics,
    ComparisonResult
)

logger = logging.getLogger(__name__)


class TopicClusteringEngine:
    """Engine for clustering articles by topic and analyzing cluster growth
    
    Uses TF-IDF vectorization to convert article content into feature vectors,
    then applies K-means clustering to group similar articles. Calculates
    growth metrics and CAGR for each cluster.
    
    Attributes:
        n_clusters: Number of clusters to create
        vectorizer: TF-IDF vectorizer for text processing
        model: K-means clustering model
    """
    
    def __init__(self, n_clusters: int = 20):
        """Initialize clustering engine
        
        Args:
            n_clusters: Number of clusters to create (default: 20)
        """
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.model: Optional[KMeans] = None
        self._cluster_labels: Dict[int, str] = {}
        
        logger.info(f"Initialized TopicClusteringEngine with {n_clusters} clusters")
    
    def cluster_articles(
        self,
        articles: List[ArticleContent]
    ) -> ClusteringResult:
        """Cluster articles using TF-IDF and K-means
        
        Converts article content (summary + categories) into TF-IDF vectors,
        then applies K-means clustering to group similar articles.
        
        Args:
            articles: List of article content objects
            
        Returns:
            ClusteringResult with cluster assignments and confidence scores
            
        Raises:
            ValueError: If articles list is empty or has fewer than n_clusters articles
        """
        if not articles:
            raise ValueError("articles list cannot be empty")
        
        if len(articles) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} articles for {self.n_clusters} clusters, "
                f"got {len(articles)}"
            )
        
        logger.info(f"Clustering {len(articles)} articles into {self.n_clusters} clusters")
        
        # Combine summary and categories for each article
        texts = []
        article_titles = []
        for article in articles:
            # Combine summary with categories for richer context
            text = article.summary + " " + " ".join(article.categories)
            texts.append(text)
            article_titles.append(article.title)
        
        # Vectorize text using TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Apply K-means clustering
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        cluster_labels = self.model.fit_predict(tfidf_matrix)
        
        # Calculate confidence scores (distance to cluster center)
        distances = self.model.transform(tfidf_matrix)
        confidence_scores = {}
        for i, title in enumerate(article_titles):
            # Confidence is inverse of normalized distance to assigned cluster
            cluster_id = cluster_labels[i]
            distance_to_center = distances[i][cluster_id]
            # Normalize: closer = higher confidence (0-1 range)
            max_distance = np.max(distances[i])
            if max_distance > 0:
                confidence = 1.0 - (distance_to_center / max_distance)
            else:
                confidence = 1.0
            confidence_scores[title] = float(confidence)
        
        # Create cluster assignments
        cluster_assignments = {
            title: int(cluster_labels[i])
            for i, title in enumerate(article_titles)
        }
        
        # Store for later use
        self._cluster_assignments = cluster_assignments
        
        # Generate cluster labels from top TF-IDF terms
        self._generate_cluster_labels(tfidf_matrix, cluster_labels)
        
        result = ClusteringResult(
            cluster_assignments=cluster_assignments,
            cluster_labels=self._cluster_labels.copy(),
            confidence_scores=confidence_scores,
            n_clusters=self.n_clusters
        )
        
        logger.info(f"Successfully clustered {len(articles)} articles")
        return result
    
    def _generate_cluster_labels(
        self,
        tfidf_matrix,
        cluster_labels: np.ndarray
    ) -> None:
        """Generate human-readable labels for clusters based on top terms
        
        Args:
            tfidf_matrix: TF-IDF matrix of article texts
            cluster_labels: Cluster assignment for each article
        """
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in range(self.n_clusters):
            # Get indices of articles in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                self._cluster_labels[cluster_id] = f"Cluster_{cluster_id}"
                continue
            
            # Calculate mean TF-IDF for this cluster
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
            
            # Get top 3 terms
            top_indices = cluster_tfidf.argsort()[-3:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            # Create label from top terms
            label = "_".join(top_terms)
            self._cluster_labels[cluster_id] = label
    
    def calculate_cluster_growth(
        self,
        cluster_id: int,
        pageviews: pd.DataFrame
    ) -> GrowthMetrics:
        """Calculate growth rate for topic cluster
        
        Aggregates pageviews for all articles in the cluster and calculates
        growth rate over the time period.
        
        Args:
            cluster_id: ID of the cluster to analyze
            pageviews: DataFrame with columns: article, date, views
            
        Returns:
            GrowthMetrics with growth rate and other cluster metrics
            
        Raises:
            ValueError: If cluster_id is invalid or pageviews is empty
        """
        if self.model is None:
            raise ValueError("Must call cluster_articles() before calculate_cluster_growth()")
        
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise ValueError(f"Invalid cluster_id: {cluster_id}")
        
        if pageviews.empty:
            raise ValueError("pageviews DataFrame cannot be empty")
        
        required_cols = {'article', 'date', 'views'}
        if not required_cols.issubset(pageviews.columns):
            raise ValueError(f"pageviews must have columns: {required_cols}")
        
        logger.info(f"Calculating growth for cluster {cluster_id}")
        
        # Get articles in this cluster - need to find from last clustering result
        # Since we don't store cluster assignments, we need them passed or stored
        # For now, we'll need to track assignments
        if not hasattr(self, '_cluster_assignments'):
            raise ValueError("No cluster assignments available. Call cluster_articles() first.")
        
        cluster_articles = [
            article for article, cid in self._cluster_assignments.items()
            if cid == cluster_id
        ]
        
        if not cluster_articles:
            # No articles in cluster, return zero metrics
            return GrowthMetrics(
                cluster_id=cluster_id,
                cluster_name=self._cluster_labels.get(cluster_id, f"Cluster_{cluster_id}"),
                growth_rate=0.0,
                cagr=0.0,
                total_views=0,
                article_count=0,
                is_emerging=False
            )
        
        # Filter pageviews for cluster articles
        cluster_pageviews = pageviews[pageviews['article'].isin(cluster_articles)].copy()
        
        if cluster_pageviews.empty:
            return GrowthMetrics(
                cluster_id=cluster_id,
                cluster_name=self._cluster_labels.get(cluster_id, f"Cluster_{cluster_id}"),
                growth_rate=0.0,
                cagr=0.0,
                total_views=0,
                article_count=len(cluster_articles),
                is_emerging=False
            )
        
        # Aggregate by date
        daily_views = cluster_pageviews.groupby('date')['views'].sum().sort_index()
        
        if len(daily_views) < 2:
            # Need at least 2 data points for growth
            return GrowthMetrics(
                cluster_id=cluster_id,
                cluster_name=self._cluster_labels.get(cluster_id, f"Cluster_{cluster_id}"),
                growth_rate=0.0,
                cagr=0.0,
                total_views=int(daily_views.sum()),
                article_count=len(cluster_articles),
                is_emerging=False
            )
        
        # Calculate growth rate (first to last)
        views_start = daily_views.iloc[0]
        views_end = daily_views.iloc[-1]
        
        if views_start > 0:
            growth_rate = ((views_end - views_start) / views_start) * 100
        else:
            growth_rate = 0.0
        
        # Calculate CAGR if we have enough data
        dates = daily_views.index
        days_span = (dates[-1] - dates[0]).days
        years = days_span / 365.25
        
        if years >= 1 and views_start > 0:
            cagr = (((views_end / views_start) ** (1 / years)) - 1) * 100
        else:
            cagr = 0.0
        
        # Check if emerging (accelerating growth)
        is_emerging = self._is_emerging_topic(daily_views)
        
        return GrowthMetrics(
            cluster_id=cluster_id,
            cluster_name=self._cluster_labels.get(cluster_id, f"Cluster_{cluster_id}"),
            growth_rate=float(growth_rate),
            cagr=float(cagr),
            total_views=int(daily_views.sum()),
            article_count=len(cluster_articles),
            is_emerging=is_emerging
        )
    
    def _is_emerging_topic(self, daily_views: pd.Series) -> bool:
        """Check if topic has accelerating growth (second derivative > 0)
        
        Args:
            daily_views: Series of daily pageview counts
            
        Returns:
            True if growth is accelerating, False otherwise
        """
        if len(daily_views) < 3:
            return False
        
        # Calculate first derivative (velocity)
        first_derivative = daily_views.diff()
        
        # Calculate second derivative (acceleration)
        second_derivative = first_derivative.diff()
        
        # Check if mean acceleration is positive
        mean_acceleration = second_derivative.mean()
        
        return mean_acceleration > 0
    
    def calculate_topic_cagr(
        self,
        cluster_id: int,
        pageviews: pd.DataFrame,
        years: int = 1
    ) -> float:
        """Calculate compound annual growth rate for topic cluster
        
        Args:
            cluster_id: ID of the cluster to analyze
            pageviews: DataFrame with columns: article, date, views
            years: Number of years for CAGR calculation (default: 1)
            
        Returns:
            CAGR as percentage
            
        Raises:
            ValueError: If cluster_id is invalid or insufficient data
        """
        if years <= 0:
            raise ValueError("years must be positive")
        
        metrics = self.calculate_cluster_growth(cluster_id, pageviews)
        
        # If we have the exact years requested, return the CAGR
        # Otherwise, calculate it from the data
        if pageviews.empty:
            return 0.0
        
        # Get date range
        dates = pd.to_datetime(pageviews['date'])
        date_span_years = (dates.max() - dates.min()).days / 365.25
        
        if date_span_years < years:
            logger.warning(
                f"Requested {years} years but only have {date_span_years:.2f} years of data"
            )
            return metrics.cagr
        
        # Filter to exact time period
        end_date = dates.max()
        start_date = end_date - pd.Timedelta(days=int(years * 365.25))
        
        period_pageviews = pageviews[
            (pd.to_datetime(pageviews['date']) >= start_date) &
            (pd.to_datetime(pageviews['date']) <= end_date)
        ].copy()
        
        # Recalculate with filtered data
        metrics = self.calculate_cluster_growth(cluster_id, period_pageviews)
        
        return metrics.cagr
    
    def compare_industries(
        self,
        cluster_ids: List[int],
        pageviews: pd.DataFrame,
        normalize_baseline: bool = True
    ) -> ComparisonResult:
        """Generate comparative metrics across industries
        
        Compares growth metrics across multiple clusters, optionally normalizing
        by baseline traffic for fair comparison.
        
        Args:
            cluster_ids: List of cluster IDs to compare
            pageviews: DataFrame with columns: article, date, views
            normalize_baseline: Whether to normalize by baseline traffic
            
        Returns:
            ComparisonResult with metrics for each cluster
            
        Raises:
            ValueError: If cluster_ids is empty or contains invalid IDs
        """
        if not cluster_ids:
            raise ValueError("cluster_ids cannot be empty")
        
        for cid in cluster_ids:
            if cid < 0 or cid >= self.n_clusters:
                raise ValueError(f"Invalid cluster_id: {cid}")
        
        logger.info(f"Comparing {len(cluster_ids)} clusters")
        
        # Calculate metrics for each cluster
        cluster_metrics = []
        for cluster_id in cluster_ids:
            metrics = self.calculate_cluster_growth(cluster_id, pageviews)
            cluster_metrics.append(metrics)
        
        # Normalize by baseline if requested
        if normalize_baseline and cluster_metrics:
            # Calculate baseline as mean of first week's views
            baselines = {}
            
            for metrics in cluster_metrics:
                cluster_id = metrics.cluster_id
                
                # Get articles in cluster
                cluster_articles = [
                    article for article, cid in self._cluster_assignments.items()
                    if cid == cluster_id
                ]
                
                if not cluster_articles:
                    baselines[cluster_id] = 1.0  # Avoid division by zero
                    continue
                
                # Get first week's data
                cluster_pageviews = pageviews[
                    pageviews['article'].isin(cluster_articles)
                ].copy()
                
                if cluster_pageviews.empty:
                    baselines[cluster_id] = 1.0
                    continue
                
                dates = pd.to_datetime(cluster_pageviews['date']).sort_values()
                if len(dates) == 0:
                    baselines[cluster_id] = 1.0
                    continue
                
                first_week_end = dates.min() + pd.Timedelta(days=7)
                first_week_data = cluster_pageviews[
                    pd.to_datetime(cluster_pageviews['date']) <= first_week_end
                ]
                
                baseline = first_week_data['views'].sum()
                baselines[cluster_id] = max(baseline, 1.0)  # Avoid division by zero
            
            # Normalize growth rates
            normalized_metrics = []
            for metrics in cluster_metrics:
                baseline = baselines[metrics.cluster_id]
                normalized_growth = (metrics.growth_rate * metrics.total_views) / baseline
                normalized_cagr = (metrics.cagr * metrics.total_views) / baseline
                
                normalized_metrics.append(GrowthMetrics(
                    cluster_id=metrics.cluster_id,
                    cluster_name=metrics.cluster_name,
                    growth_rate=float(normalized_growth),
                    cagr=float(normalized_cagr),
                    total_views=metrics.total_views,
                    article_count=metrics.article_count,
                    is_emerging=metrics.is_emerging
                ))
            
            cluster_metrics = normalized_metrics
        
        # Calculate time period
        if not pageviews.empty:
            dates = pd.to_datetime(pageviews['date'])
            time_period_days = (dates.max() - dates.min()).days
        else:
            time_period_days = 0
        
        return ComparisonResult(
            clusters=cluster_metrics,
            baseline_normalized=normalize_baseline,
            time_period_days=time_period_days
        )
