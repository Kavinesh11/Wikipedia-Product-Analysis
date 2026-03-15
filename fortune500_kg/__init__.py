"""
Fortune 500 Knowledge Graph Analytics System

A comprehensive business intelligence platform that leverages Neo4j graph database
technology to analyze Fortune 500 companies through innovation metrics, network
centrality analysis, digital maturity assessment, and predictive analytics.
"""

__version__ = '0.1.0'
__author__ = 'Fortune 500 KG Analytics Team'

from .infrastructure import Neo4jConnection, SchemaManager
from .data_ingestion_pipeline import DataIngestionPipeline
from .data_models import (
    Company,
    Relationship,
    CrawlData,
    IngestionResult,
    GitHubMetrics,
    DataQualityReport
)
from .predictive_model import PredictiveModel, PredictionResult, ValidationMetrics

__all__ = [
    'Neo4jConnection',
    'SchemaManager',
    'DataIngestionPipeline',
    'Company',
    'Relationship',
    'CrawlData',
    'IngestionResult',
    'GitHubMetrics',
    'DataQualityReport',
    'PredictiveModel',
    'PredictionResult',
    'ValidationMetrics',
]
