"""initial_schema_creation

Revision ID: 20260217_140446
Revises: 
Create Date: 2026-02-17 14:04:46

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260217_140446'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema"""
    
    # Create dimension tables
    op.create_table(
        'dim_articles',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('namespace', sa.String(length=50), nullable=True),
        sa.Column('first_seen', sa.DateTime(), nullable=False),
        sa.Column('last_updated', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('title')
    )
    op.create_index('ix_dim_articles_title', 'dim_articles', ['title'])
    
    op.create_table(
        'dim_dates',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('quarter', sa.Integer(), nullable=False),
        sa.Column('month', sa.Integer(), nullable=False),
        sa.Column('week', sa.Integer(), nullable=False),
        sa.Column('day_of_week', sa.Integer(), nullable=False),
        sa.Column('is_weekend', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date')
    )
    op.create_index('ix_dim_dates_date', 'dim_dates', ['date'])
    
    op.create_table(
        'dim_clusters',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('cluster_name', sa.String(length=200), nullable=False),
        sa.Column('industry', sa.String(length=100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create fact tables
    op.create_table(
        'fact_pageviews',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('article_id', sa.Integer(), nullable=False),
        sa.Column('date_id', sa.Integer(), nullable=False),
        sa.Column('hour', sa.Integer(), nullable=True),
        sa.Column('device_type', sa.String(length=20), nullable=False),
        sa.Column('views_human', sa.Integer(), nullable=False),
        sa.Column('views_bot', sa.Integer(), nullable=False),
        sa.Column('views_total', sa.Integer(), nullable=False),
        sa.CheckConstraint('hour >= 0 AND hour < 24'),
        sa.ForeignKeyConstraint(['article_id'], ['dim_articles.id']),
        sa.ForeignKeyConstraint(['date_id'], ['dim_dates.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('article_id', 'date_id', 'hour', 'device_type', 
                           name='uq_pageviews_article_date_hour_device')
    )
    op.create_index('idx_pageviews_article_date', 'fact_pageviews', ['article_id', 'date_id'])
    op.create_index('idx_pageviews_date', 'fact_pageviews', ['date_id'])
    op.create_index('ix_fact_pageviews_article_id', 'fact_pageviews', ['article_id'])
    op.create_index('ix_fact_pageviews_date_id', 'fact_pageviews', ['date_id'])
    
    op.create_table(
        'fact_edits',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('article_id', sa.Integer(), nullable=False),
        sa.Column('revision_id', sa.BigInteger(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('editor_type', sa.String(length=20), nullable=False),
        sa.Column('is_reverted', sa.Boolean(), nullable=True),
        sa.Column('bytes_changed', sa.Integer(), nullable=True),
        sa.Column('edit_summary', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['article_id'], ['dim_articles.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('revision_id')
    )
    op.create_index('idx_edits_article_timestamp', 'fact_edits', ['article_id', 'timestamp'])
    op.create_index('idx_edits_timestamp', 'fact_edits', ['timestamp'])
    op.create_index('ix_fact_edits_article_id', 'fact_edits', ['article_id'])
    op.create_index('ix_fact_edits_timestamp', 'fact_edits', ['timestamp'])
    
    op.create_table(
        'fact_crawl_results',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('article_id', sa.Integer(), nullable=False),
        sa.Column('crawl_timestamp', sa.DateTime(), nullable=False),
        sa.Column('content_length', sa.Integer(), nullable=True),
        sa.Column('infobox_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('categories', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('internal_links', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('tables_count', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['article_id'], ['dim_articles.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_crawl_article', 'fact_crawl_results', ['article_id'])
    op.create_index('ix_fact_crawl_results_article_id', 'fact_crawl_results', ['article_id'])
    
    # Create mapping table
    op.create_table(
        'map_article_clusters',
        sa.Column('article_id', sa.Integer(), nullable=False),
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.CheckConstraint('confidence_score >= 0 AND confidence_score <= 1'),
        sa.ForeignKeyConstraint(['article_id'], ['dim_articles.id']),
        sa.ForeignKeyConstraint(['cluster_id'], ['dim_clusters.id']),
        sa.PrimaryKeyConstraint('article_id', 'cluster_id')
    )
    
    # Create aggregated metrics tables
    op.create_table(
        'agg_article_metrics_daily',
        sa.Column('article_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('total_views', sa.Integer(), nullable=False),
        sa.Column('view_growth_rate', sa.Float(), nullable=True),
        sa.Column('edit_count', sa.Integer(), nullable=True),
        sa.Column('edit_velocity', sa.Float(), nullable=True),
        sa.Column('hype_score', sa.Float(), nullable=True),
        sa.Column('reputation_risk', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['article_id'], ['dim_articles.id']),
        sa.PrimaryKeyConstraint('article_id', 'date')
    )
    
    op.create_table(
        'agg_cluster_metrics',
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('total_views', sa.Integer(), nullable=False),
        sa.Column('article_count', sa.Integer(), nullable=False),
        sa.Column('avg_growth_rate', sa.Float(), nullable=True),
        sa.Column('topic_cagr', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['cluster_id'], ['dim_clusters.id']),
        sa.PrimaryKeyConstraint('cluster_id', 'date')
    )


def downgrade() -> None:
    """Drop all tables"""
    op.drop_table('agg_cluster_metrics')
    op.drop_table('agg_article_metrics_daily')
    op.drop_table('map_article_clusters')
    op.drop_table('fact_crawl_results')
    op.drop_table('fact_edits')
    op.drop_table('fact_pageviews')
    op.drop_table('dim_clusters')
    op.drop_table('dim_dates')
    op.drop_table('dim_articles')

