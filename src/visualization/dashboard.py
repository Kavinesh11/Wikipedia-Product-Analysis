"""Streamlit Dashboard Application

Interactive dashboard for Wikipedia Intelligence System.
Provides real-time visualizations of demand trends, reputation alerts,
emerging topics, and competitive intelligence.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import hashlib
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from src.storage.database import get_database
from src.storage.cache import get_cache
from src.storage.models import (
    DimArticle, DimCluster, AggArticleMetricsDaily, AggClusterMetrics,
    MapArticleCluster
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Wikipedia Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cache_key(data_type: str, **params) -> str:
    """Generate cache key from parameters
    
    Args:
        data_type: Type of data (demand_trends, competitor_comparison, etc.)
        **params: Query parameters
        
    Returns:
        Hash key for caching
    """
    param_str = f"{data_type}:" + ":".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()


def load_articles() -> List[str]:
    """Load list of available articles from database
    
    Returns:
        List of article titles
    """
    try:
        db = get_database()
        with db.get_session() as session:
            articles = session.query(DimArticle.title).order_by(DimArticle.title).all()
            return [a[0] for a in articles]
    except Exception as e:
        logger.error(f"Failed to load articles: {e}")
        return []


def load_industries() -> List[str]:
    """Load list of available industries from database
    
    Returns:
        List of industry names
    """
    try:
        db = get_database()
        with db.get_session() as session:
            industries = session.query(DimCluster.industry).distinct().filter(
                DimCluster.industry.isnot(None)
            ).order_by(DimCluster.industry).all()
            return [i[0] for i in industries]
    except Exception as e:
        logger.error(f"Failed to load industries: {e}")
        return []


def export_to_csv(data: pd.DataFrame) -> bytes:
    """Export dataframe to CSV bytes
    
    Args:
        data: DataFrame to export
        
    Returns:
        CSV data as bytes
    """
    return data.to_csv(index=False).encode('utf-8')


def export_to_pdf(data: pd.DataFrame, title: str) -> bytes:
    """Export dataframe to PDF bytes
    
    Args:
        data: DataFrame to export
        title: Report title
        
    Returns:
        PDF data as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    # Add title
    styles = getSampleStyleSheet()
    title_para = Paragraph(title, styles['Title'])
    elements.append(title_para)
    elements.append(Spacer(1, 12))
    
    # Add timestamp
    timestamp = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    elements.append(timestamp)
    elements.append(Spacer(1, 12))
    
    # Convert dataframe to table
    table_data = [data.columns.tolist()] + data.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================================
# SIDEBAR - FILTERS AND CONFIGURATION
# ============================================================================

def render_sidebar():
    """Render sidebar with filters and configuration options
    
    Returns:
        Dictionary of filter values
    """
    st.sidebar.markdown("## 🎛️ Filters")
    
    # Date range filter
    st.sidebar.markdown("### Date Range")
    date_preset = st.sidebar.selectbox(
        "Preset",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"]
    )
    
    if date_preset == "Custom":
        start_date = st.sidebar.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30)
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=datetime.now()
        )
    else:
        days_map = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 90 Days": 90
        }
        days = days_map[date_preset]
        end_date = datetime.now().date()
        start_date = (datetime.now() - timedelta(days=days)).date()
    
    # Industry filter
    st.sidebar.markdown("### Industry")
    industries = load_industries()
    selected_industry = st.sidebar.selectbox(
        "Select Industry",
        ["All"] + industries
    )
    
    # Metric type filter
    st.sidebar.markdown("### Metric Type")
    metric_type = st.sidebar.selectbox(
        "Select Metric",
        ["Pageviews", "Growth Rate", "Hype Score", "Reputation Risk"]
    )
    
    # Auto-refresh configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Configuration")
    
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    refresh_interval = 300  # Default 5 minutes
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=60,
            max_value=600,
            value=300,
            step=60
        )
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "industry": selected_industry if selected_industry != "All" else None,
        "metric_type": metric_type,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval
    }


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">📊 Wikipedia Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Real-time business intelligence from Wikipedia data")
    
    # Render sidebar and get filters
    filters = render_sidebar()
    
    # Auto-refresh logic
    if filters["auto_refresh"]:
        st.sidebar.info(f"Auto-refresh enabled: {filters['refresh_interval']}s")
        # Note: Streamlit's st.experimental_rerun() can be used for auto-refresh
        # but requires careful implementation to avoid infinite loops
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Demand Trends",
        "🏆 Competitor Comparison",
        "🚨 Reputation Alerts",
        "🔥 Emerging Topics",
        "👑 Traffic Leaderboard",
        "📥 Export Data"
    ])
    
    with tab1:
        render_demand_trends(filters)
    
    with tab2:
        render_competitor_comparison(filters)
    
    with tab3:
        render_reputation_alerts(filters)
    
    with tab4:
        render_emerging_topics(filters)
    
    with tab5:
        render_traffic_leaderboard(filters)
    
    with tab6:
        render_export_panel(filters)


# ============================================================================
# PLACEHOLDER RENDER FUNCTIONS (to be implemented in subsequent tasks)
# ============================================================================

def render_demand_trends(filters: Dict[str, Any]):
    """Render demand trends visualization
    
    Creates time series charts using Plotly with support for multiple
    article comparison and interactive tooltips.
    
    Args:
        filters: Dictionary of filter values from sidebar
    """
    st.markdown("### 📈 Product Demand Trends")
    
    # Article selection
    articles = load_articles()
    if not articles:
        st.warning("No articles found in database. Please run data collection first.")
        return
    
    selected_articles = st.multiselect(
        "Select articles to compare",
        articles,
        default=articles[:3] if len(articles) >= 3 else articles,
        max_selections=10
    )
    
    if not selected_articles:
        st.info("Please select at least one article to view demand trends.")
        return
    
    # Load data with caching
    cache_key = get_cache_key(
        "demand_trends",
        articles=",".join(sorted(selected_articles)),
        start_date=str(filters["start_date"]),
        end_date=str(filters["end_date"]),
        metric=filters["metric_type"]
    )
    
    cache = get_cache()
    data = cache.get_dashboard_data("demand_trends", cache_key)
    
    if data is None:
        # Load from database
        data = load_demand_trends_data(
            selected_articles,
            filters["start_date"],
            filters["end_date"],
            filters["metric_type"]
        )
        
        if data is not None and not data.empty:
            cache.set_dashboard_data("demand_trends", cache_key, data.to_dict())
    else:
        # Convert cached dict back to DataFrame
        data = pd.DataFrame(data)
    
    if data is None or data.empty:
        st.warning("No data available for selected articles and date range.")
        return
    
    # Create time series chart
    metric_column_map = {
        "Pageviews": "total_views",
        "Growth Rate": "view_growth_rate",
        "Hype Score": "hype_score",
        "Reputation Risk": "reputation_risk"
    }
    
    metric_column = metric_column_map.get(filters["metric_type"], "total_views")
    
    fig = px.line(
        data,
        x="date",
        y=metric_column,
        color="article",
        title=f"{filters['metric_type']} Over Time",
        labels={
            "date": "Date",
            metric_column: filters["metric_type"],
            "article": "Article"
        },
        markers=True
    )
    
    # Customize layout
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=filters["metric_type"],
        legend_title="Articles",
        height=500,
        template="plotly_white"
    )
    
    # Add interactive tooltips
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>" +
                      "Date: %{x|%Y-%m-%d}<br>" +
                      f"{filters['metric_type']}: %{{y:,.0f}}<br>" +
                      "<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### Summary Statistics")
    cols = st.columns(len(selected_articles))
    
    for idx, article in enumerate(selected_articles):
        article_data = data[data["article"] == article]
        if not article_data.empty:
            with cols[idx]:
                st.markdown(f"**{article}**")
                current_value = article_data[metric_column].iloc[-1]
                avg_value = article_data[metric_column].mean()
                
                st.metric(
                    label="Current",
                    value=f"{current_value:,.0f}" if pd.notna(current_value) else "N/A"
                )
                st.metric(
                    label="Average",
                    value=f"{avg_value:,.0f}" if pd.notna(avg_value) else "N/A"
                )


def load_demand_trends_data(
    articles: List[str],
    start_date: datetime,
    end_date: datetime,
    metric_type: str
) -> Optional[pd.DataFrame]:
    """Load demand trends data from database
    
    Args:
        articles: List of article titles
        start_date: Start date for data
        end_date: End date for data
        metric_type: Type of metric to load
        
    Returns:
        DataFrame with demand trends data or None
    """
    try:
        db = get_database()
        with db.get_session() as session:
            # Query aggregated daily metrics
            query = session.query(
                DimArticle.title.label("article"),
                AggArticleMetricsDaily.date,
                AggArticleMetricsDaily.total_views,
                AggArticleMetricsDaily.view_growth_rate,
                AggArticleMetricsDaily.hype_score,
                AggArticleMetricsDaily.reputation_risk
            ).join(
                DimArticle,
                AggArticleMetricsDaily.article_id == DimArticle.id
            ).filter(
                DimArticle.title.in_(articles),
                AggArticleMetricsDaily.date >= start_date,
                AggArticleMetricsDaily.date <= end_date
            ).order_by(
                DimArticle.title,
                AggArticleMetricsDaily.date
            )
            
            results = query.all()
            
            if not results:
                return None
            
            # Convert to DataFrame
            data = pd.DataFrame([
                {
                    "article": r.article,
                    "date": r.date,
                    "total_views": r.total_views or 0,
                    "view_growth_rate": r.view_growth_rate or 0,
                    "hype_score": r.hype_score or 0,
                    "reputation_risk": r.reputation_risk or 0
                }
                for r in results
            ])
            
            return data
            
    except Exception as e:
        logger.error(f"Failed to load demand trends data: {e}", exc_info=True)
        return None


def render_competitor_comparison(filters: Dict[str, Any]):
    """Render competitor comparison table
    
    Creates sortable table with key metrics and column sorting functionality.
    
    Args:
        filters: Dictionary of filter values from sidebar
    """
    st.markdown("### 🏆 Competitor Comparison")
    
    # Article selection
    articles = load_articles()
    if not articles:
        st.warning("No articles found in database. Please run data collection first.")
        return
    
    selected_articles = st.multiselect(
        "Select competitors to compare",
        articles,
        default=articles[:5] if len(articles) >= 5 else articles,
        key="competitor_select"
    )
    
    if not selected_articles:
        st.info("Please select at least one article to compare.")
        return
    
    # Load comparison data
    cache_key = get_cache_key(
        "competitor_comparison",
        articles=",".join(sorted(selected_articles)),
        start_date=str(filters["start_date"]),
        end_date=str(filters["end_date"])
    )
    
    cache = get_cache()
    data = cache.get_dashboard_data("competitor_comparison", cache_key)
    
    if data is None:
        # Load from database
        data = load_competitor_comparison_data(
            selected_articles,
            filters["start_date"],
            filters["end_date"]
        )
        
        if data is not None and not data.empty:
            cache.set_dashboard_data("competitor_comparison", cache_key, data.to_dict())
    else:
        # Convert cached dict back to DataFrame
        data = pd.DataFrame(data)
    
    if data is None or data.empty:
        st.warning("No data available for selected articles and date range.")
        return
    
    # Display sortable table
    st.markdown("#### Key Metrics Comparison")
    st.markdown("Click on column headers to sort")
    
    # Format the dataframe for display
    display_data = data.copy()
    display_data["Total Views"] = display_data["total_views"].apply(lambda x: f"{x:,.0f}")
    display_data["Avg Daily Views"] = display_data["avg_daily_views"].apply(lambda x: f"{x:,.0f}")
    display_data["Growth Rate (%)"] = display_data["growth_rate"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    display_data["Hype Score"] = display_data["hype_score"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_data["Reputation Risk"] = display_data["reputation_risk"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_data["Edit Count"] = display_data["edit_count"].apply(lambda x: f"{x:,.0f}")
    
    # Select columns for display
    display_columns = [
        "Article",
        "Total Views",
        "Avg Daily Views",
        "Growth Rate (%)",
        "Hype Score",
        "Reputation Risk",
        "Edit Count"
    ]
    
    # Use Streamlit's dataframe with sorting
    st.dataframe(
        display_data[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Visualization of comparison
    st.markdown("#### Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for total views
        fig_views = px.bar(
            data,
            x="article",
            y="total_views",
            title="Total Views Comparison",
            labels={"article": "Article", "total_views": "Total Views"},
            color="total_views",
            color_continuous_scale="Blues"
        )
        fig_views.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_views, use_container_width=True)
    
    with col2:
        # Bar chart for growth rate
        fig_growth = px.bar(
            data,
            x="article",
            y="growth_rate",
            title="Growth Rate Comparison",
            labels={"article": "Article", "growth_rate": "Growth Rate (%)"},
            color="growth_rate",
            color_continuous_scale="RdYlGn"
        )
        fig_growth.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_growth, use_container_width=True)


def load_competitor_comparison_data(
    articles: List[str],
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """Load competitor comparison data from database
    
    Args:
        articles: List of article titles
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with comparison metrics or None
    """
    try:
        db = get_database()
        with db.get_session() as session:
            # Query aggregated metrics for each article
            from sqlalchemy import func
            
            query = session.query(
                DimArticle.title.label("article"),
                func.sum(AggArticleMetricsDaily.total_views).label("total_views"),
                func.avg(AggArticleMetricsDaily.total_views).label("avg_daily_views"),
                func.avg(AggArticleMetricsDaily.view_growth_rate).label("growth_rate"),
                func.avg(AggArticleMetricsDaily.hype_score).label("hype_score"),
                func.avg(AggArticleMetricsDaily.reputation_risk).label("reputation_risk"),
                func.sum(AggArticleMetricsDaily.edit_count).label("edit_count")
            ).join(
                DimArticle,
                AggArticleMetricsDaily.article_id == DimArticle.id
            ).filter(
                DimArticle.title.in_(articles),
                AggArticleMetricsDaily.date >= start_date,
                AggArticleMetricsDaily.date <= end_date
            ).group_by(
                DimArticle.title
            ).order_by(
                func.sum(AggArticleMetricsDaily.total_views).desc()
            )
            
            results = query.all()
            
            if not results:
                return None
            
            # Convert to DataFrame
            data = pd.DataFrame([
                {
                    "Article": r.article,
                    "article": r.article,  # Keep for internal use
                    "total_views": r.total_views or 0,
                    "avg_daily_views": r.avg_daily_views or 0,
                    "growth_rate": r.growth_rate or 0,
                    "hype_score": r.hype_score or 0,
                    "reputation_risk": r.reputation_risk or 0,
                    "edit_count": r.edit_count or 0
                }
                for r in results
            ])
            
            return data
            
    except Exception as e:
        logger.error(f"Failed to load competitor comparison data: {e}", exc_info=True)
        return None


def render_reputation_alerts(filters: Dict[str, Any]):
    """Render reputation alerts panel
    
    Displays active alerts prominently with color coding and shows
    alert details (article, risk score, timestamp).
    
    Args:
        filters: Dictionary of filter values from sidebar
    """
    st.markdown("### 🚨 Reputation Risk Alerts")
    
    # Load active alerts
    alerts = load_reputation_alerts(filters["start_date"], filters["end_date"])
    
    if not alerts or len(alerts) == 0:
        st.success("✅ No active reputation risk alerts")
        return
    
    # Display alert summary
    st.markdown(f"**{len(alerts)} Active Alert(s)**")
    
    # Group alerts by priority
    high_alerts = [a for a in alerts if a["alert_level"] == "high"]
    medium_alerts = [a for a in alerts if a["alert_level"] == "medium"]
    low_alerts = [a for a in alerts if a["alert_level"] == "low"]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Priority", len(high_alerts), delta=None)
    with col2:
        st.metric("Medium Priority", len(medium_alerts), delta=None)
    with col3:
        st.metric("Low Priority", len(low_alerts), delta=None)
    
    st.markdown("---")
    
    # Display alerts by priority
    if high_alerts:
        st.markdown("#### 🔴 High Priority Alerts")
        for alert in high_alerts:
            render_alert_card(alert, "high")
    
    if medium_alerts:
        st.markdown("#### 🟡 Medium Priority Alerts")
        for alert in medium_alerts:
            render_alert_card(alert, "medium")
    
    if low_alerts:
        st.markdown("#### 🟢 Low Priority Alerts")
        for alert in low_alerts:
            render_alert_card(alert, "low")


def render_alert_card(alert: Dict[str, Any], level: str):
    """Render individual alert card with color coding
    
    Args:
        alert: Alert data dictionary
        level: Alert level (high, medium, low)
    """
    css_class = f"alert-{level}"
    
    alert_html = f"""
    <div class="{css_class}">
        <h4>📄 {alert['article']}</h4>
        <p><strong>Risk Score:</strong> {alert['risk_score']:.2f} / 1.00</p>
        <p><strong>Edit Velocity:</strong> {alert['edit_velocity']:.2f} edits/hour</p>
        <p><strong>Vandalism Rate:</strong> {alert['vandalism_rate']:.2f}%</p>
        <p><strong>Anonymous Edits:</strong> {alert['anonymous_edit_pct']:.2f}%</p>
        <p><strong>Detected:</strong> {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """
    
    st.markdown(alert_html, unsafe_allow_html=True)


def load_reputation_alerts(
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, Any]]:
    """Load active reputation alerts from database
    
    Args:
        start_date: Start date for alerts
        end_date: End date for alerts
        
    Returns:
        List of alert dictionaries
    """
    try:
        db = get_database()
        with db.get_session() as session:
            # Query articles with high reputation risk
            query = session.query(
                DimArticle.title,
                AggArticleMetricsDaily.date,
                AggArticleMetricsDaily.reputation_risk,
                AggArticleMetricsDaily.edit_velocity,
                AggArticleMetricsDaily.edit_count
            ).join(
                DimArticle,
                AggArticleMetricsDaily.article_id == DimArticle.id
            ).filter(
                AggArticleMetricsDaily.date >= start_date,
                AggArticleMetricsDaily.date <= end_date,
                AggArticleMetricsDaily.reputation_risk.isnot(None),
                AggArticleMetricsDaily.reputation_risk > 0.3  # Alert threshold
            ).order_by(
                AggArticleMetricsDaily.reputation_risk.desc(),
                AggArticleMetricsDaily.date.desc()
            )
            
            results = query.all()
            
            if not results:
                return []
            
            # Convert to alert dictionaries
            alerts = []
            for r in results:
                # Determine alert level based on risk score
                if r.reputation_risk >= 0.7:
                    alert_level = "high"
                elif r.reputation_risk >= 0.5:
                    alert_level = "medium"
                else:
                    alert_level = "low"
                
                # Calculate derived metrics (simplified for demo)
                vandalism_rate = min(r.reputation_risk * 50, 100)  # Simplified calculation
                anonymous_edit_pct = min(r.reputation_risk * 80, 100)  # Simplified calculation
                
                alerts.append({
                    "article": r.title,
                    "risk_score": r.reputation_risk,
                    "edit_velocity": r.edit_velocity or 0,
                    "vandalism_rate": vandalism_rate,
                    "anonymous_edit_pct": anonymous_edit_pct,
                    "alert_level": alert_level,
                    "timestamp": datetime.combine(r.date, datetime.min.time())
                })
            
            return alerts
            
    except Exception as e:
        logger.error(f"Failed to load reputation alerts: {e}", exc_info=True)
        return []


def render_emerging_topics(filters: Dict[str, Any]):
    """Render emerging topics heatmap
    
    Creates heatmap visualization with color-coded growth and
    cluster labels with tooltips.
    
    Args:
        filters: Dictionary of filter values from sidebar
    """
    st.markdown("### 🔥 Emerging Topics")
    
    # Load cluster data
    clusters_data = load_emerging_topics_data(
        filters["start_date"],
        filters["end_date"],
        filters["industry"]
    )
    
    if clusters_data is None or clusters_data.empty:
        st.warning("No cluster data available for the selected filters.")
        return
    
    # Display summary metrics
    st.markdown("#### Topic Growth Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_clusters = len(clusters_data)
        st.metric("Total Topics", total_clusters)
    with col2:
        emerging_count = len(clusters_data[clusters_data["is_emerging"] == True])
        st.metric("Emerging Topics", emerging_count)
    with col3:
        avg_growth = clusters_data["avg_growth_rate"].mean()
        st.metric("Avg Growth Rate", f"{avg_growth:.2f}%")
    
    st.markdown("---")
    
    # Create heatmap
    st.markdown("#### Growth Rate Heatmap")
    
    # Prepare data for heatmap
    heatmap_data = clusters_data.pivot_table(
        index="cluster_name",
        columns="industry",
        values="avg_growth_rate",
        aggfunc="mean"
    )
    
    if heatmap_data.empty:
        # If pivot fails, create a simple bar chart instead
        fig = px.bar(
            clusters_data,
            x="cluster_name",
            y="avg_growth_rate",
            color="avg_growth_rate",
            title="Topic Growth Rates",
            labels={
                "cluster_name": "Topic",
                "avg_growth_rate": "Growth Rate (%)"
            },
            color_continuous_scale="RdYlGn",
            hover_data=["industry", "article_count", "total_views"]
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Industry", y="Topic", color="Growth Rate (%)"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Topic Growth by Industry"
        )
        
        fig.update_layout(
            height=max(400, len(heatmap_data.index) * 30),
            xaxis_title="Industry",
            yaxis_title="Topic Cluster"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display emerging topics table
    st.markdown("#### 🚀 Emerging Topics (Accelerating Growth)")
    
    emerging_topics = clusters_data[clusters_data["is_emerging"] == True].sort_values(
        "avg_growth_rate", ascending=False
    )
    
    if emerging_topics.empty:
        st.info("No emerging topics detected in the selected period.")
    else:
        display_emerging = emerging_topics[[
            "cluster_name", "industry", "avg_growth_rate", 
            "article_count", "total_views"
        ]].copy()
        
        display_emerging.columns = [
            "Topic", "Industry", "Growth Rate (%)", 
            "Articles", "Total Views"
        ]
        
        display_emerging["Growth Rate (%)"] = display_emerging["Growth Rate (%)"].apply(
            lambda x: f"{x:.2f}%"
        )
        display_emerging["Total Views"] = display_emerging["Total Views"].apply(
            lambda x: f"{x:,.0f}"
        )
        
        st.dataframe(display_emerging, use_container_width=True, hide_index=True)


def load_emerging_topics_data(
    start_date: datetime,
    end_date: datetime,
    industry: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Load emerging topics data from database
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        industry: Optional industry filter
        
    Returns:
        DataFrame with cluster metrics or None
    """
    try:
        db = get_database()
        with db.get_session() as session:
            from sqlalchemy import func
            
            # Query cluster metrics
            query = session.query(
                DimCluster.cluster_name,
                DimCluster.industry,
                func.avg(AggClusterMetrics.avg_growth_rate).label("avg_growth_rate"),
                func.sum(AggClusterMetrics.article_count).label("article_count"),
                func.sum(AggClusterMetrics.total_views).label("total_views"),
                func.avg(AggClusterMetrics.topic_cagr).label("topic_cagr")
            ).join(
                DimCluster,
                AggClusterMetrics.cluster_id == DimCluster.id
            ).filter(
                AggClusterMetrics.date >= start_date,
                AggClusterMetrics.date <= end_date
            )
            
            if industry:
                query = query.filter(DimCluster.industry == industry)
            
            query = query.group_by(
                DimCluster.cluster_name,
                DimCluster.industry
            ).order_by(
                func.avg(AggClusterMetrics.avg_growth_rate).desc()
            )
            
            results = query.all()
            
            if not results:
                return None
            
            # Convert to DataFrame
            data = pd.DataFrame([
                {
                    "cluster_name": r.cluster_name,
                    "industry": r.industry or "Unknown",
                    "avg_growth_rate": r.avg_growth_rate or 0,
                    "article_count": r.article_count or 0,
                    "total_views": r.total_views or 0,
                    "topic_cagr": r.topic_cagr or 0,
                    "is_emerging": (r.avg_growth_rate or 0) > 10  # Simplified: >10% growth
                }
                for r in results
            ])
            
            return data
            
    except Exception as e:
        logger.error(f"Failed to load emerging topics data: {e}", exc_info=True)
        return None


def render_traffic_leaderboard(filters: Dict[str, Any]):
    """Render traffic leaderboard
    
    Creates ranked list of top articles by pageviews with pagination
    for large lists.
    
    Args:
        filters: Dictionary of filter values from sidebar
    """
    st.markdown("### 👑 Traffic Leaderboard")
    
    # Pagination settings
    items_per_page = st.selectbox(
        "Items per page",
        [10, 25, 50, 100],
        index=1
    )
    
    # Load leaderboard data
    leaderboard_data = load_traffic_leaderboard_data(
        filters["start_date"],
        filters["end_date"],
        filters["industry"]
    )
    
    if leaderboard_data is None or leaderboard_data.empty:
        st.warning("No traffic data available for the selected filters.")
        return
    
    # Display summary
    st.markdown("#### Top Performing Articles")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_articles = len(leaderboard_data)
        st.metric("Total Articles", total_articles)
    with col2:
        total_views = leaderboard_data["total_views"].sum()
        st.metric("Total Views", f"{total_views:,.0f}")
    with col3:
        avg_views = leaderboard_data["total_views"].mean()
        st.metric("Avg Views", f"{avg_views:,.0f}")
    
    st.markdown("---")
    
    # Calculate pagination
    total_pages = (len(leaderboard_data) - 1) // items_per_page + 1
    
    # Page selector
    if total_pages > 1:
        page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )
    else:
        page = 1
    
    # Get page data
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_data = leaderboard_data.iloc[start_idx:end_idx].copy()
    
    # Add rank column
    page_data.insert(0, "Rank", range(start_idx + 1, start_idx + len(page_data) + 1))
    
    # Format for display
    display_data = page_data.copy()
    display_data["Total Views"] = display_data["total_views"].apply(lambda x: f"{x:,.0f}")
    display_data["Avg Daily Views"] = display_data["avg_daily_views"].apply(lambda x: f"{x:,.0f}")
    display_data["Growth Rate"] = display_data["growth_rate"].apply(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
    )
    
    # Select columns for display
    display_columns = [
        "Rank",
        "Article",
        "Total Views",
        "Avg Daily Views",
        "Growth Rate"
    ]
    
    # Display leaderboard table
    st.dataframe(
        display_data[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Visualization - Top 10 bar chart
    st.markdown("#### 📊 Top 10 Articles by Views")
    
    top_10 = leaderboard_data.head(10)
    
    fig = px.bar(
        top_10,
        x="total_views",
        y="article",
        orientation="h",
        title="Top 10 Articles",
        labels={"total_views": "Total Views", "article": "Article"},
        color="total_views",
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show pagination info
    if total_pages > 1:
        st.info(f"Showing {start_idx + 1}-{min(end_idx, len(leaderboard_data))} of {len(leaderboard_data)} articles")


def load_traffic_leaderboard_data(
    start_date: datetime,
    end_date: datetime,
    industry: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Load traffic leaderboard data from database
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        industry: Optional industry filter
        
    Returns:
        DataFrame with ranked articles or None
    """
    try:
        db = get_database()
        with db.get_session() as session:
            from sqlalchemy import func
            
            # Base query
            query = session.query(
                DimArticle.title.label("article"),
                func.sum(AggArticleMetricsDaily.total_views).label("total_views"),
                func.avg(AggArticleMetricsDaily.total_views).label("avg_daily_views"),
                func.avg(AggArticleMetricsDaily.view_growth_rate).label("growth_rate")
            ).join(
                DimArticle,
                AggArticleMetricsDaily.article_id == DimArticle.id
            ).filter(
                AggArticleMetricsDaily.date >= start_date,
                AggArticleMetricsDaily.date <= end_date
            )
            
            # Apply industry filter if specified
            if industry:
                query = query.join(
                    MapArticleCluster,
                    DimArticle.id == MapArticleCluster.article_id
                ).join(
                    DimCluster,
                    MapArticleCluster.cluster_id == DimCluster.id
                ).filter(
                    DimCluster.industry == industry
                )
            
            query = query.group_by(
                DimArticle.title
            ).order_by(
                func.sum(AggArticleMetricsDaily.total_views).desc()
            )
            
            results = query.all()
            
            if not results:
                return None
            
            # Convert to DataFrame
            data = pd.DataFrame([
                {
                    "Article": r.article,
                    "article": r.article,  # Keep for internal use
                    "total_views": r.total_views or 0,
                    "avg_daily_views": r.avg_daily_views or 0,
                    "growth_rate": r.growth_rate or 0
                }
                for r in results
            ])
            
            return data
            
    except Exception as e:
        logger.error(f"Failed to load traffic leaderboard data: {e}", exc_info=True)
        return None


def render_export_panel(filters: Dict[str, Any]):
    """Render data export panel
    
    Provides CSV and PDF export functionality for all dashboard data.
    
    Args:
        filters: Dictionary of filter values from sidebar
    """
    st.markdown("### 📥 Export Data")
    
    st.markdown("""
    Export dashboard data for offline analysis or reporting.
    Select the data type and format below.
    """)
    
    # Export type selection
    export_type = st.selectbox(
        "Select data to export",
        [
            "Demand Trends",
            "Competitor Comparison",
            "Reputation Alerts",
            "Emerging Topics",
            "Traffic Leaderboard",
            "All Data"
        ]
    )
    
    # Export format selection
    export_format = st.radio(
        "Select export format",
        ["CSV", "PDF"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Export button
    if st.button("Generate Export", type="primary"):
        with st.spinner("Generating export..."):
            try:
                # Load data based on selection
                if export_type == "Demand Trends":
                    articles = load_articles()
                    data = load_demand_trends_data(
                        articles[:10],  # Limit to first 10 for export
                        filters["start_date"],
                        filters["end_date"],
                        filters["metric_type"]
                    )
                    title = "Demand Trends Report"
                
                elif export_type == "Competitor Comparison":
                    articles = load_articles()
                    data = load_competitor_comparison_data(
                        articles[:20],  # Limit to first 20
                        filters["start_date"],
                        filters["end_date"]
                    )
                    title = "Competitor Comparison Report"
                
                elif export_type == "Reputation Alerts":
                    alerts = load_reputation_alerts(
                        filters["start_date"],
                        filters["end_date"]
                    )
                    data = pd.DataFrame(alerts) if alerts else None
                    title = "Reputation Alerts Report"
                
                elif export_type == "Emerging Topics":
                    data = load_emerging_topics_data(
                        filters["start_date"],
                        filters["end_date"],
                        filters["industry"]
                    )
                    title = "Emerging Topics Report"
                
                elif export_type == "Traffic Leaderboard":
                    data = load_traffic_leaderboard_data(
                        filters["start_date"],
                        filters["end_date"],
                        filters["industry"]
                    )
                    title = "Traffic Leaderboard Report"
                
                else:  # All Data
                    st.warning("Exporting all data may take some time...")
                    # Combine all data types
                    all_data = {}
                    
                    articles = load_articles()
                    all_data["demand_trends"] = load_demand_trends_data(
                        articles[:10], filters["start_date"], 
                        filters["end_date"], filters["metric_type"]
                    )
                    all_data["competitor_comparison"] = load_competitor_comparison_data(
                        articles[:20], filters["start_date"], filters["end_date"]
                    )
                    all_data["emerging_topics"] = load_emerging_topics_data(
                        filters["start_date"], filters["end_date"], filters["industry"]
                    )
                    all_data["traffic_leaderboard"] = load_traffic_leaderboard_data(
                        filters["start_date"], filters["end_date"], filters["industry"]
                    )
                    
                    # For "All Data", we'll export each as a separate file
                    data = None
                    title = "Complete Dashboard Report"
                
                # Generate export
                if data is None and export_type != "All Data":
                    st.error("No data available for export.")
                    return
                
                if export_format == "CSV":
                    if export_type == "All Data":
                        # Create a zip file with multiple CSVs
                        import zipfile
                        zip_buffer = io.BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for name, df in all_data.items():
                                if df is not None and not df.empty:
                                    csv_data = export_to_csv(df)
                                    zip_file.writestr(f"{name}.csv", csv_data)
                        
                        zip_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download All Data (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                    else:
                        csv_data = export_to_csv(data)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"{export_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                else:  # PDF
                    if export_type == "All Data":
                        st.warning("PDF export for 'All Data' is not supported. Please select a specific data type or use CSV format.")
                        return
                    
                    pdf_data = export_to_pdf(data, title)
                    
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_data,
                        file_name=f"{export_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                
                st.success("✅ Export generated successfully!")
                
            except Exception as e:
                logger.error(f"Export failed: {e}", exc_info=True)
                st.error(f"Export failed: {str(e)}")
    
    # Export info
    st.markdown("---")
    st.markdown("#### Export Information")
    st.info(f"""
    **Date Range:** {filters['start_date']} to {filters['end_date']}  
    **Industry Filter:** {filters['industry'] or 'All'}  
    **Metric Type:** {filters['metric_type']}
    
    CSV exports include all data fields. PDF exports include formatted tables.
    """)



# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
