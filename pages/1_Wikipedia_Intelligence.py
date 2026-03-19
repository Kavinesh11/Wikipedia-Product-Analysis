"""Wikipedia Intelligence Platform — Streamlit Dashboard (demo data)."""

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Wikipedia Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTICLES = [
    "Artificial Intelligence", "Climate Change", "Quantum Computing",
    "CRISPR", "Blockchain", "Electric Vehicles", "SpaceX", "ChatGPT",
    "Metaverse", "Renewable Energy", "Cybersecurity", "Machine Learning",
    "Vaccine", "Nuclear Fusion", "Autonomous Vehicles",
]
DEVICE_TYPES = ["desktop", "mobile-web", "mobile-app"]


@st.cache_data
def gen_pageviews(n_days: int = 365) -> pd.DataFrame:
    random.seed(7)
    rows = []
    base = datetime(2023, 1, 1)
    for article in ARTICLES:
        base_views = random.randint(5_000, 80_000)
        for d in range(n_days):
            dt = base + timedelta(days=d)
            trend = 1 + d * random.uniform(0.0001, 0.0008)
            for dev in DEVICE_TYPES:
                mult = {"desktop": 0.45, "mobile-web": 0.35, "mobile-app": 0.20}[dev]
                total = int(base_views * mult * trend * random.uniform(0.7, 1.3))
                bot = int(total * random.uniform(0.02, 0.12))
                rows.append({
                    "article": article, "date": dt, "device_type": dev,
                    "views_human": total - bot, "views_bot": bot, "views_total": total,
                })
    return pd.DataFrame(rows)


@st.cache_data
def gen_hype_scores() -> pd.DataFrame:
    random.seed(42)
    rows = []
    for article in ARTICLES:
        vv = random.uniform(0.1, 0.9)
        eg = random.uniform(0.1, 0.9)
        ce = random.uniform(0.1, 0.9)
        hype = 0.5 * vv + 0.3 * eg + 0.2 * ce
        rows.append({
            "article": article, "hype_score": round(hype, 3),
            "view_velocity": round(vv, 3), "edit_growth": round(eg, 3),
            "content_expansion": round(ce, 3), "is_trending": hype > 0.6,
        })
    return pd.DataFrame(rows)


@st.cache_data
def gen_reputation_scores() -> pd.DataFrame:
    random.seed(13)
    rows = []
    for article in ARTICLES:
        ev = random.uniform(0.5, 15.0)
        vr = random.uniform(1.0, 30.0)
        anon = random.uniform(5.0, 60.0)
        risk = 0.3 * (ev / 15) + 0.4 * (vr / 30) + 0.3 * (anon / 60)
        level = "high" if risk > 0.6 else ("medium" if risk > 0.35 else "low")
        rows.append({
            "article": article, "risk_score": round(risk, 3),
            "edit_velocity": round(ev, 2), "vandalism_rate": round(vr, 2),
            "anonymous_edit_pct": round(anon, 2), "alert_level": level,
        })
    return pd.DataFrame(rows)


@st.cache_data
def gen_clusters() -> pd.DataFrame:
    random.seed(99)
    cluster_map = {
        "Artificial Intelligence": 0, "Machine Learning": 0, "ChatGPT": 0,
        "Quantum Computing": 1, "Nuclear Fusion": 1,
        "Climate Change": 2, "Renewable Energy": 2, "Electric Vehicles": 2,
        "CRISPR": 3, "Vaccine": 3,
        "Blockchain": 4, "Metaverse": 4, "Cybersecurity": 4,
        "SpaceX": 5, "Autonomous Vehicles": 5,
    }
    cluster_labels = {0: "AI & ML", 1: "Physics", 2: "Climate & Energy",
                      3: "Biotech", 4: "Digital", 5: "Mobility"}
    rows = []
    for article in ARTICLES:
        cid = cluster_map[article]
        rows.append({
            "article": article, "cluster_id": cid,
            "cluster_label": cluster_labels[cid],
            "growth_rate": round(random.uniform(2.0, 35.0), 2),
            "cagr": round(random.uniform(1.0, 20.0), 2),
            "total_views": random.randint(500_000, 10_000_000),
            "is_emerging": random.random() > 0.6,
            "confidence": round(random.uniform(0.6, 0.99), 3),
        })
    return pd.DataFrame(rows)


@st.cache_data
def gen_forecast(article: str, days_ahead: int = 90) -> pd.DataFrame:
    random.seed(hash(article) % 1000)
    base = datetime(2024, 1, 1)
    base_val = random.randint(10_000, 60_000)
    rows = []
    for d in range(days_ahead):
        dt = base + timedelta(days=d)
        trend = base_val * (1 + d * 0.002)
        noise = random.gauss(0, base_val * 0.05)
        yhat = max(0, trend + noise)
        rows.append({"date": dt, "yhat": yhat,
                     "yhat_lower": yhat * 0.85, "yhat_upper": yhat * 1.15})
    return pd.DataFrame(rows)


@st.cache_data
def gen_kg_centrality() -> pd.DataFrame:
    random.seed(55)
    cluster_map = gen_clusters().set_index("article")["cluster_id"].to_dict()
    rows = []
    for article in ARTICLES:
        rows.append({
            "article": article,
            "pagerank": round(random.uniform(0.01, 0.12), 4),
            "betweenness": round(random.uniform(0.0, 0.4), 4),
            "degree": random.randint(3, 25),
            "community_id": cluster_map.get(article, 0),
        })
    return pd.DataFrame(rows)


# -- sidebar -------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/wikipedia.png", width=60)
st.sidebar.title("Wikipedia Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Pageviews Analytics", "Hype Detection",
     "Reputation Monitor", "Topic Clusters", "Forecasting", "Knowledge Graph"],
)

selected_articles = st.sidebar.multiselect("Articles", ARTICLES, default=ARTICLES[:8])
if not selected_articles:
    selected_articles = ARTICLES[:8]

pv_df = gen_pageviews()
hype_df = gen_hype_scores()
rep_df = gen_reputation_scores()
cluster_df = gen_clusters()
kg_df = gen_kg_centrality()

pv_filtered = pv_df[pv_df["article"].isin(selected_articles)]
hype_filtered = hype_df[hype_df["article"].isin(selected_articles)]
rep_filtered = rep_df[rep_df["article"].isin(selected_articles)]
cluster_filtered = cluster_df[cluster_df["article"].isin(selected_articles)]
kg_filtered = kg_df[kg_df["article"].isin(selected_articles)]


def kpi(col, label, value, delta=None, fmt="{:.3f}"):
    col.metric(label, fmt.format(value) if isinstance(value, float) else value,
               delta=f"{delta:+.2f}" if delta is not None else None)


# -- OVERVIEW ------------------------------------------------------------------
if page == "Overview":
    st.title("📡 Wikipedia Intelligence Platform")
    st.caption(f"Demo dataset · {len(selected_articles)} articles monitored")
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Articles Tracked", len(selected_articles), fmt="{}")
    kpi(c2, "Avg Hype Score", hype_filtered["hype_score"].mean())
    kpi(c3, "Trending Articles", int(hype_filtered["is_trending"].sum()), fmt="{}")
    kpi(c4, "High-Risk Articles", int((rep_filtered["alert_level"] == "high").sum()), fmt="{}")
    kpi(c5, "Avg Risk Score", rep_filtered["risk_score"].mean())
    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Hype Score by Article")
        fig = px.bar(hype_filtered.sort_values("hype_score", ascending=True),
                     x="hype_score", y="article", orientation="h",
                     color="hype_score", color_continuous_scale="YlOrRd",
                     labels={"hype_score": "Hype Score (0-1)", "article": ""},
                     template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.subheader("Reputation Risk by Article")
        color_map = {"low": "#00CC96", "medium": "#FFA15A", "high": "#EF553B"}
        fig = px.bar(rep_filtered.sort_values("risk_score", ascending=True),
                     x="risk_score", y="article", orientation="h",
                     color="alert_level", color_discrete_map=color_map,
                     labels={"risk_score": "Risk Score (0-1)", "article": ""},
                     template="plotly_white")
        fig.update_layout(height=380, legend_title_text="Alert Level")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Total Daily Pageviews (all selected articles)")
    daily = pv_filtered.groupby("date")["views_total"].sum().reset_index()
    fig = px.area(daily, x="date", y="views_total",
                  labels={"views_total": "Total Views", "date": ""},
                  template="plotly_white", color_discrete_sequence=["#636EFA"])
    fig.update_layout(height=260)
    st.plotly_chart(fig, use_container_width=True)


# -- PAGEVIEWS ANALYTICS -------------------------------------------------------
elif page == "Pageviews Analytics":
    st.title("📊 Pageviews Analytics")
    st.caption("Daily pageview breakdown by article and device type")
    st.markdown("---")

    device_filter = st.sidebar.multiselect("Device Types", DEVICE_TYPES, default=DEVICE_TYPES)
    pv_dev = pv_filtered[pv_filtered["device_type"].isin(device_filter)]

    c1, c2, c3, c4 = st.columns(4)
    total_views = pv_dev["views_total"].sum()
    human_views = pv_dev["views_human"].sum()
    bot_views = pv_dev["views_bot"].sum()
    bot_pct = bot_views / total_views * 100 if total_views > 0 else 0
    c1.metric("Total Views", f"{total_views:,.0f}")
    c2.metric("Human Views", f"{human_views:,.0f}")
    c3.metric("Bot Views", f"{bot_views:,.0f}")
    c4.metric("Bot Traffic %", f"{bot_pct:.1f}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Daily Views by Article")
        daily_art = pv_dev.groupby(["date", "article"])["views_total"].sum().reset_index()
        fig = px.line(daily_art, x="date", y="views_total", color="article",
                      labels={"views_total": "Views", "date": ""},
                      template="plotly_white")
        fig.update_layout(height=380, legend_title_text="Article")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Device Type Share")
        dev_share = pv_dev.groupby("device_type")["views_total"].sum().reset_index()
        fig = px.pie(dev_share, names="device_type", values="views_total",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     template="plotly_white")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Human vs Bot Traffic Over Time")
    daily_hb = pv_dev.groupby("date")[["views_human", "views_bot"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_hb["date"], y=daily_hb["views_human"],
                             name="Human", fill="tozeroy", line_color="#636EFA"))
    fig.add_trace(go.Scatter(x=daily_hb["date"], y=daily_hb["views_bot"],
                             name="Bot", fill="tozeroy", line_color="#EF553B"))
    fig.update_layout(template="plotly_white", height=280,
                      xaxis_title="", yaxis_title="Views")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Articles by Total Views")
    art_totals = pv_dev.groupby("article")["views_total"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(art_totals, x="article", y="views_total",
                 color="views_total", color_continuous_scale="Blues",
                 labels={"views_total": "Total Views", "article": ""},
                 template="plotly_white")
    fig.update_layout(coloraxis_showscale=False, height=300, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


# -- HYPE DETECTION ------------------------------------------------------------
elif page == "Hype Detection":
    st.title("🔥 Hype Detection")
    st.caption("Composite hype score = 0.5×view_velocity + 0.3×edit_growth + 0.2×content_expansion")
    st.markdown("---")

    trending = hype_filtered[hype_filtered["is_trending"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Trending Articles", len(trending))
    c2.metric("Avg Hype Score", f"{hype_filtered['hype_score'].mean():.3f}")
    c3.metric("Max Hype Score", f"{hype_filtered['hype_score'].max():.3f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Hype Score Breakdown")
        melt = hype_filtered.melt(
            id_vars="article",
            value_vars=["view_velocity", "edit_growth", "content_expansion"],
            var_name="component", value_name="score",
        )
        fig = px.bar(melt, x="score", y="article", color="component",
                     orientation="h", barmode="stack",
                     labels={"score": "Component Score", "article": ""},
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     template="plotly_white")
        fig.update_layout(height=400, legend_title_text="Component")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Hype vs Edit Growth Scatter")
        fig = px.scatter(hype_filtered, x="edit_growth", y="hype_score",
                         size="view_velocity", color="is_trending",
                         hover_name="article",
                         color_discrete_map={True: "#EF553B", False: "#636EFA"},
                         labels={"edit_growth": "Edit Growth", "hype_score": "Hype Score"},
                         template="plotly_white")
        fig.add_hline(y=0.6, line_dash="dash", line_color="gray",
                      annotation_text="Trending threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    if not trending.empty:
        st.subheader("🚨 Trending Articles")
        for _, row in trending.sort_values("hype_score", ascending=False).iterrows():
            with st.expander(f"**{row['article']}** — Hype: {row['hype_score']:.3f}"):
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("View Velocity", f"{row['view_velocity']:.3f}")
                cc2.metric("Edit Growth", f"{row['edit_growth']:.3f}")
                cc3.metric("Content Expansion", f"{row['content_expansion']:.3f}")

    st.subheader("Hype Score Table")
    st.dataframe(
        hype_filtered.sort_values("hype_score", ascending=False),
        use_container_width=True,
    )


# -- REPUTATION MONITOR --------------------------------------------------------
elif page == "Reputation Monitor":
    st.title("🛡️ Reputation Monitor")
    st.caption("Edit velocity, vandalism rate, and anonymous edit percentage")
    st.markdown("---")

    color_map = {"low": "#00CC96", "medium": "#FFA15A", "high": "#EF553B"}
    high_risk = rep_filtered[rep_filtered["alert_level"] == "high"]
    med_risk = rep_filtered[rep_filtered["alert_level"] == "medium"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High Risk", len(high_risk))
    c2.metric("Medium Risk", len(med_risk))
    c3.metric("Avg Vandalism Rate", f"{rep_filtered['vandalism_rate'].mean():.2f}")
    c4.metric("Avg Anon Edit %", f"{rep_filtered['anonymous_edit_pct'].mean():.1f}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Risk Score by Article")
        fig = px.bar(rep_filtered.sort_values("risk_score", ascending=True),
                     x="risk_score", y="article", orientation="h",
                     color="alert_level", color_discrete_map=color_map,
                     labels={"risk_score": "Risk Score", "article": ""},
                     template="plotly_white")
        fig.update_layout(height=400, legend_title_text="Alert Level")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Vandalism Rate vs Edit Velocity")
        fig = px.scatter(rep_filtered, x="edit_velocity", y="vandalism_rate",
                         color="alert_level", size="anonymous_edit_pct",
                         hover_name="article",
                         color_discrete_map=color_map,
                         labels={"edit_velocity": "Edit Velocity (edits/day)",
                                 "vandalism_rate": "Vandalism Rate (%)"},
                         template="plotly_white")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anonymous Edit % by Article")
    fig = px.bar(rep_filtered.sort_values("anonymous_edit_pct", ascending=False),
                 x="article", y="anonymous_edit_pct",
                 color="alert_level", color_discrete_map=color_map,
                 labels={"anonymous_edit_pct": "Anonymous Edit %", "article": ""},
                 template="plotly_white")
    fig.add_hline(y=rep_filtered["anonymous_edit_pct"].mean(), line_dash="dash",
                  annotation_text="Average")
    fig.update_layout(height=300, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    if not high_risk.empty:
        st.error(f"⚠️ {len(high_risk)} article(s) at HIGH risk require immediate review.")
        st.dataframe(high_risk[["article", "risk_score", "edit_velocity",
                                 "vandalism_rate", "anonymous_edit_pct"]],
                     use_container_width=True)


# -- TOPIC CLUSTERS ------------------------------------------------------------
elif page == "Topic Clusters":
    st.title("🗂️ Topic Clusters")
    st.caption("Semantic clusters with growth rates and CAGR")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", cluster_filtered["cluster_id"].nunique())
    c2.metric("Emerging Topics", int(cluster_filtered["is_emerging"].sum()))
    c3.metric("Avg CAGR", f"{cluster_filtered['cagr'].mean():.1f}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Cluster Size & Growth")
        cluster_agg = cluster_filtered.groupby("cluster_label").agg(
            articles=("article", "count"),
            avg_growth=("growth_rate", "mean"),
            avg_cagr=("cagr", "mean"),
            total_views=("total_views", "sum"),
        ).reset_index()
        fig = px.scatter(cluster_agg, x="avg_growth", y="avg_cagr",
                         size="total_views", color="cluster_label",
                         hover_data=["articles"],
                         labels={"avg_growth": "Avg Growth Rate (%)",
                                 "avg_cagr": "Avg CAGR (%)",
                                 "cluster_label": "Cluster"},
                         template="plotly_white")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Total Views by Cluster")
        fig = px.pie(cluster_agg, names="cluster_label", values="total_views",
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     template="plotly_white")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Article Growth Rates")
    fig = px.bar(cluster_filtered.sort_values("growth_rate", ascending=False),
                 x="article", y="growth_rate", color="cluster_label",
                 labels={"growth_rate": "Growth Rate (%)", "article": ""},
                 template="plotly_white")
    fig.update_layout(height=320, xaxis_tickangle=-30, legend_title_text="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Emerging Topics")
    emerging = cluster_filtered[cluster_filtered["is_emerging"]]
    if emerging.empty:
        st.info("No emerging topics in current selection.")
    else:
        st.success(f"{len(emerging)} emerging topic(s) detected.")
        st.dataframe(emerging[["article", "cluster_label", "growth_rate", "cagr",
                                "total_views", "confidence"]]
                     .sort_values("growth_rate", ascending=False),
                     use_container_width=True)


# -- FORECASTING ---------------------------------------------------------------
elif page == "Forecasting":
    st.title("📈 Forecasting")
    st.caption("90-day pageview forecast with prediction intervals")
    st.markdown("---")

    forecast_article = st.selectbox("Select Article", selected_articles)
    horizon = st.slider("Forecast Horizon (days)", 30, 180, 90)

    fc_df = gen_forecast(forecast_article, days_ahead=horizon)

    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast Start", fc_df["date"].min().strftime("%Y-%m-%d"))
    c2.metric("Forecast End", fc_df["date"].max().strftime("%Y-%m-%d"))
    c3.metric("Avg Predicted Views", f"{fc_df['yhat'].mean():,.0f}")

    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["yhat_upper"],
        mode="lines", line=dict(width=0), name="Upper 95%",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["yhat_lower"],
        mode="lines", line=dict(width=0), name="Lower 95%",
        fill="tonexty", fillcolor="rgba(99,110,250,0.15)",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["yhat"],
        mode="lines", name="Forecast",
        line=dict(color="#636EFA", width=2.5),
    ))
    fig.update_layout(
        title=f"Pageview Forecast — {forecast_article}",
        xaxis_title="", yaxis_title="Predicted Views",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All Articles — Forecast Summary")
    summary_rows = []
    for art in selected_articles:
        fc = gen_forecast(art, days_ahead=90)
        summary_rows.append({
            "article": art,
            "avg_forecast": round(fc["yhat"].mean(), 0),
            "min_forecast": round(fc["yhat"].min(), 0),
            "max_forecast": round(fc["yhat"].max(), 0),
            "growth_pct": round((fc["yhat"].iloc[-1] - fc["yhat"].iloc[0]) / fc["yhat"].iloc[0] * 100, 1),
        })
    summary_df = pd.DataFrame(summary_rows)
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(summary_df.sort_values("avg_forecast", ascending=False),
                     x="article", y="avg_forecast",
                     color="growth_pct", color_continuous_scale="RdYlGn",
                     labels={"avg_forecast": "Avg Forecast Views", "article": ""},
                     template="plotly_white")
        fig.update_layout(height=320, xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.dataframe(
            summary_df.sort_values("growth_pct", ascending=False),
            use_container_width=True,
        )


# -- KNOWLEDGE GRAPH -----------------------------------------------------------
elif page == "Knowledge Graph":
    st.title("🕸️ Knowledge Graph")
    st.caption("Article centrality, community detection, and network topology")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles", len(kg_filtered))
    c2.metric("Communities", kg_filtered["community_id"].nunique())
    c3.metric("Avg PageRank", f"{kg_filtered['pagerank'].mean():.4f}")
    c4.metric("Avg Betweenness", f"{kg_filtered['betweenness'].mean():.4f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("PageRank by Article")
        fig = px.bar(kg_filtered.sort_values("pagerank", ascending=True),
                     x="pagerank", y="article", orientation="h",
                     color="pagerank", color_continuous_scale="Purples",
                     labels={"pagerank": "PageRank", "article": ""},
                     template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Betweenness vs PageRank")
        fig = px.scatter(kg_filtered, x="pagerank", y="betweenness",
                         size="degree", color="community_id",
                         hover_name="article",
                         color_continuous_scale="Turbo",
                         labels={"pagerank": "PageRank",
                                 "betweenness": "Betweenness Centrality",
                                 "community_id": "Community"},
                         template="plotly_white")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Network Graph (force-directed layout)")
    n = len(kg_filtered)
    angles = [2 * math.pi * i / n for i in range(n)]
    xs = [math.cos(a) * (1 + kg_filtered.iloc[i]["pagerank"] * 10) for i, a in enumerate(angles)]
    ys = [math.sin(a) * (1 + kg_filtered.iloc[i]["pagerank"] * 10) for i, a in enumerate(angles)]

    edge_x, edge_y = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if kg_filtered.iloc[i]["community_id"] == kg_filtered.iloc[j]["community_id"]:
                edge_x += [xs[i], xs[j], None]
                edge_y += [ys[i], ys[j], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.8, color="#ccc"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(
            size=[8 + kg_filtered.iloc[i]["degree"] * 1.5 for i in range(n)],
            color=kg_filtered["community_id"].tolist(),
            colorscale="Turbo", showscale=True,
            colorbar=dict(title="Community"),
            line=dict(width=1, color="white"),
        ),
        text=kg_filtered["article"].tolist(),
        textposition="top center",
        textfont=dict(size=8),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "PageRank: %{customdata[0]:.4f}<br>"
            "Betweenness: %{customdata[1]:.4f}<br>"
            "Degree: %{customdata[2]}"
        ),
        customdata=kg_filtered[["pagerank", "betweenness", "degree"]].values,
    ))
    fig.update_layout(
        showlegend=False, height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Centrality Table")
    st.dataframe(
        kg_filtered.sort_values("pagerank", ascending=False),
        use_container_width=True,
    )
